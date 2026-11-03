"""Runtime : budget CPU, priorité des jobs, sondes matérielles.

Objectif : que les traitements lourds (librosa, scenedetect, ffmpeg, whisper)
aillent vite **sans** prendre toute la machine. Trois leviers :

1. **Budget de threads** — numpy/OpenBLAS/OpenMP/MKL sont limités à
   `THREAD_BUDGET` threads *avant* le premier import de numpy. Sans cette
   limite, un `onset_strength` se répartit sur les 12 threads et le bureau
   devient inutilisable pendant l'analyse.
2. **Priorité** — les sous-processus (ffmpeg, whisper) sont lancés via
   `nice -n NICE_LEVEL` : l'interface et le reste du système passent devant.
3. **Sondes** — CPU (charge + %) et GPU AMD (via sysfs, pas de rocm-smi
   requis) pour afficher « ce job utilise CPU / GPU » et savoir quoi lancer
   en parallèle.

Tout est local (lecture de /proc et /sys), aucune dépendance.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# ── Détection CPU ──────────────────────────────────────────────────────────
CPU_COUNT = os.cpu_count() or 4


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ[name]))
    except (KeyError, TypeError, ValueError):
        return default


# Budget par défaut : on laisse au moins 2 cœurs au système (interface,
# navigateur, Resolve). Surchargeable par EDITTINGHUB_THREADS.
DEFAULT_BUDGET = max(2, min(CPU_COUNT - 2, 8)) if CPU_COUNT > 4 else max(1, CPU_COUNT - 1)
THREAD_BUDGET = _env_int("EDITTINGHUB_THREADS", DEFAULT_BUDGET)

# Priorité des sous-processus (0 = normal, 19 = le plus bas). 10 = confortable :
# ffmpeg reste à pleine vitesse quand la machine est libre, mais cède le pas
# dès que Firefox/Resolve demandent du CPU.
NICE_LEVEL = _env_int("EDITTINGHUB_NICE", 10)

# Nombre de jobs lourds simultanés autorisés (au-delà : mise en attente).
MAX_HEAVY_JOBS = _env_int("EDITTINGHUB_MAX_JOBS", 2)

_LIMITS_APPLIED = False
_LIMITS_LOCK = threading.Lock()

# Variables lues par les bibliothèques natives au moment de leur chargement.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply_thread_limits(force: bool = False) -> dict:
    """Limite les threads des bibliothèques natives. À appeler tôt.

    Idempotent. Retourne l'état appliqué (utile pour /api/system).
    """
    global _LIMITS_APPLIED
    with _LIMITS_LOCK:
        if _LIMITS_APPLIED and not force:
            return thread_state()
        budget = str(THREAD_BUDGET)
        for var in _THREAD_ENV:
            # On respecte une valeur déjà fixée par l'utilisateur.
            os.environ.setdefault(var, budget)
        # OpenCV : pas de pool de threads interne (scenedetect décode en
        # série, un pool ne ferait que voler des cœurs au système).
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        _LIMITS_APPLIED = True

    # threadpoolctl limite aussi les BLAS déjà chargés (numpy importé plus
    # tôt par un autre module). Optionnel : absence = pas bloquant.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=THREAD_BUDGET)
    except Exception:  # noqa: BLE001 — threadpoolctl facultatif
        pass
    return thread_state()


def thread_state() -> dict:
    return {
        "cpu_count": CPU_COUNT,
        "budget": THREAD_BUDGET,
        "nice": NICE_LEVEL,
        "max_heavy_jobs": MAX_HEAVY_JOBS,
        "env": {v: os.environ.get(v) for v in _THREAD_ENV if os.environ.get(v)},
    }


def nice_prefix() -> list[str]:
    """Préfixe `nice` pour les sous-processus (vide si nice absent)."""
    return ["nice", "-n", str(NICE_LEVEL)] if NICE_LEVEL > 0 else []


# ── Sonde CPU ──────────────────────────────────────────────────────────────
_cpu_lock = threading.Lock()
_cpu_prev: tuple[float, float] | None = None  # (idle, total)


def _read_cpu_totals() -> tuple[float, float]:
    """Retourne (idle_jiffies, total_jiffies) depuis /proc/stat."""
    try:
        with open("/proc/stat", "r", encoding="ascii") as fh:
            parts = fh.readline().split()[1:]
        values = [float(p) for p in parts[:8]]
        idle = values[3] + (values[4] if len(values) > 4 else 0.0)  # idle + iowait
        return idle, sum(values)
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0


def cpu_percent() -> float | None:
    """Charge CPU globale (%) depuis le dernier appel (None au 1er appel)."""
    global _cpu_prev
    idle, total = _read_cpu_totals()
    with _cpu_lock:
        prev, _cpu_prev = _cpu_prev, (idle, total)
    if not prev or total <= prev[1]:
        return None
    d_idle = idle - prev[0]
    d_total = total - prev[1]
    if d_total <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total))), 1)


def load_average() -> list[float]:
    try:
        with open("/proc/loadavg", "r", encoding="ascii") as fh:
            parts = fh.read().split()
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except (OSError, ValueError, IndexError):
        return [0.0, 0.0, 0.0]


# ── Sonde mémoire ──────────────────────────────────────────────────────────
def memory() -> dict:
    """Mémoire système et swap (Kio, comme /proc/meminfo)."""
    data: dict[str, dict] = {}
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                info[key.strip()] = int(val.split()[0])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        if total:
            data["ram"] = {
                "total": total,
                "used": total - avail,
                "percent": round((total - avail) / total * 100, 1),
            }
        swap_total = info.get("SwapTotal", 0)
        if swap_total:
            used = swap_total - info.get("SwapFree", 0)
            data["swap"] = {
                "total": swap_total,
                "used": used,
                "percent": round(used / swap_total * 100, 1),
            }
    except (OSError, ValueError):
        pass
    return data


# ── Sonde GPU (AMD via sysfs, NVIDIA via nvidia-smi) ───────────────────────
_DRM = Path("/sys/class/drm")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def gpu_stats() -> list[dict]:
    """GPU détectés et leur charge. Vide si aucun GPU interrogeable.

    AMD : `gpu_busy_percent` + `mem_info_vram_*` exposés par l'amdgpu du noyau
    (aucun besoin de rocm-smi). NVIDIA : repli nvidia-smi.
    """
    gpus: list[dict] = []
    try:
        cards = sorted(_DRM.glob("card[0-9]"))
    except OSError:
        cards = []
    for card in cards:
        device = card / "device"
        busy = _read_int(device / "gpu_busy_percent")
        if busy is None:
            continue  # carte sans compteur (ex. carte virtuelle, VGA non AMD)
        vram_used = _read_int(device / "mem_info_vram_used")
        vram_total = _read_int(device / "mem_info_vram_total")
        temp = None
        for sensor in sorted(device.glob("hwmon/hwmon*/temp1_input")):
            raw = _read_int(sensor)
            if raw is not None:
                temp = round(raw / 1000.0, 1)
                break
        name = None
        try:
            for line in (device / "uevent").read_text(encoding="utf-8").splitlines():
                if line.startswith("DRIVER="):
                    name = line.split("=", 1)[1]
        except OSError:
            pass
        gpus.append({
            "vendor": "amd",
            "name": name or card.name,
            "busy": busy,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "vram_percent": (round(vram_used / vram_total * 100, 1)
                             if vram_used and vram_total else None),
            "temp": temp,
        })
    if gpus:
        return gpus
    # Repli NVIDIA
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "vendor": "nvidia", "name": parts[0],
                        "busy": int(float(parts[1])),
                        "vram_used": int(float(parts[2])) * 1024 * 1024,
                        "vram_total": int(float(parts[3])) * 1024 * 1024,
                        "vram_percent": round(float(parts[2]) / max(1.0, float(parts[3])) * 100, 1),
                        "temp": int(float(parts[4])),
                    })
    except Exception:  # noqa: BLE001 — nvidia-smi absent
        pass
    return gpus


def gpu_available() -> bool:
    return bool(gpu_stats())


_WHISPER_GPU: bool | None = None
_WHISPER_LOCK = threading.Lock()


def whisper_gpu_available() -> bool:
    """Le sous-titrage peut-il utiliser le GPU (torch ROCm dans tts-rocm) ?

    Sonde en sous-processus (jamais d'import torch ici : torch ROCm +
    faster-whisper dans le même processus = crash natif). Résultat mémorisé.
    """
    global _WHISPER_GPU
    with _WHISPER_LOCK:
        if _WHISPER_GPU is not None:
            return _WHISPER_GPU
        base = Path(os.path.expanduser("~/.venvs/tts-rocm"))
        py = base / "bin" / "python"
        has_whisper = any(base.glob("lib/python3*/site-packages/whisper/__init__.py"))
        if py.is_file() and has_whisper:
            try:
                import subprocess

                r = subprocess.run(
                    [str(py), "-c", "import torch;print(torch.cuda.is_available())"],
                    capture_output=True, text=True, timeout=120,
                )
                _WHISPER_GPU = r.stdout.strip().endswith("True")
            except Exception:  # noqa: BLE001
                _WHISPER_GPU = False
        else:
            _WHISPER_GPU = False
        return _WHISPER_GPU


def system_snapshot() -> dict:
    """Photo complète pour /api/system et la barre d'état de l'UI."""
    mem = memory()
    loads = load_average()
    gpus = gpu_stats()
    return {
        "cpu": {
            "percent": cpu_percent(),
            "count": CPU_COUNT,
            "budget": THREAD_BUDGET,
            "load": {"1min": loads[0], "5min": loads[1], "15min": loads[2]},
            "load_percent": round(min(100.0, loads[0] / max(1, CPU_COUNT) * 100), 1),
        },
        "memory": mem.get("ram"),
        "swap": mem.get("swap"),
        "gpu": gpus,
        "gpu_detected": bool(gpus),
        "whisper_gpu": _WHISPER_GPU,
        "nice": NICE_LEVEL,
        "timestamp": round(time.time(), 3),
    }


# Applique la limite dès l'import : le serveur importe ce module avant
# librosa / numpy / cv2.
apply_thread_limits()
