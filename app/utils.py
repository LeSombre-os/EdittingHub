"""Fonctions utilitaires partagées par tous les services.

Gestion des timecodes, formatage, parsing de la sortie des scripts,
et helpers de conversion pour l'export EDL/CSV/XML.
"""
import math
import os
import re
from datetime import datetime


# ── Formatage de durée / taille ────────────────────────────────────────────

def fmt_duration(seconds):
    """Formatte une durée en secondes → 'H:MM:SS' ou 'M:SS'."""
    s = max(0, int(round(seconds or 0)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_size(n):
    """Formatte une taille en octets → '1,2 Go' / '500 Mo' / '12 Ko'."""
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1000 or unit == "Go":
            if unit != "o":
                txt = f"{n:.1f}".replace(".", ",")
            else:
                txt = f"{n:d}"
            return f"{txt} {unit}"
        n /= 1000.0


# ── Conversion timecode (secondes ↔ SMPTE HH:MM:SS:FF) ─────────────────────

def seconds_to_srt(seconds):
    """Convertit des secondes en timestamp SRT : 00:00:05,234."""
    t = max(0.0, seconds)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_frames(seconds, fps=24):
    # Arrondi round-half-up, identique à la conversion du script DaVinci.
    fps = float(fps)
    return max(0, int(math.floor(float(seconds) * fps + 0.5)))


def seconds_to_edl_timecode(seconds, fps=24):
    """Convertit des secondes en timecode EDL : HH:MM:SS:FF.

    Utilise le fps fourni (24 par défaut pour le cinéma).
    """
    fps = int(round(fps))
    total_frames = seconds_to_frames(seconds, fps)
    h = total_frames // (fps * 3600)
    m = (total_frames // (fps * 60)) % 60
    s = (total_frames // fps) % 60
    f = int(total_frames % fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def edl_timecode_to_seconds(tc, fps=24):
    """Convertit un timecode EDL (HH:MM:SS:FF) en secondes."""
    parts = tc.split(":")
    if len(parts) != 4:
        return 0.0
    h, m, s, f = [int(x) for x in parts]
    return h * 3600 + m * 60 + s + f / fps


# ── Navigation de fichiers ─────────────────────────────────────────────────

DEFAULT_ROOTS = [
    "/media",
    "/home",
    os.environ.get("HOME", ""),
]


def get_default_roots():
    """Retourne les racines de navigation (points de départ du browser de fichiers)."""
    user = os.environ.get("USER", "")
    roots = []
    # Racines utiles sur un système Linux
    home = os.environ.get("HOME", "")
    if home and os.path.isdir(home):
        roots.append(home)
    if user and os.path.isdir(f"/media/{user}"):
        roots.append(f"/media/{user}")
    if os.path.isdir("/media"):
        roots.append("/media")
    roots.append("/")
    return list(dict.fromkeys(roots))


def list_dir(path):
    """Liste les fichiers/dossiers d'un chemin, retourne un dict structuré."""
    path = os.path.abspath(path or get_default_roots()[0])
    if not os.path.isdir(path):
        path = get_default_roots()[0]
    entries = []
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    isdir = e.is_dir(follow_symlinks=True)
                    entries.append({
                        "name": e.name,
                        "path": e.path,
                        "isdir": isdir,
                        "size": 0 if isdir else e.stat().st_size,
                    })
                except OSError:
                    continue
    except OSError:
        pass
    entries.sort(key=lambda e: (not e["isdir"], e["name"].lower()))
    parent = os.path.dirname(path) if path != "/" else ""
    return {"path": path, "parent": parent, "entries": entries,
            "roots": get_default_roots()}


def collect_files(paths, recursive=True, extensions=None):
    """Collecte des fichiers à partir de chemins (fichiers ou dossiers).

    Filtre optionnel par extension. Déduplit en conservant l'ordre.
    """
    VIDEO_EXTS = {".mp4", ".m4v", ".mkv", ".avi", ".mov", ".mts", ".m2ts",
                  ".mxf", ".ts", ".vob", ".webm"}
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga",
                  ".opus", ".wma", ".aiff", ".aif"}
    ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS

    exts = extensions if extensions is not None else ALL_EXTS

    files = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in exts:
                files.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                for n in sorted(names):
                    if os.path.splitext(n)[1].lower() in exts:
                        files.append(os.path.join(root, n))
                if not recursive:
                    break
    return list(dict.fromkeys(files))


# ── Parsing de la sortie des scripts subprocess ────────────────────────────

def parse_progress_line(line, total=None):
    """Parse une ligne de sortie pour en extraire un pourcentage de progression.

    Formats supportés :
    - "[3/5] ..." → done=3, total=5
    - "Progression: 45%" → frac=0.45
    """
    # Format [done/total]
    m = re.match(r"\[(\d+)/(\d+)\]", line.strip())
    if m:
        done = int(m.group(1))
        tot = int(m.group(2))
        return {"done": done, "total": tot,
                "frac": done / tot if tot else 0.0}
    # Format "X%" n'importe où dans la ligne
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    if m:
        pct = float(m.group(1))
        return {"frac": pct / 100.0}
    return None


def now_iso():
    """Retourne l'horodatage ISO 8601 courant."""
    return datetime.now().isoformat(timespec="seconds")
