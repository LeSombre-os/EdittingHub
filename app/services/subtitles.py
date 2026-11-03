"""Hub de Sous-titrage — redirige vers ResolveSubs/transcrire.py par sous-processus.

Ce service wrappe le script de transcription existant (faster-whisper) sans le
modifier. Il orchestre :
  1. un *scan* (probe, estimation de durée) via --estimate ;
  2. un *run* (transcription) exécuté dans un sous-processus avec suivi
     de progression et annulation.

Le sous-processus utilise le venv dédié resolvesubs (ou le python système en
repli) pour garantir l'accès à faster-whisper.
"""
import json
import os
import re
import subprocess
import threading
import time

from app import procs
from app.config import SUBTITLES_PYTHON, SUBTITLES_PYTHON_GPU, SUBTITLES_SCRIPT
from app.utils import fmt_duration, parse_progress_line


# Regex pour parser la sortie de transcrire.py
_RE_SUMMARY = re.compile(
    # transcrire.py imprime : « Terminé en 0:03 : 1 transcrits, 0 déjà OK, 0 échecs. »
    r"Terminé.*?: (\d+) transcrits?, (\d+) déjà OK, (\d+) échecs"
)
_RE_HEADER = re.compile(
    r"ResolveSubs\s*—\s*(\d+) fichier\(s\), langue=(\w+), modèle=(\w+)"
)


class SubtitleService:
    """Orchestration des sous-titres via ResolveSubs."""

    # Modèles whisper disponibles
    MODELS = ("base", "small", "medium", "turbo", "large-v3")
    LANGS = ("auto", "fr", "en")

    def __init__(self):
        self.script = str(SUBTITLES_SCRIPT)
        # Détermine le bon python (venv resolvesubs ou système)
        self.python = str(SUBTITLES_PYTHON) if os.path.isfile(str(SUBTITLES_PYTHON)) else "python3"
        # Python GPU (venv ROCm : torch + openai-whisper) ou None
        self.python_gpu = (str(SUBTITLES_PYTHON_GPU)
                           if SUBTITLES_PYTHON_GPU and os.path.isfile(str(SUBTITLES_PYTHON_GPU))
                           else None)
        self._proc = None
        self._lock = threading.Lock()
        self._backend_cache: dict[str, bool] = {}

    # ── Annulation externe ─────────────────────────────────────────────────
    def terminate(self):
        """Tue la transcription en cours et tous ses enfants.

        Important : whisper charge le modèle en mémoire. Tuer le groupe libère
        la VRAM au lieu de laisser un processus fantôme l'occuper.
        """
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        return procs.kill_group(proc)

    # ── Vérification ───────────────────────────────────────────────────────
    def available(self):
        """Vérifie que le script de sous-titrage est présent."""
        return os.path.isfile(self.script)

    def _can_import(self, python: str, module: str) -> bool:
        """Ce python sait-il importer `module` ? (test isolé, mémorisé)

        On ne teste **pas** dans le processus du serveur : torch ROCm et
        faster-whisper dans le même processus = crash natif.
        """
        key = f"{python}|{module}"
        if key not in self._backend_cache:
            try:
                r = subprocess.run(
                    [python, "-c",
                     "import importlib.util as u;"
                     f"raise SystemExit(0 if u.find_spec('{module}') else 1)"],
                    capture_output=True, text=True, timeout=60,
                )
                self._backend_cache[key] = r.returncode == 0
            except Exception:  # noqa: BLE001
                self._backend_cache[key] = False
        return self._backend_cache[key]

    def resolve_backend(self, device: str = "auto") -> tuple[str, str, str]:
        """Choisit l'interpréteur et le device réels avant de lancer.

        Retourne (python, device_effectif, note) où `note` explique le choix
        (affichée dans le journal du job).

        Pourquoi : `transcrire.py --device auto` sélectionne le GPU d'après le
        venv ROCm, mais l'interpréteur lancé doit lui aussi contenir
        openai-whisper. Lancer le venv `resolvesubs` avec `--device auto`
        échouait immédiatement (« openai-whisper manquant »), d'où l'« erreur
        inconnue » côté interface.
        """
        if device == "cpu":
            return self.python, "cpu", "Transcription sur CPU (faster-whisper int8)."

        # GPU demandé ou automatique : il faut un python qui a openai-whisper.
        if self.python_gpu and self._can_import(self.python_gpu, "whisper"):
            return (self.python_gpu, "gpu",
                    "Transcription sur GPU (venv ROCm, openai-whisper).")

        if device == "gpu":
            raise RuntimeError(
                "GPU demandé mais aucun interpréteur avec openai-whisper "
                f"n'a été trouvé (attendu : {self.python_gpu or '~/.venvs/tts-rocm'}). "
                "Choisissez « CPU », ou installez openai-whisper dans le venv "
                "ROCm : ~/.venvs/tts-rocm/bin/pip install openai-whisper"
            )
        return (self.python, "cpu",
                "GPU indisponible pour la transcription → repli CPU "
                "(faster-whisper int8).")

    def _check_backend(self, device="auto", model="small"):
        """Vérifie que le backend requis (faster-whisper ou whisper) existe.

        Retourne (ok: bool, message: str).
        """
        if device != "cpu":
            if self.python_gpu and self._can_import(self.python_gpu, "whisper"):
                return True, "OK (GPU, openai-whisper)"
            if device == "gpu":
                return False, (
                    "GPU demandé mais openai-whisper est absent du venv ROCm "
                    "(~/.venvs/tts-rocm/bin/pip install openai-whisper)."
                )
        if self._can_import(self.python, "faster_whisper"):
            return True, "OK (CPU, faster-whisper)"
        return False, (
            "faster-whisper introuvable dans le venv resolvesubs. "
            "Installez-le : ~/.venvs/resolvesubs/bin/pip install faster-whisper"
        )

    # ── Scan (estimation sans transcription) ─────────────────────────────────
    def scan(self, paths, language="auto", model="small", offset=0.0):
        """Lance un scan d'estimation via --estimate.

        Retourne le plan JSON décodé.
        """
        if not self.available():
            raise RuntimeError(
                f"Script introuvable : {self.script}. "
                "Vérifiez que le projet ResolveSubs est présent."
            )
        cmd = [self.python, self.script, "--estimate",
               "--language", language, "--model", model]
        if offset:
            cmd.extend(["--offset", str(offset)])
        cmd.extend(paths)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=True
            )
            data = json.loads(result.stdout)
            return data
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Erreur de parsing JSON : {e}\nSortie : {result.stdout[-500:]}"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Le script de sous-titrage a échoué (code {e.returncode}) :\n"
                f"{e.stderr or e.stdout}"
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout du scan (120s).")
        except FileNotFoundError:
            raise RuntimeError(f"Python introuvable : {self.python}")

    # ── Exécution (transcription) ──────────────────────────────────────────
    def run(self, paths, language="auto", model="small", offset=0.0,
            device="auto", max_chars=42, max_lines=2, mode="phrases",
            words_per_cue=1, music=False, prompt=None, force=False,
            jobs=1, cancel_event=None, progress_callback=None):
        """Exécute la transcription via sous-processus.

        Args (voir transcrire.py pour la documentation complète) :
            paths : fichiers/dossiers à transcrire
            language : auto/fr/en
            model : base/small/medium/turbo/large-v3
            offset : décalage en secondes
            device : auto/gpu/cpu
            max_chars : caractères max par ligne
            max_lines : lignes max par sous-titre
            mode : phrases/mots
            words_per_cue : mots par sous-titre (mode mots)
            music : mode chanson
            prompt : indice de contexte
            force : retranscrire même si .srt existe
            jobs : fichiers en parallèle
            cancel_event : threading.Event pour annuler
            progress_callback : fonction(done, total, frac, line, stats)

        Returns:
            dict avec status, stats, logs, error (message lisible si échec).
        """
        if not self.available():
            raise RuntimeError(f"Script introuvable : {self.script}")

        # Interpréteur + device réels (GPU si le venv ROCm a openai-whisper).
        python, use_device, note = self.resolve_backend(device)
        ok, msg = self._check_backend(use_device, model)
        if not ok:
            raise RuntimeError(msg)

        result = self._execute(
            python=python, paths=paths, language=language, model=model,
            offset=offset, device=use_device, max_chars=max_chars,
            max_lines=max_lines, mode=mode, words_per_cue=words_per_cue,
            music=music, prompt=prompt, force=force, jobs=jobs,
            cancel_event=cancel_event, progress_callback=progress_callback,
            preamble=note,
        )

        # Repli automatique : un média déclaré « aucune parole » par le VAD
        # passe très souvent en mode musique (VAD désactivé). On retente une
        # fois pour éviter un échec incompréhensible côté utilisateur.
        if (not music and result["status"] == "error"
                and any("aucune parole" in l.lower() for l in result["logs"])
                and not (cancel_event is not None and cancel_event.is_set())):
            if progress_callback:
                progress_callback(0, 0, 0.0,
                                  "Aucune parole détectée → nouvelle tentative "
                                  "avec VAD désactivé (mode musique)…",
                                  result.get("stats"))
            retry = self._execute(
                python=python, paths=paths, language=language, model=model,
                offset=offset, device=use_device, max_chars=max_chars,
                max_lines=max_lines, mode=mode, words_per_cue=words_per_cue,
                music=True, prompt=prompt, force=True, jobs=jobs,
                cancel_event=cancel_event, progress_callback=progress_callback,
                preamble="[REPLI] Aucune parole avec le VAD → mode musique (VAD désactivé).",
            )
            retry["logs"] = (
                ["[REPLI] Première passe : aucune parole détectée par le VAD.",
                 "[REPLI] Seconde passe en mode musique (VAD désactivé)."]
                + retry["logs"]
            )
            retry["auto_retry"] = True
            result = retry

        return result

    def _execute(self, python, paths, language, model, offset, device,
                 max_chars, max_lines, mode, words_per_cue, music, prompt,
                 force, jobs, cancel_event, progress_callback,
                 preamble: str = ""):
        """Lance une passe de transcription et retourne le résultat détaillé."""
        cmd = [python, self.script]
        cmd.extend(paths)
        cmd.extend(["--language", language, "--model", model])
        cmd.extend(["--device", device])
        cmd.extend(["--max-chars", str(max_chars), "--max-lines", str(max_lines)])
        cmd.extend(["--mode", mode, "--words-per-cue",
                    str(max(1, min(10, words_per_cue)))])
        if offset:
            cmd.extend(["--offset", str(offset)])
        if music:
            cmd.append("--music")
        if prompt:
            cmd.extend(["--prompt", prompt])
        if force:
            cmd.append("--force")
        if jobs > 1:
            cmd.extend(["--jobs", str(jobs)])

        logs = [preamble] if preamble else []
        stats = {"ok": 0, "skip": 0, "fail": 0}
        done = 0
        total = 0

        try:
            proc = subprocess.Popen(
                procs.nice_cmd(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Python introuvable : {python}. "
                "Vérifiez que le venv resolvesubs est installé."
            )

        with self._lock:
            self._proc = proc

        cancelled = False

        def _on_cancel():
            nonlocal cancelled
            cancelled = True
            logs.append("[ANNULÉ] Arrêt demandé — transcription et enfants tués.")

        try:
            for line in procs.iter_lines(proc, cancel_event, on_cancel=_on_cancel,
                                         poll=0.25):
                line = line.rstrip("\n")
                logs.append(line)

                # Parse le header
                m = _RE_HEADER.search(line)
                if m:
                    total = int(m.group(1))

                # Parse [done/total]
                pm = parse_progress_line(line, total)
                if pm and "done" in pm:
                    done = pm["done"]
                    total = pm.get("total", total)

                # Parse la ligne de résumé
                sm = _RE_SUMMARY.search(line)
                if sm:
                    stats = {
                        "ok": int(sm.group(1)),
                        "skip": int(sm.group(2)),
                        "fail": int(sm.group(3)),
                    }

                if progress_callback:
                    frac = done / total if total else 0.0
                    progress_callback(done, total, frac, line, stats)

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                procs.kill_group(proc)
        except Exception as e:  # noqa: BLE001
            logs.append(f"[ERREUR] Exception pendant le subprocess : {e}")
            procs.kill_group(proc)
        finally:
            with self._lock:
                self._proc = None
            procs.kill_group(proc)

        if cancelled or (cancel_event is not None and cancel_event.is_set()):
            return {
                "status": "cancelled",
                "stats": stats,
                "logs": logs,
                "done": done,
                "total": total,
                "error": None,
            }

        # Vérifie si le modèle est trop gros pour la VRAM (8 Go)
        if proc.returncode != 0 and model == "large-v3":
            last_lines = "\n".join(logs[-5:])
            if "out of memory" in last_lines.lower() or "VRAM" in last_lines:
                logs.append(
                    "[CONSEIL] VRAM insuffisante pour large-v3 (8 Go) → "
                    "utilisez le modèle 'turbo' ou 'small'."
                )

        success = proc.returncode == 0 and stats.get("fail", 0) == 0
        error = None
        if not success:
            # La cause réelle est dans le journal : on la remonte dans `error`
            # pour que l'UI ne dise plus « Erreur inconnue ».
            fails = [l for l in logs if re.search(r"\]\s*FAIL\b", l)]
            meaningful = [l for l in logs if l.strip()
                          and not l.startswith("    ")][-6:]
            detail = "\n".join(fails[-3:] or meaningful)
            if proc.returncode != 0:
                error = (f"Transcription échouée (code {proc.returncode}) :\n"
                         f"{detail or 'aucun détail — voir le journal.'}")
            else:
                error = ("Transcription terminée avec des échecs "
                         f"({stats.get('fail', 0)} fichier(s)) :\n"
                         f"{detail or 'voir le journal.'}")
        return {
            "status": "done" if success else "error",
            "stats": stats,
            "logs": logs,
            "done": done,
            "total": total,
            "returncode": proc.returncode,
            "device": device,
            "error": error,
        }
