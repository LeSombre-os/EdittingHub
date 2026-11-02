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
from app.config import SUBTITLES_SCRIPT, SUBTITLES_PYTHON
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
        self._proc = None
        self._lock = threading.Lock()

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

    def _check_backend(self, device="auto", model="small"):
        """Vérifie que le backend (faster-whisper) est disponible.

        Retourne (ok: bool, message: str).
        """
        try:
            import importlib
            spec = importlib.util.find_spec("faster_whisper")
            if spec is None:
                # Test dans le venv resolvesubs
                try:
                    r = subprocess.run(
                        [self.python, "-c", "import faster_whisper"],
                        capture_output=True, text=True, timeout=30
                    )
                    if r.returncode != 0:
                        return False, (
                            "faster-whisper introuvable dans le venv resolvesubs. "
                            "Installez-le : ~/.venvs/resolvesubs/bin/pip install faster-whisper"
                        )
                except Exception:
                    return False, "faster-whisper introuvable."
            return True, "OK"
        except Exception as e:
            return False, str(e)

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
            dict avec status, stats, logs
        """
        if not self.available():
            raise RuntimeError(
                f"Script introuvable : {self.script}"
            )

        # Vérification du backend avant de lancer (message clair)
        ok, msg = self._check_backend(device, model)
        if not ok:
            raise RuntimeError(msg)

        cmd = [self.python, self.script]
        cmd.extend(paths)
        cmd.extend(["--language", language, "--model", model])
        cmd.extend(["--device", device])
        cmd.extend(["--max-chars", str(max_chars), "--max-lines", str(max_lines)])
        cmd.extend(["--mode", mode, "--words-per-cue", str(max(1, min(10, words_per_cue)))])
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

        logs = []
        stats = {"ok": 0, "skip": 0, "fail": 0}
        done = 0
        total = 0

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Python introuvable : {self.python}. "
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
        except Exception as e:
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
            }

        # Vérifie si le modèle est trop gros pour la VRAM (8 Go)
        if proc.returncode != 0 and model == "large-v3":
            last_lines = "\n".join(logs[-5:])
            if "out of memory" in last_lines.lower() or "VRAM" in last_lines:
                logs.append(
                    "[CONSEIL] VRAM insuffisante pour large-v3 (8 Go) → "
                    "utilisez le modèle 'turbo' ou 'small'."
                )

        success = proc.returncode == 0
        return {
            "status": "done" if success else "error",
            "stats": stats,
            "logs": logs,
            "done": done,
            "total": total,
            "returncode": proc.returncode,
        }
