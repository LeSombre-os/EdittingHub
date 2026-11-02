"""Hub de Conversion — redirige vers ResolveConvert/convert.py par sous-processus.

Ce service wrappe le script de conversion existant sans le modifier.
Il orchestre :
  1. un *scan* (probe + estimation de taille via --estimate) ;
  2. un *run* (conversion réelle) exécuté dans un sous-processus, avec
     suivi de progression et annulation.

Le subprocess est isolé : un crash dans convert.py ne fait pas tomber le hub.
"""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from app import procs
from app.config import CONVERT_SCRIPT, CONVERT_PYTHON
from app.utils import fmt_size, fmt_duration, parse_progress_line


# Regex pour parser les lignes de sortie de convert.py
_RE_SUMMARY = re.compile(
    r"Terminé.*?: (\d+) convertis, (\d+) déjà OK, (\d+) échecs"
)
_RE_HEADER = re.compile(
    r"ResolveConvert\s*—\s*(\d+) fichier\(s\), cible=(\w+), preset (\S+), jobs=(\d+)"
)


class ConvertService:
    """Orchestration des conversions vidéo via ResolveConvert."""

    def __init__(self):
        self.script = str(CONVERT_SCRIPT)
        self.python = CONVERT_PYTHON
        self._proc = None
        self._lock = threading.Lock()

    # ── Annulation externe ─────────────────────────────────────────────────
    def terminate(self):
        """Tue le sous-processus en cours et tous ses enfants (ffmpeg)."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        return procs.kill_group(proc)

    # ── Vérification préalable ─────────────────────────────────────────────
    def available(self):
        """Vérifie que le script de conversion est présent."""
        return os.path.isfile(self.script)

    # ── Scan (estimation sans conversion) ──────────────────────────────────
    def scan(self, paths, target="resolve", preset="sq", keep_4k=False):
        """Lance un scan d'estimation via --estimate.

        Retourne le plan JSON décodé ou lève RuntimeError.
        """
        if not self.available():
            raise RuntimeError(
                f"Script introuvable : {self.script}. "
                "Vérifiez que le projet ResolveConvert est présent."
            )
        cmd = [self.python, self.script, "--estimate",
               "--target", target, "--preset", preset]
        if keep_4k:
            cmd.append("--keep-4k")
        cmd.extend(paths)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=True
            )
            data = json.loads(result.stdout)
            return data
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Erreur de parsing JSON du script de conversion : {e}\n"
                f"Sortie : {result.stdout[-500:]}"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Le script de conversion a échoué (code {e.returncode}) :\n"
                f"{e.stderr or e.stdout}"
            ) from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout du scan (120s) — fichier trop volumineux ?")
        except FileNotFoundError:
            raise RuntimeError(f"Python introuvable : {self.python}")

    # ── Exécution (conversion) ─────────────────────────────────────────────
    def run(self, paths, target="resolve", preset="sq", keep_4k=False,
            force=False, jobs=4, cancel_event=None, progress_callback=None):
        """Exécute la conversion via sous-processus.

        Args:
            paths : liste de chemins à convertir
            target : 'resolve' ou 'tiktok'
            preset : 'lb', 'sq', ou 'hq'
            keep_4k : garder la résolution 4K (cible resolve)
            force : reconvertir même si la sortie existe
            jobs : nombre de conversions parallèles
            cancel_event : threading.Event pour annuler
            progress_callback : fonction(done, total, frac, line, stats)

        Returns:
            dict avec status, stats, logs
        """
        if not self.available():
            raise RuntimeError(
                f"Script introuvable : {self.script}"
            )

        cmd = [self.python, self.script]
        cmd.extend(paths)
        cmd.extend(["--target", target, "--preset", preset])
        if keep_4k:
            cmd.append("--keep-4k")
        if force:
            cmd.append("--force")
        if jobs and jobs > 1:
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
                # Session dédiée : permet de tuer tout le groupe (ffmpeg inclus)
                start_new_session=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Python introuvable : {self.python}. "
                "Vérifiez l'installation."
            )

        with self._lock:
            self._proc = proc

        cancelled = False

        def _on_cancel():
            nonlocal cancelled
            cancelled = True
            logs.append("[ANNULÉ] Arrêt demandé — processus et enfants tués.")

        # Lecture ligne par ligne, sans blocage (annulation réactive)
        try:
            for line in procs.iter_lines(proc, cancel_event, on_cancel=_on_cancel):
                line = line.rstrip("\n")
                logs.append(line)

                # Parse le header pour détecter le nombre total de fichiers
                m = _RE_HEADER.search(line)
                if m:
                    total = int(m.group(1))

                # Parse les lignes de progression [done/total]
                pm = parse_progress_line(line, total)
                if pm and "done" in pm:
                    done = pm["done"]
                    total = pm.get("total", total)

                # Parse la ligne de résumé finale
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
            # Nettoyage défensif : aucun enfant ne doit survivre au job
            procs.kill_group(proc)

        if cancelled or (cancel_event is not None and cancel_event.is_set()):
            stats["fail"] += (total - done) if total else 1
            return {
                "status": "cancelled",
                "stats": stats,
                "logs": logs,
                "done": done,
                "total": total,
            }

        success = proc.returncode == 0
        return {
            "status": "done" if success else "error",
            "stats": stats,
            "logs": logs,
            "done": done,
            "total": total,
            "returncode": proc.returncode,
        }
