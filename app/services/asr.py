"""Transcription ASR locale partagée (autotrim + ScriptCheck).

Un seul endroit pour lancer Whisper, parce que les deux modules avaient deux
copies divergentes du même worker (l'une ignorait `device`, l'autre écrivait
son script temporaire sous un nom fixe : collision entre jobs simultanés).

Deux backends possibles, choisis à l'exécution :
  - GPU  : openai-whisper dans le venv ROCm (`tts-rocm`), `device="cuda"` ;
  - CPU  : faster-whisper dans le venv `resolvesubs` (int8).

Le worker tourne toujours en sous-processus : torch ROCm et faster-whisper
chargés dans le même processus font crasher l'interpréteur. L'annulation tue
le groupe de processus (donc le modèle en cours de chargement inclus).

Note CPU : on lit le WAV en numpy avant de le passer à faster-whisper, car la
version de PyAV du venv `resolvesubs` (19.x) n'est pas compatible avec la
signature `decode_audio(..., metadata_errors=...)` attendue par faster-whisper
1.2.1. Passer un tableau évite PyAV entièrement.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time

from app import procs, runtime
from app.config import SUBTITLES_PYTHON, SUBTITLES_PYTHON_GPU
from app.errors import JobCancelled

DEFAULT_MODEL = "small"

# Worker faster-whisper (CPU). Reçoit wav + modèle en argv.
_CPU_WORKER = r'''
import json, sys, wave
import numpy as np
from faster_whisper import WhisperModel
wav_path = sys.argv[1]
with wave.open(wav_path, "rb") as wf:
    n = wf.getnframes()
    ch = wf.getnchannels()
    data = wf.readframes(n)
    arr = np.frombuffer(data, dtype=np.int16).astype("float32") / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
m = WhisperModel(sys.argv[2], device="cpu", compute_type="int8")
segments, _info = m.transcribe(arr, word_timestamps=True)
out = []
for seg in segments:
    for w in (seg.words or []):
        out.append({"start": float(w.start), "end": float(w.end),
                    "word": w.word})
print(json.dumps(out))
'''

# Worker openai-whisper (GPU ROCm).
_GPU_WORKER = r'''
import contextlib
import io
import json
import sys
import whisper
model = whisper.load_model(sys.argv[2], device="cuda")
with contextlib.redirect_stdout(io.StringIO()):
    result = model.transcribe(sys.argv[1], word_timestamps=True, fp16=True,
                              verbose=False)
out = []
for seg in result.get("segments", []):
    for w in (seg.get("words") or []):
        out.append({"start": float(w["start"]), "end": float(w["end"]),
                    "word": w["word"]})
print(json.dumps(out))
'''

# (python, device_effectif, code_worker), par famille de demande
_BACKENDS: dict[str, tuple[str, str, str]] = {}
_BACKEND_LOCK = threading.Lock()


def _has_module(python: str, module: str) -> bool:
    try:
        r = subprocess.run(
            [python, "-c", f"import {module}"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def backend(device: str = "auto") -> tuple[str, str, str]:
    """Retourne (python, device_effectif, code_worker). Mémoïsé.

    `device` : "auto" | "gpu" | "cpu". En "auto", le GPU n'est utilisé que si
    le venv ROCm sait vraiment transcrire, sinon repli CPU faster-whisper.
    """
    key = "cpu" if device == "cpu" else "auto"
    with _BACKEND_LOCK:
        if key in _BACKENDS:
            return _BACKENDS[key]
        gpu_py = None
        if key == "auto" and SUBTITLES_PYTHON_GPU:
            cand = str(SUBTITLES_PYTHON_GPU)
            if os.path.isfile(cand) and runtime.whisper_gpu_available():
                gpu_py = cand
        if gpu_py:
            _BACKENDS[key] = (gpu_py, "gpu", _GPU_WORKER)
        else:
            _BACKENDS[key] = (str(SUBTITLES_PYTHON), "cpu", _CPU_WORKER)
        return _BACKENDS[key]


def available(device: str = "auto") -> bool:
    """Vrai si un interpréteur de transcription utilisable existe."""
    python, dev, _ = backend(device)
    if not os.path.isfile(python):
        return False
    return _has_module(python, "whisper" if dev == "gpu" else "faster_whisper")


def transcribe_words(wav_path: str, *, device: str = "auto",
                     model: str = DEFAULT_MODEL, cancel_event=None,
                     timeout: float = 1800.0) -> list[dict]:
    """Transcrit un WAV en mots horodatés ``[{start, end, word}]``.

    Lève JobCancelled si `cancel_event` est levé, RuntimeError sinon.
    """
    python, dev, worker_src = backend(device)
    if not os.path.isfile(python):
        raise RuntimeError(f"Interpréteur de transcription introuvable : {python}")

    fd, worker = tempfile.mkstemp(prefix="asr_", suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(worker_src)

        try:
            proc = subprocess.Popen(
                procs.nice_cmd([python, worker, wav_path, model]),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Python introuvable : {python}") from e

        lines: list[str] = []
        err_tail: list[str] = []
        started = time.time()
        killed = threading.Event()

        def _on_cancel():
            killed.set()

        def _drain_stderr():
            try:
                err_tail.extend((proc.stderr.read() or "").splitlines()[-8:])
            except Exception:  # noqa: BLE001
                pass

        reader = threading.Thread(target=_drain_stderr, daemon=True)
        reader.start()

        try:
            for line in procs.iter_lines(proc, cancel_event, on_cancel=_on_cancel):
                lines.append(line)
                if time.time() - started > timeout:
                    procs.kill_group(proc)
                    raise RuntimeError(
                        f"Transcription trop longue (> {int(timeout)} s).")
            proc.wait(timeout=10)
        finally:
            procs.kill_group(proc)

        reader.join(timeout=2)

        if killed.is_set() or (cancel_event is not None and cancel_event.is_set()):
            raise JobCancelled("Transcription annulée.")
        if proc.returncode != 0:
            detail = " | ".join(err_tail) or "(pas de sortie d'erreur)"
            raise RuntimeError(f"Échec de la transcription : {detail[:400]}")

        raw = "".join(lines).strip()
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Sortie ASR illisible : {raw[:200]}") from e
        return data if isinstance(data, list) else []
    finally:
        try:
            os.unlink(worker)
        except OSError:
            pass
