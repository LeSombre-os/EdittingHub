"""Hub « Vérification de script » : compare ce qui est dit (ASR) avec ce qui
est écrit dans un script de travail.

Produit un rapport lisible : couverture (% de mots du script effectivement dits),
mots manquants, mots en plus et déviations dans l'ordre. Tout est local
(faster-whisper ou Whisper selon venv).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
import threading

import subprocess


def _normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\\s]", " ", text)
    return [t for t in text.split() if t]


def _extract_audio(src, dst, sr=16000):
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", src, "-map", "0:a:0", "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Extraction audio impossible : {(r.stderr or b'').decode(errors='replace')[:300]}")


def _probe_duration(path):
    try:
        from app.media import probe
        d = probe(path).get("duration")
        return float(d) if d else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _transcribe_words(audio_wav, device="auto"):
    from app.config import SUBTITLES_PYTHON, SUBTITLES_PYTHON_GPU

    # choisir l'interpréteur avec faster-whisper (resolvesubs) — stable
    python = None
    if os.path.isfile(str(SUBTITLES_PYTHON)):
        python = str(SUBTITLES_PYTHON)
    elif SUBTITLES_PYTHON_GPU and os.path.isfile(str(SUBTITLES_PYTHON_GPU)):
        python = str(SUBTITLES_PYTHON_GPU)
    if not python:
        raise RuntimeError("Aucun interpréteur de transcription disponible.")

    # faster_whisper d'abord (CPU), repli openai-whisper si dispo
    probe = subprocess.run(
        [python, "-c", "import faster_whisper; print('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    worker = None
    try:
        if probe.returncode == 0:
            worker = os.path.join(tempfile.gettempdir(), f"sc_words_{os.getpid()}.py")
            with open(worker, "w", encoding="utf-8") as f:
                f.write(
                    "import json,sys,wave\n"
                    "import numpy as np\n"
                    "from faster_whisper import WhisperModel\n"
                    "wav_path = sys.argv[1]\n"
                    "with wave.open(wav_path,'rb') as wf:\n"
                    "  n=wf.getnframes(); sr_in=wf.getframerate(); ch=wf.getnchannels()\n"
                    "  data=wf.readframes(n)\n"
                    "  arr=np.frombuffer(data,dtype=np.int16).astype('float32')/(32768.0+1e-9)\n"
                    "  if ch>1: arr=arr.reshape(-1,ch).mean(axis=1)\n"
                    "  if sr_in!=16000:\n"
                    "    pass  # on suppose déjà 16k\n"
                    "m = WhisperModel(sys.argv[2], device='cpu', compute_type='int8')\n"
                    "segs,_ = m.transcribe(arr, word_timestamps=True)\n"
                    "out=[]\n"
                    "for s in segs:\n"
                    "  for w in (s.words or []):\n"
                    "    out.append({'start':float(w.start),'end':float(w.end),'word':w.word})\n"
                    "print(json.dumps(out))\n"
                )
            r = subprocess.run([python, worker, audio_wav, "small"],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout)[:500])
            data = json.loads(r.stdout or "[]")
            return data if isinstance(data, list) else []
        raise RuntimeError("faster-whisper introuvable")
    finally:
        if worker and os.path.exists(worker):
            os.unlink(worker)


def check_script(src_path, script_text, progress_cb=None, cancel_event=None,
                 device="auto"):
    if not src_path or not os.path.isfile(src_path):
        raise FileNotFoundError(f"Media introuvable : {src_path}")
    if not script_text or not script_text.strip():
        raise ValueError("Script vide — fournir un texte de référence.")

    def step(frac, label):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Annulé.")
        if progress_cb:
            progress_cb(frac, label)

    duration = _probe_duration(src_path)
    step(0.05, "Extraction audio")
    workdir = tempfile.mkdtemp(prefix="sc_")
    try:
        wav = os.path.join(workdir, "audio.wav")
        _extract_audio(src_path, wav)
        step(0.25, "Transcription (ASR)")
        words = _transcribe_words(wav, device)
        step(0.6, "Analyse et comparaison")
        asr_tokens = _normalize(" ".join(w.get("word", "") for w in words))
        script_tokens = _normalize(script_text)

        # difflib aligns sequences (good for read-outs)
        sm = difflib.SequenceMatcher(a=script_tokens, b=asr_tokens)
        blocks = sm.get_matching_blocks()
        # coverage: sum of matching tags length vs script length
        matches = sum(b.size for b in blocks if b.size > 0)
        coverage = (matches / len(script_tokens) * 100.0) if script_tokens else 0.0

        # opcodes (replace/delete/insert)
        opcodes = sm.get_opcodes()
        deviations = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            deviations.append({
                "type": tag,  # replace / delete / insert
                "script_words": script_tokens[i1:i2],
                "said_words": asr_tokens[j1:j2],
            })

        step(1.0, "Terminé")
        return {
            "status": "done",
            "coverage_pct": round(coverage, 1),
            "script_word_count": len(script_tokens),
            "said_word_count": len(asr_tokens),
            "matches": matches,
            "deviations": deviations,
            "verdict": ("OK — script lu intégralement" if coverage >= 95
                        else ("Partiel — oublis/déviations détectés"
                              if coverage >= 70 else "Divergence importante")),
            "asr_text": " ".join(w.get("word", "") for w in words),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
