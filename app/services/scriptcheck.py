"""Hub « Vérification de script » : compare ce qui est dit (ASR) avec ce qui
est écrit dans un script de travail.

Produit un rapport lisible : couverture (% de mots du script effectivement dits),
mots manquants, mots en plus et déviations dans l'ordre. Tout est local
(faster-whisper CPU ou openai-whisper ROCm selon `device`, voir
`app.services.asr`).
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import tempfile

from app.errors import JobCancelled

# `\w` est Unicode par défaut en Python 3 : les accents sont conservés (sans
# eux, « très » et « tres » ne s'alignaient pas et la couverture chutait).
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalize(text):
    text = (text or "").lower().replace("’", "'").replace("'", " ")
    return [t for t in _WORD_RE.split(text) if t]


def _extract_audio(src, dst, sr=16000):
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-i", src, "-map", "0:a:0", "-ar", str(sr), "-ac", "1",
         "-c:a", "pcm_s16le", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "Extraction audio impossible : "
            f"{(r.stderr or b'').decode(errors='replace')[:300]}")


def _probe_duration(path):
    try:
        from app.media import probe
        d = probe(path).get("duration")
        return float(d) if d else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def check_script(src_path, script_text, progress_cb=None, cancel_event=None,
                 device="auto"):
    if not src_path or not os.path.isfile(src_path):
        raise FileNotFoundError(f"Media introuvable : {src_path}")
    if not script_text or not script_text.strip():
        raise ValueError("Script vide — fournir un texte de référence.")

    def step(frac, label):
        if cancel_event and cancel_event.is_set():
            raise JobCancelled("Annulé.")
        if progress_cb:
            progress_cb(frac, label)

    step(0.05, "Extraction audio")
    workdir = tempfile.mkdtemp(prefix="sc_")
    try:
        wav = os.path.join(workdir, "audio.wav")
        _extract_audio(src_path, wav)
        step(0.25, "Transcription (ASR)")
        from app.services.asr import available, transcribe_words

        if not available(device):
            raise RuntimeError(
                "Aucun moteur de transcription disponible "
                "(faster-whisper ou whisper ROCm).")
        words = transcribe_words(wav, device=device, cancel_event=cancel_event)
        step(0.6, "Analyse et comparaison")
        asr_tokens = _normalize(" ".join(w.get("word", "") for w in words))
        script_tokens = _normalize(script_text)

        # difflib aligne les séquences (adapté à une lecture à voix haute).
        sm = difflib.SequenceMatcher(a=script_tokens, b=asr_tokens)
        blocks = sm.get_matching_blocks()
        matches = sum(b.size for b in blocks if b.size > 0)
        coverage = (matches / len(script_tokens) * 100.0) if script_tokens else 0.0

        # opcodes (replace/delete/insert)
        deviations = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
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
