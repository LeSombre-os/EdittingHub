"""Hub « Coupe des blancs et hésitations » (inspiré de autotrim.app).

Détecte les silences longs et les hésitations (« euh… », « hmm… ») dans un
rush audio/vidéo, puis produit un nouveau fichier où ces segments sont supprimés,
en gardant des respirations naturelles (quelques ms de silence en bord de coupe).

100 % local : ffmpeg + librosa + faster-whisper (venv resolvesubs) pour les
fillers. Le produit est un fichier média réel (pas seulement une timeline),
placé à côté du fichier source avec le suffixe `_trimmed`.

Matériel : détection silences → CPU (librosa), transcription fillers → GPU si
le venv ROCm le permet (sinon repli CPU).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

# Liste des hésitations courantes (FR + EN). Comparaison sur les mots transcrits
# normalisés (minuscules, sans ponctuation).
FILLER_WORDS = {
    # Français
    "euh", "eu", "euhh", "hum", "hmm", "hem", "hmmmm", "pfff", "bah",
    "ben", "donc", "alors", "tu", "voilà",
    # Anglais
    "um", "uh", "umm", "uhh", "like", "so", "well", "you know",
    "sort of", "kind of",
}

# Seuil en secondes : ne couper ces mots que s'ils sont courts et isolés
FILLER_MAX_DURATION = 0.75
FILLER_MIN_DURATION = 0.08


def _probe_duration(path) -> float:
    try:
        from app.media import probe

        d = probe(path).get("duration")
        return float(d) if d else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _ffmpeg(path, *args, **kw):
    return subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, *args],
        capture_output=True, **kw,
    )


def _extract_audio(src, dst, sr=16000):
    """Extrait la piste audio en WAV PCM mono."""
    r = _ffmpeg(src, "-map", "0:a:0", "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", dst)
    if r.returncode != 0:
        raise RuntimeError(f"Extraction audio impossible : {(r.stderr or b'').decode(errors='replace')[:300]}")


def _detect_silence_gaps(y, sr, *, top_db=35.0, min_silence=0.30, keep_silence=0.10):
    """Retourne les intervalles de silence à supprimer.

    Utilise `librosa.effects.split` (segments *non*-silencieux) puis complète les
    trous. `keep_silence` : on conserve un peu de silence aux bords pour que
    la coupe paraisse naturelle (autotrim garde ~80 ms).
    """
    import librosa
    import numpy as np

    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    duration = len(y) / sr
    if intervals is None or len(intervals) == 0:
        return [], [(0.0, duration)]

    # Construire la liste des gaps entre intervalles
    gaps = []
    if intervals[0][0] > 0:
        gaps.append((0.0, intervals[0][0] / sr))
    for i in range(len(intervals) - 1):
        gaps.append((intervals[i][1] / sr, intervals[i + 1][0] / sr))
    if intervals[-1][1] < len(y):
        gaps.append((intervals[-1][1] / sr, duration))

    removes = []
    for s, e in gaps:
        if (e - s) >= min_silence:
            # On garde `keep_silence` aux bords à l'intérieur du gap.
            cut_s = s + keep_silence
            cut_e = e - keep_silence
            if cut_e - cut_s > 0.05:
                removes.append((cut_s, cut_e))
    return removes, intervals


def _transcribe_words(audio_wav_path, python, cancel_event=None):
    """Transcrit avec word-timestamps via le venv ROCm/CPU. Retourne [] si
    impossible (on se contente alors des silences)."""
    if not python:
        return []
    # transcrire.py n'émet pas les mots — on utilise faster-whisper s'il est
    # dans le venv, ou openai-whisper (venv ROCm) en direct.
    # Pragmatique : on tente `faster_whisper` (cpu) en premier.
    import subprocess
    try:
        probe = subprocess.run(
            [python, "-c", "import faster_whisper; print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        probe = None
    if not probe or probe.returncode != 0:
        # Pas de faster-whisper → on tente openai-whisper (ROCm) en direct
        # (plus lent mais fonctionne). On renvoie [] si les deux manquent.
        return []
    worker = os.path.join(tempfile.gettempdir(), f"at_words_{os.getpid()}.py")
    with open(worker, "w", encoding="utf-8") as f:
        f.write(
            "import json,sys\n"
            "from faster_whisper import WhisperModel\n"
            "m = WhisperModel(sys.argv[2], device='cpu', compute_type='int8')\n"
            "segs,_ = m.transcribe(sys.argv[1], word_timestamps=True)\n"
            "out=[]\n"
            "for s in segs:\n"
            "  for w in (s.words or []):\n"
            "    out.append({'start':float(w.start),'end':float(w.end),'word':w.word})\n"
            "print(json.dumps(out))\n"
        )
    try:
        r = subprocess.run([python, worker, audio_wav_path, "small"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []
    finally:
        try:
            os.unlink(worker)
        except OSError:
            pass


def _detect_filler_cuts(words, *, pad=0.06):
    """Détecte les plages à supprimer autour des mots filler."""
    cuts = []
    for w in words:
        word = re.sub(r"[^\w]", "", (w.get("word") or "").strip().lower())
        if word not in FILLER_WORDS:
            continue
        dur = w["end"] - w["start"]
        if dur < FILLER_MIN_DURATION or dur > FILLER_MAX_DURATION:
            continue
        cuts.append((w["start"] - pad, w["end"] + pad))
    # Fusion des cuts proches
    merged = []
    for s, e in sorted(cuts):
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _merge_remove_ranges(ranges):
    """Fusionne et tri des plages à supprimer (bornes inclusives)."""
    ranges = sorted((max(0, s), e) for s, e in ranges)
    out = []
    for s, e in ranges:
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _compute_keep_ranges(duration, remove_ranges, *, min_keep=0.15):
    """Renvoie les intervalles qu'on garde (complémentaires)."""
    removes = _merge_remove_ranges(remove_ranges)
    keeps = []
    cursor = 0.0
    for s, e in removes:
        if s - cursor >= min_keep:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if duration - cursor >= min_keep:
        keeps.append((cursor, duration))
    return keeps, removes


def _render_audio(keep_ranges, src_wav, dst):
    """Produit un nouveau wav avec seulement les segments gardés."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(src_wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    chunks = []
    for s, e in keep_ranges:
        a = int(round(s * sr))
        b = int(round(e * sr))
        a = max(0, min(len(y), a))
        b = max(a, min(len(y), b))
        if b - a > 0:
            chunks.append(y[a:b])
    if not chunks:
        raise RuntimeError("Rien à garder après découpe.")
    out = np.concatenate(chunks)
    sf.write(dst, out, sr)


def _render_video(keep_ranges, src_video, dst, job_id=None, cancel_event=None):
    """Concatène les segments conservés via ffmpeg (ré-encodage H.264/AAC).

    Approche robuste : extraire chaque segment en fichier MPEG-TS indépendant
    puis concaténer via le demuxer concat (évite les problèmes de keyframes).
    """
    workdir = tempfile.mkdtemp(prefix="autotrim_")
    try:
        seg_files = []
        for i, (s, e) in enumerate(keep_ranges):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Annulé.")
            seg_path = os.path.join(workdir, f"seg_{i:04d}.ts")
            a = i
            r = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                 "-i", src_video,
                 "-map", "0:v:0?", "-map", "0:a:0?",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                 "-c:a", "aac", "-b:a", "160k",
                 "-movflags", "+faststart", seg_path],
                capture_output=True,
            )
            if r.returncode != 0 or not os.path.isfile(seg_path):
                # Réessayer sans audio
                r = subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                     "-i", src_video,
                     "-map", "0:v:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                     seg_path],
                    capture_output=True,
                )
                if r.returncode != 0 or not os.path.isfile(seg_path):
                    raise RuntimeError(
                        f"FFmpeg n'a pas pu extraire le segment {i} "
                        f"({s:.2f}s–{e:.2f}s)."
                    )
            seg_files.append(seg_path)

        if not seg_files:
            raise RuntimeError("Aucun segment à produire.")

        concat_list = os.path.join(workdir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for sp in seg_files:
                f.write(f"file '{sp}'\n")

        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c", "copy", "-shortest", dst],
            capture_output=True,
        )
        if r.returncode != 0:
            # Re-encode complet en repli (meilleure compatibilité)
            r = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                 "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", dst],
                capture_output=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"Assemblage FFmpeg échoué : {(r.stderr or b'').decode(errors='replace')[:400]}"
                )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _sub_progress_progress(label, frac, cb, jid):
    if cb:
        try:
            cb(frac, label)
        except Exception:  # noqa: BLE001
            pass



def _normalize_word(w):
    import re
    return re.sub(r"[^\w]", "", (w or "").strip().lower())

def detect_repeats(words, min_overlap_words=2, max_gap_words=3):
    """Détecte les reprises de phrase : suite de mots identiques répétée.

    Retourne une liste de dict {start, end, text, count} décrivant chaque
    répétition détectée (creux de durée pour la coupe).
    """
    n = len(words)
    if n < 4:
        return []
    norm = [_normalize_word(w.get("word", "")) for w in words]
    out = []
    i = 0
    # brute-force sur les longueurs de 2-8 mots
    max_len = min(8, n // 2)
    found_spans = []
    for L in range(max_len, min_overlap_words - 1, -1):
        i = 0
        while i + 2 * L <= n:
            a = norm[i:i + L]
            b = norm[i + L:i + 2 * L]
            if a == b and all(t for t in a):
                # vérifie qu'on n'est pas en overlap avec un span déjà trouvé
                span = (i, i + L)  # la répétition tient au 1er passage
                if not any(s[0] <= i < s[1] for s in found_spans):
                    found_spans.append(span)
                    out.append({
                        "start": words[i]["start"],
                        "end": words[i]["end"],
                        "text": " ".join(w.get("word", "") for w in words[i:i + L]),
                        "count": 2,
                    })
                i += L
            else:
                i += 1
    return out


def analyze_and_trim(src_path, *, min_silence=0.35, keep_silence=0.10, top_db=35.0,
                     enable_fillers=True, progress_cb=None, cancel_event=None,
                     device="auto", jobs=1, cut_repeats=False):
    """Analyse + découpe. Renvoie un dict résultat.

    Résultat :
      {
        'output': chemin du fichier produit,
        'input_duration': float,
        'output_duration': float,
        'removed': float, 'removed_pct': float,
        'gaps': [...], 'fillers': [...],
        'removed_ranges': [...], 'keep_ranges': [...],
      }
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Fichier introuvable : {src_path}")

    def step(frac, label):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Annulé.")
        if progress_cb:
            try:
                progress_cb(frac, label)
            except Exception:  # noqa: BLE001
                pass

    step(0.0, "Analyse…")
    # Une étape bidon pour montrer l'activité immédiatement ; remplacée.
    duration = _probe_duration(src_path)
    if duration <= 0:
        raise RuntimeError("Impossible de lire le fichier (durée nulle).")

    # 1. Extraction audio
    step(0.05, "Extraction de l'audio")
    workdir = tempfile.mkdtemp(prefix="autotrim_")
    try:
        wav = os.path.join(workdir, "audio.wav")
        _extract_audio(src_path, wav)
        import soundfile as sf

        y, sr = sf.read(wav)
        if y.ndim > 1:
            y = y.mean(axis=1)

        is_video = any(src_path.lower().endswith(e) for e in (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"))

        # 2. Détection silences
        step(0.15, "Détection des silences")
        silence_ranges, _ = _detect_silence_gaps(
            y, sr, top_db=top_db, min_silence=min_silence, keep_silence=keep_silence,
        )

        # 3. Détection fillers
        fillers = []
        filler_ranges = []
        if enable_fillers:
            step(0.25, "Transcription (détection des hésitations)")
            python = None
            from app.config import SUBTITLES_PYTHON, SUBTITLES_PYTHON_GPU
            if os.path.isfile(str(SUBTITLES_PYTHON_GPU)) if SUBTITLES_PYTHON_GPU else False:
                python = str(SUBTITLES_PYTHON_GPU)
            elif os.path.isfile(str(SUBTITLES_PYTHON)):
                python = str(SUBTITLES_PYTHON)
            words = _transcribe_words(wav, python, cancel_event)
            filler_ranges = _detect_filler_cuts(words)
            fillers = words

        # 2bis. Repétitions de mots/phrases (par ASR)
        repeats = []
        repeat_ranges = []
        if enable_fillers:
            repeats = detect_repeats(words if 'words' in locals() else [], min_overlap_words=2)
            repeat_ranges = [(r['start'], r['end']) for r in repeats]

        # 4. Fusion + calcul des plages à garder
        step(0.45, "Calcul des plages à conserver")
        _ranges_to_merge = list(silence_ranges) + list(filler_ranges)
        if cut_repeats:
            _ranges_to_merge += list(repeat_ranges)
        removed_ranges = _merge_remove_ranges(_ranges_to_merge)
        keep_ranges, removed_ranges = _compute_keep_ranges(duration, removed_ranges)

        if not keep_ranges:
            raise RuntimeError("Tout serait coupé — annulation par sécurité.")

        removed_time = sum(e - s for s, e in removed_ranges)
        removed_pct = round(removed_time / duration * 100, 1) if duration else 0.0

        base, ext = os.path.splitext(src_path)
        is_video = any(src_path.lower().endswith(e) for e in (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"))
        if is_video:
            out_ext = ".mp4"
        else:
            out_ext = ".wav"
        out_path = base + "_trimmed" + out_ext
        # Évite d'écraser un fichier existant sans force
        i = 2
        while os.path.exists(out_path):
            out_path = f"{base}_trimmed_{i}{out_ext}"
            i += 1

        # Si rien n'a été coupé, pas de re-encodage : copie brute (durée et
        # qualité identiques, et pas d'expansion de 2 frames au mux).
        if removed_time < 0.05:
            shutil.copyfile(src_path, out_path)
            step(1.0, "Terminé (aucune modification détectée)")
            out_dur = _probe_duration(out_path)
            return {
                "status": "done", "output": out_path,
                "input_duration": round(duration, 3), "output_duration": round(out_dur, 3),
                "removed_duration": 0.0, "removed_pct": 0.0,
                "removed_ranges": [], "keep_ranges": keep_ranges,
                "silence_cuts": 0, "filler_words": 0, "seconds_saved": 0.0,
            }

        # 5. Rendu
        step(0.55, "Rendu du média raccourci")
        if is_video:
            _render_video(keep_ranges, src_path, out_path, cancel_event=cancel_event)
        else:
            _render_audio(keep_ranges, wav, out_path)

        step(1.0, "Terminé")
        out_dur = _probe_duration(out_path)
        return {
            "status": "done",
            "output": out_path,
            "input_duration": round(duration, 3),
            "output_duration": round(out_dur, 3),
            "removed_duration": round(removed_time, 3),
            "removed_pct": removed_pct,
            "removed_ranges": [[round(s, 3), round(e, 3)] for s, e in removed_ranges],
            "keep_ranges": [[round(s, 3), round(e, 3)] for s, e in keep_ranges],
            "silence_cuts": len(silence_ranges),
            "filler_words": len(filler_ranges),
            "seconds_saved": round(removed_time, 2),
            "repeats": repeats,
            "repeat_cuts": len(repeat_ranges) if cut_repeats else 0,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    # end of analyze_and_trim (any unhandled exception propagates)
