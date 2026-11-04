"""Hub « Coupe des blancs et hésitations » (inspiré de autotrim.app).

Détecte les silences longs et les hésitations (« euh… », « hmm… ») dans un
rush audio/vidéo, puis produit un nouveau fichier où ces segments sont supprimés,
en gardant des respirations naturelles (quelques ms de silence en bord de coupe).

100 % local : ffmpeg + librosa + faster-whisper/openai-whisper (voir
`app.services.asr`) pour les fillers. Le produit est un fichier média réel (pas
seulement une timeline), placé à côté du fichier source avec le suffixe
`_trimmed`.

Rendu : **une seule passe ffmpeg** (`select`/`aselect` + `setpts`), au lieu
d'un encodage par segment suivi d'une concaténation. Sur un rush de 10 min
coupé en 60 morceaux, c'est ~60× moins de travail d'encodage.

Matériel : détection silences → CPU (librosa), transcription fillers → GPU si
le venv ROCm le permet (sinon repli CPU).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from app import procs
from app.errors import JobCancelled

# Hésitations courantes (FR + EN). Comparaison sur les mots transcrits
# normalisés (minuscules, sans ponctuation). Volontairement restrictif : les
# mots-outils (« donc », « alors », « like », « so ») sont du discours normal,
# les couper produisait des phrases hachées.
FILLER_WORDS = {
    # Français
    "euh", "eu", "euhh", "heu", "heuu", "hum", "hmm", "hem", "hmmmm",
    "pfff", "bof", "ben", "bah", "hein",
    # Anglais
    "um", "uh", "umm", "uhh", "erm", "er", "mm", "mmm",
}

# Seuil en secondes : ne couper ces mots que s'ils sont courts et isolés
FILLER_MAX_DURATION = 0.75
FILLER_MIN_DURATION = 0.08

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts")


def _is_video(path: str) -> bool:
    return str(path).lower().endswith(VIDEO_EXTS)


def _probe_duration(path) -> float:
    try:
        from app.media import probe

        d = probe(path).get("duration")
        return float(d) if d else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _extract_audio(src, dst, sr=16000):
    """Extrait la piste audio en WAV PCM mono (analyse + ASR)."""
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


def _detect_silence_gaps(y, sr, *, top_db=35.0, min_silence=0.30, keep_silence=0.10):
    """Retourne les intervalles de silence à supprimer.

    Utilise `librosa.effects.split` (segments *non*-silencieux) puis complète les
    trous. `keep_silence` : on conserve un peu de silence aux bords pour que
    la coupe paraisse naturelle (autotrim garde ~80 ms).
    """
    import librosa

    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048,
                                      hop_length=512)
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


def _run_ffmpeg(cmd, *, cancel_event=None, total=0.0, on_progress=None):
    """Lance ffmpeg, suit `-progress` et respecte l'annulation."""
    proc = subprocess.Popen(
        procs.nice_cmd(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    tail: list[str] = []
    try:
        for line in procs.iter_lines(proc, cancel_event):
            line = line.strip()
            if not line:
                continue
            if line.startswith(("out_time_us=", "out_time_ms=")):
                if on_progress and total > 0:
                    try:
                        done = int(line.split("=", 1)[1]) / 1_000_000.0
                    except ValueError:
                        continue
                    on_progress(max(0.0, min(1.0, done / total)))
                continue
            tail.append(line)
            del tail[:-15]
        proc.wait(timeout=10)
    finally:
        procs.kill_group(proc)

    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Rendu annulé.")
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg a échoué : " + " | ".join(tail[-4:]))


def _select_expr(ranges) -> str:
    """Expression `between(t,…)` acceptée par les filtres select/aselect."""
    return "+".join(f"between(t,{s:.4f},{e:.4f})" for s, e in ranges)


def _has_audio(path) -> bool:
    from app.media import probe

    try:
        return bool(probe(path).get("has_audio"))
    except Exception:  # noqa: BLE001
        return False


def _render_video(keep_ranges, src_video, dst, *, cancel_event=None, on_progress=None):
    """Rend la vidéo découpée en **une seule passe** ffmpeg.

    `select` filtre les images conservées, `setpts=N/FRAME_RATE/TB` recolle les
    timestamps, `aselect`/`asetpts` font de même pour l'audio. Un seul encodage
    au lieu d'un par segment + concaténation : c'est le gros gain de vitesse.
    """
    expr = _select_expr(keep_ranges)
    total = sum(e - s for s, e in keep_ranges)
    vf = f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[v]"
    maps = ["-map", "[v]"]
    if _has_audio(src_video):
        vf += f";[0:a]aselect='{expr}',asetpts=N/SR/TB[a]"
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-progress", "pipe:1", "-i", src_video,
           "-filter_complex", vf, *maps,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", dst]
    _run_ffmpeg(cmd, cancel_event=cancel_event, total=total,
                on_progress=on_progress)


def _render_audio(keep_ranges, src_audio, dst, *, cancel_event=None, on_progress=None):
    """Rend l'audio découpé depuis la **source** (pas la copie 16 kHz mono).

    Conserve le taux d'échantillonnage et le nombre de canaux d'origine :
    l'ancienne version ré-encodait le WAV d'analyse (16 kHz mono), ce qui
    divisait la qualité par deux.
    """
    expr = _select_expr(keep_ranges)
    total = sum(e - s for s, e in keep_ranges)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-progress", "pipe:1", "-i", src_audio,
           "-filter_complex", f"[0:a]aselect='{expr}',asetpts=N/SR/TB[a]",
           "-map", "[a]", "-c:a", "pcm_s24le", dst]
    _run_ffmpeg(cmd, cancel_event=cancel_event, total=total,
                on_progress=on_progress)


def _normalize_word(w):
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
    # brute-force sur les longueurs de 2-8 mots
    max_len = min(8, n // 2)
    found_spans = []
    for length in range(max_len, min_overlap_words - 1, -1):
        i = 0
        while i + 2 * length <= n:
            a = norm[i:i + length]
            b = norm[i + length:i + 2 * length]
            if a == b and all(t for t in a):
                # vérifie qu'on n'est pas en overlap avec un span déjà trouvé
                span = (i, i + length)  # la répétition tient au 1er passage
                if not any(s[0] <= i < s[1] for s in found_spans):
                    found_spans.append(span)
                    out.append({
                        "start": words[i]["start"],
                        "end": words[i]["end"],
                        "text": " ".join(w.get("word", "") for w in words[i:i + length]),
                        "count": 2,
                    })
                i += length
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
            raise JobCancelled("Annulé.")
        if progress_cb:
            progress_cb(frac, label)

    step(0.0, "Analyse…")
    duration = _probe_duration(src_path)
    if duration <= 0:
        raise RuntimeError("Impossible de lire le fichier (durée nulle).")

    # 1. Extraction audio (16 kHz mono : analyse + ASR)
    step(0.05, "Extraction de l'audio")
    workdir = tempfile.mkdtemp(prefix="autotrim_")
    try:
        wav = os.path.join(workdir, "audio.wav")
        _extract_audio(src_path, wav)
        import soundfile as sf

        y, sr = sf.read(wav)
        if y.ndim > 1:
            y = y.mean(axis=1)

        is_video = _is_video(src_path)

        # 2. Détection silences
        step(0.15, "Détection des silences")
        silence_ranges, _ = _detect_silence_gaps(
            y, sr, top_db=top_db, min_silence=min_silence,
            keep_silence=keep_silence,
        )

        # 3. Détection fillers (ASR, GPU si dispo)
        words: list[dict] = []
        fillers = []
        filler_ranges = []
        if enable_fillers:
            step(0.25, "Transcription (détection des hésitations)")
            from app.services.asr import available, transcribe_words

            if available(device):
                words = transcribe_words(wav, device=device,
                                         cancel_event=cancel_event)
                filler_ranges = _detect_filler_cuts(words)
                fillers = words
            else:
                step(0.40, "ASR indisponible — silences seuls")

        # 3bis. Répétitions de mots/phrases (par ASR)
        repeats = []
        repeat_ranges = []
        if enable_fillers and words:
            repeats = detect_repeats(words, min_overlap_words=2)
            repeat_ranges = [(r["start"], r["end"]) for r in repeats]

        # 4. Fusion + calcul des plages à garder
        step(0.50, "Calcul des plages à conserver")
        to_merge = list(silence_ranges) + list(filler_ranges)
        if cut_repeats:
            to_merge += list(repeat_ranges)
        removed_ranges = _merge_remove_ranges(to_merge)
        keep_ranges, removed_ranges = _compute_keep_ranges(duration,
                                                           removed_ranges)

        if not keep_ranges:
            raise RuntimeError("Tout serait coupé — annulation par sécurité.")

        removed_time = sum(e - s for s, e in removed_ranges)
        removed_pct = round(removed_time / duration * 100, 1) if duration else 0.0

        base = os.path.splitext(src_path)[0]
        out_ext = ".mp4" if is_video else ".wav"
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
                "input_duration": round(duration, 3),
                "output_duration": round(out_dur, 3),
                "removed_duration": 0.0, "removed_pct": 0.0,
                "removed_ranges": [], "keep_ranges": keep_ranges,
                "silence_cuts": 0, "filler_words": 0, "seconds_saved": 0.0,
            }

        # 5. Rendu (une passe ffmpeg)
        step(0.55, "Rendu du média raccourci")

        def _render_progress(frac):
            step(0.55 + 0.40 * float(frac), "Rendu du média raccourci")

        if is_video:
            _render_video(keep_ranges, src_path, out_path,
                          cancel_event=cancel_event,
                          on_progress=_render_progress)
        else:
            _render_audio(keep_ranges, src_path, out_path,
                          cancel_event=cancel_event,
                          on_progress=_render_progress)

        step(1.0, "Terminé")
        out_dur = _probe_duration(out_path)
        return {
            "status": "done",
            "output": out_path,
            "input_duration": round(duration, 3),
            "output_duration": round(out_dur, 3),
            "removed_duration": round(removed_time, 3),
            "removed_pct": removed_pct,
            "removed_ranges": [[round(s, 3), round(e, 3)]
                               for s, e in removed_ranges],
            "keep_ranges": [[round(s, 3), round(e, 3)] for s, e in keep_ranges],
            "silence_cuts": len(silence_ranges),
            "filler_words": len(filler_ranges),
            "seconds_saved": round(removed_time, 2),
            "repeats": repeats,
            "repeat_cuts": len(repeat_ranges) if cut_repeats else 0,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
