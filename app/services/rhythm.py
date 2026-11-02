"""Montage au rythme — découpe automatique d'une vidéo sur les beats.

Principe
--------
On dispose déjà, via :mod:`app.services.audio_beats`, des beats classés en
trois rangs d'énergie. Ce module transforme cette liste en **plan de montage** :

1. on ne garde que les beats des rangs demandés (rang 1 = impacts forts) ;
2. on impose une durée de plan minimale (deux beats trop proches → un seul
   plan) et maximale (un long silence → plusieurs plans de durée égale) ;
3. on parcourt la vidéo **séquentiellement** : chaque plan prend le morceau
   suivant du rush, dans l'ordre. Si la musique est plus longue que la vidéo,
   on boucle (option) au lieu de s'arrêter.

Le résultat est exportable en FCPXML 1.9 (importable par DaVinci Resolve, avec
la musique sur une piste liée), en EDL CMX 3600 (4 timecodes par événement) ou
en CSV.

100 % local, aucun appel réseau.
"""
from __future__ import annotations

import math
import os

from app.media import probe
from app.utils import seconds_to_edl_timecode


# ── Sélection des points de coupe ─────────────────────────────────────────

def selected_beats(beats, ranks) -> list[float]:
    """Temps (s) des beats appartenant aux rangs demandés, triés et dédoublonnés."""
    allowed = {int(r) for r in (ranks or [])}
    times: list[float] = []
    for beat in beats or []:
        try:
            t = float(beat.get("time"))
        except (TypeError, ValueError):
            continue
        if t <= 0:
            continue
        try:
            rank = int(beat.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        if allowed and rank not in allowed:
            continue
        times.append(round(t, 6))
    return sorted(set(times))


def _append_span(points: list[float], cursor: float, target: float,
                 min_shot: float, max_shot: float) -> None:
    """Ajoute les coupes entre ``cursor`` et ``target``.

    Un écart plus long que ``max_shot`` est divisé en plans de durée égale
    (mais jamais plus courts que ``min_shot``).
    """
    gap = target - cursor
    if gap <= 0:
        return
    n = max(1, math.ceil(gap / max_shot))
    n = min(n, max(1, int(gap // min_shot)))
    step = gap / n
    for k in range(1, n):
        points.append(round(cursor + step * k, 6))
    points.append(round(target, 6))


def cut_points(beats, audio_duration: float, ranks=(1,), min_shot: float = 0.8,
               max_shot: float = 3.0) -> tuple[list[float], int]:
    """Construit les instants de coupe (début 0, fin = durée de la musique).

    Returns:
        (points, nombre_de_beats_retenus)
    """
    min_shot = max(0.05, float(min_shot))
    max_shot = max(min_shot, float(max_shot))
    audio_duration = float(audio_duration or 0.0)
    if audio_duration <= 0:
        raise ValueError("Durée audio inconnue : impossible de calculer le montage.")

    times = selected_beats(beats, ranks)
    points: list[float] = [0.0]
    cursor = 0.0
    used = 0

    for t in times:
        if t >= audio_duration - 1e-6:
            break
        gap = t - cursor
        if gap < min_shot:
            continue
        _append_span(points, cursor, t, min_shot, max_shot)
        cursor = t
        used += 1

    tail = audio_duration - cursor
    if tail >= min_shot:
        _append_span(points, cursor, audio_duration, min_shot, max_shot)
    elif len(points) > 1:
        # Fin trop courte : on rallonge le dernier plan au lieu de créer un
        # flash de quelques centièmes.
        points[-1] = round(audio_duration, 6)
    else:
        points.append(round(audio_duration, 6))

    return points, used


# ── Découpage séquentiel de la vidéo ──────────────────────────────────────

def _video_chunks(cursor: float, duration: float, start: float, end: float,
                  loop: bool) -> list[tuple[float, float]]:
    """Morceaux ``(in, durée)`` du rush nécessaires pour couvrir ``duration``.

    Avec ``loop``, on repart au début de la zone utile quand on atteint la fin.
    Sans ``loop``, on s'arrête à la fin du média (le plan est tronqué).
    """
    span = end - start
    if span <= 0 or duration <= 0:
        return []

    chunks: list[tuple[float, float]] = []
    remaining = duration
    pos = min(max(cursor, start), end)

    while remaining > 1e-9:
        available = end - pos
        if available <= 1e-9:
            if loop:
                pos = start
                continue
            break
        take = min(available, remaining)
        chunks.append((pos, take))
        remaining -= take
        pos = start if loop else end

    return chunks


def _advance(cursor: float, duration: float, start: float, end: float,
             loop: bool) -> float:
    """Position suivante dans le rush après un plan de ``duration`` secondes."""
    span = end - start
    if span <= 0:
        return end
    if loop:
        return start + ((cursor - start + duration) % span)
    return min(end, cursor + duration)


def build_cut_plan(beats, video_duration: float, audio_duration: float,
                   ranks=(1,), min_shot: float = 0.8, max_shot: float = 3.0,
                   loop: bool = True, fps: float = 24.0,
                   video_start: float = 0.0) -> dict:
    """Plan de montage complet (coupes + statistiques).

    Args:
        beats : liste de dicts ``{time, rank, …}``.
        video_duration : durée du rush.
        audio_duration : durée de la musique (fin de la timeline).
        ranks : rangs de beats à utiliser (1 = impacts forts).
        min_shot / max_shot : durée mini / maxi d'un plan (secondes).
        loop : boucler la vidéo si elle est plus courte que la musique.
        fps : fps de la timeline (timecodes EDL).
        video_start : point de départ dans la vidéo (secondes).

    Returns:
        dict avec ``cuts``, ``segments``, ``beats_used``, durées, etc.
    """
    video_duration = float(video_duration or 0.0)
    audio_duration = float(audio_duration or 0.0)
    if video_duration <= 0:
        raise ValueError("Durée vidéo inconnue : impossible de calculer le montage.")
    if audio_duration <= 0:
        raise ValueError("Durée audio inconnue : impossible de calculer le montage.")

    fps = float(fps or 24.0)
    min_shot = max(0.05, float(min_shot))
    max_shot = max(min_shot, float(max_shot))
    start = min(max(0.0, float(video_start or 0.0)), max(0.0, video_duration - 1e-3))
    end = video_duration
    if end - start < 1e-3:
        start = 0.0

    points, beats_used = cut_points(beats, audio_duration, ranks=ranks,
                                    min_shot=min_shot, max_shot=max_shot)

    cuts: list[dict] = []
    cursor = start
    timeline = 0.0
    shot = 0
    truncated = False

    for index in range(len(points) - 1):
        shot_start, shot_end = points[index], points[index + 1]
        duration = shot_end - shot_start
        if duration <= 1e-6:
            continue
        shot += 1

        chunks = _video_chunks(cursor, duration, start, end, loop)
        if not chunks:
            truncated = True
            break

        for src_in, chunk_dur in chunks:
            record_in = timeline
            timeline += chunk_dur
            cuts.append({
                "shot": shot,
                "beat": round(shot_start, 3),
                "start": round(src_in, 6),
                "end": round(src_in + chunk_dur, 6),
                "duration": round(chunk_dur, 6),
                "timeline": round(record_in, 6),
                "start_tc": seconds_to_edl_timecode(src_in, fps),
                "end_tc": seconds_to_edl_timecode(src_in + chunk_dur, fps),
                "record_in_tc": seconds_to_edl_timecode(record_in, fps),
                "record_out_tc": seconds_to_edl_timecode(timeline, fps),
            })

        consumed = sum(chunk[1] for chunk in chunks)
        if consumed + 1e-6 < duration:
            truncated = True

        cursor = _advance(cursor, duration, start, end, loop)
        if not loop and cursor >= end - 1e-6:
            # Dernier plan : il peut se terminer pile à la fin du média.
            cursor = end
            if index < len(points) - 2:
                truncated = True
                break

    return {
        "cuts": cuts,
        "points": [round(p, 3) for p in points],
        "segments": shot,
        "clips": len(cuts),
        "beats_used": beats_used,
        "beats_total": len(beats or []),
        "timeline_duration": round(timeline, 3),
        "audio_duration": round(audio_duration, 3),
        "video_duration": round(video_duration, 3),
        "video_start": round(start, 3),
        "loop": bool(loop),
        "truncated": truncated,
        "min_shot": round(min_shot, 3),
        "max_shot": round(max_shot, 3),
        "fps": round(fps, 3),
    }


# ── Exports ────────────────────────────────────────────────────────────────

def export_rhythm_edl(video_path: str, cuts, fps: float = 24.0,
                      title: str | None = None) -> str:
    """EDL CMX 3600 : un événement par plan, 4 timecodes (source + record)."""
    title = title or os.path.splitext(os.path.basename(video_path))[0]
    video_name = os.path.basename(video_path)

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    if not cuts:
        lines.append("001  BL       V     C        00:00:00:00 00:00:00:01 "
                     "00:00:00:00 00:00:00:01")
        lines.append(f"     * FROM CLIP NAME: {video_name}")
        lines.append("")
        return "\n".join(lines)

    for index, cut in enumerate(cuts, 1):
        lines.append(
            f"{index:03d}  AX       V     C        "
            f"{cut['start_tc']} {cut['end_tc']} "
            f"{cut['record_in_tc']} {cut['record_out_tc']}"
        )
        lines.append(f"     * FROM CLIP NAME: {video_name}")
        lines.append("")

    return "\n".join(lines)


def export_rhythm_csv(cuts) -> str:
    """CSV lisible : plan, beat, source in/out, durée, timeline in/out."""
    lines = ["shot,beat_s,source_in_s,source_out_s,duration_s,"
             "timeline_in_s,timeline_out_s"]
    for cut in cuts:
        lines.append(
            f"{cut['shot']},{cut['beat']},{cut['start']},{cut['end']},"
            f"{cut['duration']},{cut['timeline']},"
            f"{round(cut['timeline'] + cut['duration'], 6)}"
        )
    return "\n".join(lines)


def export_rhythm(video_path: str, cuts, fps: float, fmt: str = "fcpxml",
                  audio_path: str | None = None, title: str | None = None,
                  audio_duration: float | None = None,
                  include_audio: bool = True) -> str:
    """Exporte le montage dans le format demandé.

    fmt : ``fcpxml`` (alias ``xml``), ``edl`` ou ``csv``.
    """
    if fmt in ("fcpxml", "xml"):
        from app.fcpxml import rhythm_document
        return rhythm_document(
            video_path, fps, cuts, title=title, audio_path=audio_path,
            audio_duration=audio_duration, include_audio=include_audio,
        )
    if fmt == "edl":
        return export_rhythm_edl(video_path, cuts, fps, title=title)
    if fmt == "csv":
        return export_rhythm_csv(cuts)
    raise ValueError(f"Format d'export inconnu : {fmt}")


def rhythm_summary(plan: dict) -> dict:
    """Résumé court pour les logs / l'UI."""
    return {
        "segments": plan.get("segments", 0),
        "clips": plan.get("clips", 0),
        "beats_used": plan.get("beats_used", 0),
        "beats_total": plan.get("beats_total", 0),
        "timeline_duration": plan.get("timeline_duration", 0.0),
        "truncated": plan.get("truncated", False),
    }


def probe_sources(video_path: str, audio_path: str) -> tuple[dict, dict]:
    """Sonde la vidéo et la musique (ffprobe)."""
    return probe(video_path), probe(audio_path)
