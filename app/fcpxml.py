"""Construction de timelines FCPXML 1.9 importables par DaVinci Resolve.

Pourquoi ce module : les anciens exports XML plaçaient les `<marker>` comme
enfants directs de `<sequence>` et déclaraient un `<asset>` sans `src`, sans
`duration` ni `format`. Ce n'est pas du FCPXML valide : Resolve ignore alors
tout le contenu (timeline vide à l'import).

Structure correcte (celle produite ici) :

    <fcpxml version="1.9">
      <resources>
        <format id="r1" …/>
        <asset id="r2" … duration="…" hasVideo="…" hasAudio="…" format="r1">
          <media-rep kind="original-media" src="file:///…"/>
        </asset>
      </resources>
      <library><event><project>
        <sequence …><spine>
          <asset-clip ref="r2" offset="…" start="…" duration="…">
            <marker start="…" duration="…" value="…"/>   ← markers sur le CLIP
          </asset-clip>
        </spine></sequence>
      </project></event></library>
    </fcpxml>

Les temps sont des rationnels en secondes (`1001/30000s`), comme dans les
exports de Final Cut Pro.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from urllib.parse import quote

from app.media import fps_exact, probe

LIBRARY_NAME = "EdittingHub"


def _fraction(value: float):
    """Fraction exacte d'un fps (23.976 → 24000/1001)."""
    return fps_exact(value)


def _frames(seconds: float, fps: float) -> int:
    """Convertit des secondes en nombre de frames (arrondi, min 0).

    Arrondi round-half-up pour rester cohérent avec le script DaVinci
    (`int(seconds * fps + 0.5)`) et l'EDL (`seconds_to_frames`) : sans cela,
    le même beat peut tomber sur deux frames voisines selon le format
    d'export.
    """
    from app.utils import seconds_to_frames
    return seconds_to_frames(seconds, fps)


def _time(frames: int, fps: float) -> str:
    """Nombre de frames → valeur de temps FCPXML ('N/fps s')."""
    frac = _fraction(fps)
    # frameDuration = 1/fps = den/num → durée N frames = N*den/num
    return f"{frames * frac.denominator}/{frac.numerator}s"


def _frame_duration(fps: float) -> str:
    frac = _fraction(fps)
    return f"{frac.denominator}/{frac.numerator}s"


def _file_url(path: str) -> str:
    """URL file:// absolue et encodée (espaces, accents, #, ?)."""
    abspath = os.path.abspath(path)
    return "file://" + quote(abspath, safe="/")


def _format_name(width: int, height: int, fps: float) -> str:
    """Nom lisible du format (Resolve se base sur frameDuration/width/height)."""
    fps_txt = f"{fps:.2f}".rstrip("0").rstrip(".").replace(".", "") or "24"
    return f"FFVideoFormat{height or 1080}p{fps_txt}"


def _base_document(title: str, asset_path: str, fps: float, duration: float):
    """Crée l'arbre FCPXML avec resources + library/event/project.

    Retourne (root, project, sequence, asset_id, info_probe).
    """
    info = probe(asset_path)
    asset_duration = float(info.get("duration") or 0.0) or float(duration) or 1.0
    # La timeline ne peut pas dépasser le média
    timeline_duration = min(float(duration) or asset_duration, asset_duration)
    if timeline_duration <= 0:
        timeline_duration = asset_duration

    width = int(info.get("width") or 1920)
    height = int(info.get("height") or 1080)
    has_video = bool(info.get("has_video"))
    has_audio = bool(info.get("has_audio")) or not has_video

    root = ET.Element("fcpxml", {"version": "1.9"})
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", {
        "id": "r1",
        "name": _format_name(width, height, fps),
        "frameDuration": _frame_duration(fps),
        "width": str(width),
        "height": str(height),
    })
    asset_attrs = {
        "id": "r2",
        "name": os.path.basename(asset_path) or "asset",
        "start": "0s",
        "duration": _time(_frames(asset_duration, fps), fps),
        "hasVideo": "1" if has_video else "0",
        "hasAudio": "1" if has_audio else "0",
    }
    # Un asset purement audio ne doit pas référencer un format vidéo :
    # Resolve refuse sinon le média (et n'importe aucun marqueur).
    if has_video:
        asset_attrs["format"] = "r1"
    asset = ET.SubElement(resources, "asset", asset_attrs)
    ET.SubElement(asset, "media-rep", {
        "kind": "original-media",
        "src": _file_url(asset_path),
    })

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", {"name": LIBRARY_NAME})
    project = ET.SubElement(event, "project", {"name": title})
    sequence = ET.SubElement(project, "sequence", {
        "format": "r1",
        "duration": _time(_frames(timeline_duration, fps), fps),
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k",
    })
    spine = ET.SubElement(sequence, "spine")

    return root, spine, info, asset_duration, timeline_duration


def _add_marker(parent, seconds: float, fps: float, value: str, note: str = ""):
    """Ajoute un marker sur un clip (et non sur la séquence)."""
    attrs = {
        "start": _time(_frames(seconds, fps), fps),
        "duration": _frame_duration(fps),
        "value": value,
    }
    if note:
        attrs["note"] = note
    ET.SubElement(parent, "marker", attrs)


def _serialize(root) -> str:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n"
        f"{body}\n"
    )


def markers_document(asset_path: str, fps: float, markers, title: str | None = None,
                     duration: float | None = None) -> str:
    """Timeline = 1 clip source + markers (beats audio).

    Args:
        asset_path : fichier audio ou vidéo référencé.
        fps : fps de la timeline (24 par défaut).
        markers : liste de dicts {time, value, note}.
        duration : durée de la timeline (sinon durée du média).
        title : nom du projet Resolve.
    """
    title = title or os.path.splitext(os.path.basename(asset_path))[0]
    markers = list(markers or [])
    if duration is None:
        last = markers[-1]["time"] if markers else 0.0
        duration = max(last + 1.0, 1.0)

    root, spine, info, asset_duration, timeline_duration = _base_document(
        title, asset_path, fps, duration
    )
    clip = ET.SubElement(spine, "asset-clip", {
        "ref": "r2",
        "offset": "0s",
        "start": "0s",
        "duration": _time(_frames(timeline_duration, fps), fps),
        "name": os.path.basename(asset_path),
        "audioRole": "dialogue",
    })
    for marker in markers:
        t = float(marker.get("time") or 0.0)
        if t > timeline_duration:
            continue
        _add_marker(clip, t, fps, str(marker.get("value") or "Marker"),
                    str(marker.get("note") or ""))

    return _serialize(root)


def _append_cut_clips(spine, video_path: str, fps: float, cuts,
                      asset_duration: float) -> int:
    """Ajoute un asset-clip par plan et retourne la durée totale en frames."""
    offset_frames = 0
    for index, cut in enumerate(cuts):
        start_frames = _frames(float(cut.get("start") or 0.0), fps)
        dur_frames = max(1, _frames(float(cut.get("duration") or 0.0), fps))
        # Ne jamais dépasser le média source
        if asset_duration and (start_frames + dur_frames) > _frames(asset_duration, fps):
            dur_frames = max(1, _frames(asset_duration, fps) - start_frames)
        ET.SubElement(spine, "asset-clip", {
            "ref": "r2",
            "offset": _time(offset_frames, fps),
            "start": _time(start_frames, fps),
            "duration": _time(dur_frames, fps),
            "name": f"{os.path.splitext(os.path.basename(video_path))[0]} — plan {index + 1}",
        })
        offset_frames += dur_frames
    return offset_frames


def cuts_document(video_path: str, fps: float, cuts, title: str | None = None) -> str:
    """Timeline découpée : un asset-clip par plan détecté.

    Args:
        video_path : média source.
        fps : fps de la timeline.
        cuts : liste de dicts {start, duration}.
    """
    title = title or os.path.splitext(os.path.basename(video_path))[0]
    cuts = list(cuts or [])
    total = sum(float(c.get("duration") or 0.0) for c in cuts)
    if total <= 0:
        info = probe(video_path)
        total = float(info.get("duration") or 1.0)

    root, spine, info, asset_duration, _ = _base_document(
        title, video_path, fps, total
    )
    # La durée de séquence doit correspondre à la somme des plans
    sequence = root.find("./library/event/project/sequence")
    offset_frames = _append_cut_clips(spine, video_path, fps, cuts,
                                      asset_duration)

    if sequence is not None:
        sequence.set("duration", _time(offset_frames or _frames(total, fps), fps))

    return _serialize(root)


def rhythm_document(video_path: str, fps: float, cuts, title: str | None = None,
                    audio_path: str | None = None,
                    audio_duration: float | None = None,
                    include_audio: bool = True) -> str:
    """Timeline « montage au rythme » : plans vidéo + musique liée.

    La musique est placée sur une piste connectée (``lane="-1"``), alignée sur
    le début de la timeline. Elle est optionnelle : si l'import d'une piste
    liée pose problème dans une version de Resolve, l'export sans musique
    reste un FCPXML de coupes valide.

    Args:
        video_path : rush découpé.
        fps : fps de la timeline.
        cuts : liste de dicts {start, duration} (start = in-point source).
        title : nom du projet.
        audio_path : musique à poser sous les plans.
        audio_duration : durée de la musique (sinon ffprobe).
        include_audio : False = timeline vidéo seule.
    """
    title = title or os.path.splitext(os.path.basename(video_path))[0]
    cuts = list(cuts or [])
    total = sum(float(c.get("duration") or 0.0) for c in cuts)
    if total <= 0:
        total = float(probe(video_path).get("duration") or 1.0)

    root, spine, info, asset_duration, _ = _base_document(
        title, video_path, fps, total
    )
    resources = root.find("./resources")
    sequence = root.find("./library/event/project/sequence")

    offset_frames = _append_cut_clips(spine, video_path, fps, cuts,
                                      asset_duration)

    audio_frames = 0
    if include_audio and audio_path and resources is not None:
        audio_info = probe(audio_path)
        audio_len = float(audio_duration or audio_info.get("duration") or 0.0)
        if audio_len > 0:
            audio_asset = ET.SubElement(resources, "asset", {
                "id": "r3",
                "name": os.path.basename(audio_path) or "music",
                "start": "0s",
                "duration": _time(_frames(audio_len, fps), fps),
                "hasVideo": "0",
                "hasAudio": "1",
            })
            ET.SubElement(audio_asset, "media-rep", {
                "kind": "original-media",
                "src": _file_url(audio_path),
            })
            # Piste connectée sous la spine, alignée sur 0.
            audio_frames = min(_frames(audio_len, fps), offset_frames)
            ET.SubElement(spine, "asset-clip", {
                "ref": "r3",
                "lane": "-1",
                "offset": "0s",
                "start": "0s",
                "duration": _time(audio_frames, fps),
                "name": os.path.basename(audio_path) or "music",
                "audioRole": "music",
            })

    if sequence is not None:
        sequence.set("duration",
                     _time(offset_frames or _frames(total, fps), fps))

    return _serialize(root)
