"""Sondage des fichiers média via ffprobe.

Fournit durée, résolution, fps et présence des pistes vidéo/audio.
Utilisé pour :
  - calculer les estimations de durée de traitement ;
  - construire des timelines FCPXML correctes (format, durée d'asset).

100 % local (ffprobe), avec cache mémoire indexé sur (chemin, mtime, taille).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from fractions import Fraction

# Extensions reconnues
VIDEO_EXTS = {
    ".mp4", ".m4v", ".mkv", ".avi", ".mov", ".mts", ".m2ts", ".mxf",
    ".ts", ".vob", ".webm", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp",
}
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".alac",
}

_CACHE: dict[tuple, dict] = {}
_LOCK = threading.Lock()

DEFAULT_PROBE = {
    "duration": 0.0,
    "width": 0,
    "height": 0,
    "fps": 24.0,
    "has_video": False,
    "has_audio": False,
    "kind": None,
    "start_time": 0.0,
    "video_start": None,
    "audio_start": None,
}


def media_kind(path: str) -> str | None:
    """Retourne 'video', 'audio' ou None selon l'extension."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def is_media_file(path: str) -> bool:
    return os.path.isfile(path) and media_kind(path) is not None


def _parse_rate(rate: str | None) -> float:
    """Convertit '30000/1001' → 29.97, '25/1' → 25.0."""
    if not rate:
        return 0.0
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str, timeout: float = 20.0) -> dict:
    """Sonde un fichier média avec ffprobe (résultat mis en cache).

    Retourne toujours un dict (valeurs par défaut si ffprobe échoue).
    """
    path = os.path.abspath(str(path))
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return dict(DEFAULT_PROBE, path=path, error="fichier introuvable")

    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    info = dict(DEFAULT_PROBE, path=path, error=None)
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        data = json.loads(out.stdout or "{}")
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        info["error"] = str(exc)
        return info

    fmt = data.get("format") or {}
    try:
        info["duration"] = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info["duration"] = 0.0
    try:
        info["start_time"] = float(fmt.get("start_time") or 0.0)
    except (TypeError, ValueError):
        info["start_time"] = 0.0

    for stream in data.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            # Ignore les vignettes de couverture (image fixe attachée)
            if stream.get("disposition", {}).get("attached_pic") == 1:
                continue
            # Capture le début de la piste vidéo (référence temporelle).
            if info.get("video_start") is None:
                try:
                    info["video_start"] = float(stream.get("start_time") or 0.0)
                except (TypeError, ValueError):
                    info["video_start"] = 0.0
            if not info["has_video"]:
                info["has_video"] = True
                info["width"] = int(stream.get("width") or 0)
                info["height"] = int(stream.get("height") or 0)
                info["fps"] = _parse_rate(stream.get("avg_frame_rate")) or \
                    _parse_rate(stream.get("r_frame_rate")) or 24.0
                if not info["duration"]:
                    try:
                        info["duration"] = float(stream.get("duration") or 0.0)
                    except (TypeError, ValueError):
                        pass
        elif codec_type == "audio":
            info["has_audio"] = True
            if info.get("audio_start") is None:
                try:
                    info["audio_start"] = float(stream.get("start_time") or 0.0)
                except (TypeError, ValueError):
                    info["audio_start"] = 0.0

    if info["has_video"]:
        info["kind"] = "video"
    elif info["has_audio"]:
        info["kind"] = "audio"
    else:
        info["kind"] = media_kind(path)

    with _LOCK:
        _CACHE[key] = info
    return info


def probe_many(paths) -> list[dict]:
    """Sonde une liste de chemins (ignore les erreurs)."""
    return [probe(p) for p in paths]


def fps_exact(fps: float):
    """Retourne une Fraction exacte pour un fps (gère 29.97, 23.976…)."""
    # Tolérance sur les fps NTSC classiques
    known = {
        23.976: Fraction(24000, 1001),
        29.97: Fraction(30000, 1001),
        59.94: Fraction(60000, 1001),
        47.952: Fraction(48000, 1001),
    }
    for value, frac in known.items():
        if abs(float(fps) - value) < 0.01:
            return frac
    return Fraction(float(fps)).limit_denominator(100000)
