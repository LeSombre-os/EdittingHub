"""Découpage Automatique de Plans — utilise PySceneDetect en local.

Analyse un fichier vidéo brut pour détecter les changements de plans
(cuts) et exporte le résultat sous :
  - EDL (format CMX 3600, importable par DaVinci Resolve) ;
  - XML (FCPXML, importable par DaVinci Resolve) ;
  - CSV simple (timecodes des coupes).

100 % local, zéro dépendance cloud.
"""
import os
import shutil
import time

from app.config import DEFAULT_FPS
from app.errors import JobCancelled
from app.utils import seconds_to_edl_timecode


def _open_video_stream(video_path):
    """Ouvre un flux vidéo avec le backend OpenCV disponible.

    scenedetect 0.7.x : VideoStream est abstrait, il faut utiliser
    AVAILABLE_BACKENDS['opencv'] (qui pointe vers VideoStreamCv2).
    """
    from scenedetect import AVAILABLE_BACKENDS

    for backend_name, backend_class in AVAILABLE_BACKENDS.items():
        try:
            video = backend_class(video_path)
            # Vérifie que le flux est valide
            if video.duration is not None and video.duration > 0:
                return video
        except Exception:
            continue
    raise RuntimeError(
        f"Impossible d'ouvrir la vidéo (aucun backend disponible) : {video_path}"
    )


def _detect(video_path, threshold, min_scene_len, downscale_factor,
            progress_cb=None, cancel_event=None):
    """Détecte les scènes avec scenedetect 0.7.x.

    progress_cb(frac, frame_num, total_frames) est appelée à chaque frame :
    l'annulation est ainsi effective en quelques dizaines de millisecondes.

    Retourne une liste de dictionnaires : [{start, end, start_tc, end_tc}, ...]
    """
    from scenedetect import SceneManager
    from scenedetect.detectors import ContentDetector

    video = _open_video_stream(video_path)
    if video.duration is None or video.duration == 0:
        raise RuntimeError("Impossible de lire la vidéo (durée nulle).")

    fps = float(video.frame_rate or DEFAULT_FPS)
    # En scenedetect 0.7.x, `duration` est un FrameTimecode, pas un float.
    try:
        duration_s = float(video.duration.seconds)
    except AttributeError:
        duration_s = float(video.duration or 0.0)
    total_frames = int(duration_s * fps) if duration_s else 0

    scene_manager = SceneManager()
    # min_scene_len en frames (scenedetect l'attend en frames)
    min_frames = int(min_scene_len * fps) if isinstance(min_scene_len, (int, float)) else min_scene_len
    scene_manager.add_detector(
        ContentDetector(
            threshold=float(threshold),
            min_scene_len=max(1, min_frames),
        )
    )

    # Downscale pour accélérer (1 = native, >1 = réduit)
    # En 0.7.x, downscale est un attribut entier (pas une méthode)
    # et auto_downscale doit être désactivé pour que downscale soit pris en compte.
    if downscale_factor > 1:
        scene_manager.auto_downscale = False
        scene_manager.downscale = downscale_factor

    def _on_frame(frame_img, frame_timecode):
        # Abandon immédiat si l'utilisateur a cliqué sur « Arrêter »
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled("Détection annulée par l'utilisateur.")
        if progress_cb and total_frames:
            try:
                frame_num = int(frame_timecode)
            except (TypeError, ValueError):
                frame_num = getattr(frame_timecode, "frame_num", 0)
            progress_cb(min(1.0, frame_num / total_frames), frame_num, total_frames)

    scene_manager.detect_scenes(video, show_progress=False, callback=_on_frame)
    scene_list = scene_manager.get_scene_list()
    # Note : VideoStreamCv2 n'a pas de méthode close() en 0.7.x —
    # la libération se fait par le garbage collector.

    # Conversion des timecodes scenedetect → secondes → timecode EDL
    cuts = []
    for start_tc, end_tc in scene_list:
        start_s = start_tc.seconds if hasattr(start_tc, "seconds") else start_tc.get_seconds()
        end_s = end_tc.seconds if hasattr(end_tc, "seconds") else end_tc.get_seconds()
        cuts.append({
            "start": round(start_s, 3),
            "end": round(end_s, 3),
            "start_tc": seconds_to_edl_timecode(start_s, fps),
            "end_tc": seconds_to_edl_timecode(end_s, fps),
            "duration": round(end_s - start_s, 3),
        })

    return cuts, fps


def detect_scenes(video_path, threshold=30.0, min_scene_len=15,
                  downscale_factor=2, progress_cb=None, cancel_event=None):
    """Point d'entrée du service de détection de scènes.

    Args:
        video_path : chemin vers le fichier vidéo
        threshold : seuil de détection de changement (0-100, défaut 30)
        min_scene_len : durée minimale d'une scène en secondes
        downscale_factor : réduction d'image pour accélérer (1=native, 2=½)
        progress_cb : callback(frac, frame_num, total_frames)
        cancel_event : threading.Event (annulation immédiate)

    Returns:
        dict avec : cuts, fps, duration, total_cuts
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Fichier vidéo introuvable : {video_path}")
    if not shutil.which("ffmpeg") and not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe requis mais introuvable.")
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Détection annulée avant démarrage.")

    t0 = time.time()
    cuts, fps = _detect(video_path, threshold, min_scene_len, downscale_factor,
                        progress_cb=progress_cb, cancel_event=cancel_event)
    elapsed = time.time() - t0

    # Durée totale de la vidéo
    total_duration = cuts[-1]["end"] if cuts else 0.0

    return {
        "status": "done",
        "cuts": cuts,
        "fps": round(fps, 3),
        "total_cuts": len(cuts),
        "duration": round(total_duration, 3),
        "elapsed": round(elapsed, 2),
        "video": os.path.basename(video_path),
    }


# ── Export EDL (CMX 3600) ──────────────────────────────────────────────────

def export_edl(video_path, cuts, fps=24.0, title=None):
    """Génère un EDL CMX 3600 (format importable par DaVinci Resolve).

    Chaque cut devient un événement AX (clip) dans la séquence.
    Le dernier cut est élargi jusqu'à la fin de la vidéo.
    """
    title = title or os.path.splitext(os.path.basename(video_path))[0]
    video_name = os.path.basename(video_path)

    lines = []
    lines.append(f"TITLE: {title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    if not cuts:
        lines.append("001  BL       V     C        00:00:00:00 00:00:00:01")
        lines.append(f"     * FROM CLIP NAME: {video_name}")
        lines.append("")
        return "\n".join(lines)

    for i, cut in enumerate(cuts, 1):
        num = f"{i:03d}"
        start_tc = cut["start_tc"]
        end_tc = cut["end_tc"]
        lines.append(f"{num}  AX       V     C        {start_tc} {end_tc}")
        lines.append(f"     * FROM CLIP NAME: {video_name}")
        lines.append("")

    return "\n".join(lines)


# ── Export XML (FCPXML) ────────────────────────────────────────────────────

def export_xml(video_path, cuts, fps=24.0, title=None):
    """Génère une timeline FCPXML 1.9 découpée (importable par Resolve).

    Un `<asset-clip>` par plan, placé sur la spine, avec offset/start/duration
    explicites. L'ancienne version écrivait `<clip>` + `<media><video><ref>`,
    ce qui n'est pas du FCPXML valide : Resolve importait une timeline vide.
    """
    from app.fcpxml import cuts_document
    return cuts_document(video_path, fps, cuts, title=title)


# ── Export CSV simple ──────────────────────────────────────────────────────

def export_csv(cuts, fps=24.0):
    """Génère un CSV simple : start_s,end_s,duration_s,start_tc,end_tc."""
    lines = ["start_s,end_s,duration_s,start_tc,end_tc"]
    for cut in cuts:
        lines.append(
            f"{cut['start']},{cut['end']},{cut['duration']},"
            f"{cut['start_tc']},{cut['end_tc']}"
        )
    return "\n".join(lines)


# ── Fonction unifiée d'export ────────────────────────────────────────────────

def export_cuts(video_path, cuts, fps, fmt="edl"):
    """Exporte les coupes dans le format demandé.

    fmt : 'edl' | 'xml' | 'fcpxml' | 'csv'
    Retourne le contenu texte.
    """
    if fmt == "edl":
        return export_edl(video_path, cuts, fps)
    elif fmt == "xml" or fmt == "fcpxml":
        return export_xml(video_path, cuts, fps)
    elif fmt == "csv":
        return export_csv(cuts, fps)
    else:
        raise ValueError(f"Format d'export inconnu : {fmt}")
