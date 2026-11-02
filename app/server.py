#!/usr/bin/env python3
"""EdittingHub — serveur FastAPI local.

Expose une API REST et sert l'interface web statique.
8005 par défaut. 100 % local, zéro cloud.

Endpoints :
  GET  /                              → index.html
  GET  /api/ping                     → santé du serveur
  GET  /api/config                   → config + outils disponibles
  GET  /api/ls?path=…                → browser de fichiers intégré
  GET  /api/pick?kind=…              → sélecteur NATIF (zenity), repli côté UI
  GET  /api/job/{id}                 → état d'un job (progression, ETA)
  POST /api/job/{id}/cancel          → annule un job (kill du groupe de process)

  POST /api/convert/scan             → plan de conversion + estimation de durée
  POST /api/convert/start            → lancer conversion (job)

  POST /api/subtitles/scan           → plan de sous-titrage + estimation
  POST /api/subtitles/start          → lancer transcription (job)

  POST /api/scenes/scan              → plan de détection (lot) + estimation
  POST /api/scenes/detect            → détection de plans (job, lot)
  POST /api/scenes/export            → export EDL/XML/CSV (texte ou ZIP)

  POST /api/audio/scan               → plan d'analyse (lot) + estimation
  POST /api/audio/analyze            → analyse de beats (job, lot)
  POST /api/audio/export             → export CSV/EDL/XML (texte ou ZIP)
"""
import io
import itertools
import os
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import (FileResponse, HTMLResponse, PlainTextResponse,
                               Response)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app import estimates
from app.config import (DEFAULT_FPS, HOST, PORT, WORKSPACE_ROOT,
                        check_external_tools)
from app.errors import JobCancelled
from app.media import AUDIO_EXTS, VIDEO_EXTS, probe
from app.utils import collect_files, fmt_duration, fmt_size
from app.utils import list_dir as util_list_dir

# ── Import des services (lazy : évite d'importer librosa/scenedetect au démarrage)
from app.services.audio_beats import analyze_audio, export_markers
from app.services.convert import ConvertService
from app.services.rhythm import build_cut_plan, export_rhythm
from app.services.scene_detect import detect_scenes, export_cuts
from app.services.subtitles import SubtitleService

# ── Application FastAPI ────────────────────────────────────────────────────
app = FastAPI(
    title="EdittingHub",
    description="Hub local 100% offline pour montage vidéo — conversion, "
                "sous-titrage, détection de plans, analyse audio.",
    version="1.4.0",
    docs_url=None,  # désactive l'UI Swagger (UI custom)
    redoc_url=None,
)

# Monte les fichiers statiques
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Gestionnaire de jobs (état en mémoire) ─────────────────────────────────
JOBS: dict[int, dict] = {}
_JOB_SEQ = itertools.count(1)
_JOBS_LOCK = threading.Lock()

# Un seul job lourd par module : évite de tuer le mauvais sous-processus
_RUNNING = {"convert": None, "subtitles": None}


def create_job(job_type: str, config: dict, estimate: dict | None = None) -> int:
    """Crée un nouveau job et retourne son ID."""
    with _JOBS_LOCK:
        jid = next(_JOB_SEQ)
        JOBS[jid] = {
            "id": jid,
            "type": job_type,
            "status": "queued",
            "progress": 0.0,
            "done": 0,
            "total": 0,
            "logs": [],
            "result": None,
            "error": None,
            "config": config,
            "estimate": estimate,
            "eta": None,
            "started": None,
            "ended": None,
            "_cancel": threading.Event(),
        }
        return jid


def get_job(jid: int) -> Optional[dict]:
    with _JOBS_LOCK:
        return JOBS.get(jid)


def cancel_job(jid: int) -> bool:
    """Demande l'arrêt d'un job.

    Le statut passe à « cancelling » immédiatement (l'UI affiche alors
    « Arrêt en cours… »), puis le thread du job confirme « cancelled ».
    Pour les modules à sous-processus, le groupe de processus est tué tout de
    suite (ffmpeg / whisper inclus) : plus rien ne tourne au fond.
    """
    with _JOBS_LOCK:
        job = JOBS.get(jid)
        if not job:
            return False
        job["_cancel"].set()
        if job["status"] in ("queued", "running"):
            job["status"] = "cancelling"
        job_type = job["type"]

    if job_type == "convert":
        _convert_svc.terminate()
    elif job_type == "subtitles":
        _subtitle_svc.terminate()
    return True


def update_job(jid: int, **kwargs):
    """Met à jour les champs du job (thread-safe)."""
    with _JOBS_LOCK:
        job = JOBS.get(jid)
        if job:
            # Champs protégés (pas de remplacement direct)
            for k, v in kwargs.items():
                if k == "_cancel":
                    continue
                job[k] = v


def append_log(jid: int, line: str, limit: int = 600):
    """Ajoute une ligne de log en bornant la mémoire."""
    with _JOBS_LOCK:
        job = JOBS.get(jid)
        if not job:
            return
        logs = job["logs"]
        logs.append(line)
        if len(logs) > limit:
            del logs[: len(logs) - limit]


def job_public_view(job: dict) -> dict:
    """Retire les champs internes (_cancel) pour l'API."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _running_job(job_type: str) -> Optional[int]:
    """Retourne l'ID d'un job encore actif pour ce module (ou None)."""
    with _JOBS_LOCK:
        jid = _RUNNING.get(job_type)
        if jid is None:
            return None
        job = JOBS.get(jid)
        if job and job["status"] in ("queued", "running", "cancelling"):
            return jid
        _RUNNING[job_type] = None
        return None


def _claim_slot(job_type: str, jid: int):
    with _JOBS_LOCK:
        _RUNNING[job_type] = jid


def _release_slot(job_type: str, jid: int):
    with _JOBS_LOCK:
        if _RUNNING.get(job_type) == jid:
            _RUNNING[job_type] = None


# ── Instances de services ───────────────────────────────────────────────────
_convert_svc = ConvertService()
_subtitle_svc = SubtitleService()


# ── Modèles Pydantic (request bodies) ───────────────────────────────────────

class ScanRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    target: str = "resolve"
    preset: str = "sq"
    keep_4k: bool = False


class ConvertRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    target: str = "resolve"
    preset: str = "sq"
    keep_4k: bool = False
    force: bool = False
    jobs: int = 4

    @field_validator("target")
    @classmethod
    def validate_target(cls, v):
        if v not in ("resolve", "tiktok"):
            raise ValueError("target doit être 'resolve' ou 'tiktok'")
        return v

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, v):
        if v not in ("lb", "sq", "hq"):
            raise ValueError("preset doit être 'lb', 'sq' ou 'hq'")
        return v


class SubtitleScanRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    language: str = "auto"
    model: str = "small"
    offset: float = 0.0
    device: str = "auto"


class SubtitleRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    language: str = "auto"
    model: str = "small"
    offset: float = 0.0
    device: str = "cpu"
    max_chars: int = 42
    max_lines: int = 2
    mode: str = "phrases"
    words_per_cue: int = 1
    music: bool = False
    prompt: Optional[str] = None
    force: bool = False
    jobs: int = 1

    @field_validator("language")
    @classmethod
    def validate_lang(cls, v):
        if v not in ("auto", "fr", "en"):
            raise ValueError("language doit être 'auto', 'fr' ou 'en'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in ("base", "small", "medium", "turbo", "large-v3"):
            raise ValueError("model invalide")
        return v


class SceneScanRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    threshold: float = 30.0
    min_scene_len: float = 1.0
    downscale_factor: int = 2


class SceneRequest(SceneScanRequest):
    # Compatibilité avec l'ancien champ unique
    paths: list[str] = Field(default_factory=list)
    video_path: Optional[str] = None

    def all_paths(self) -> list[str]:
        if self.paths:
            return list(self.paths)
        return [self.video_path] if self.video_path else []


class AudioScanRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    sample_rate: int = 22050


class AudioRequest(AudioScanRequest):
    paths: list[str] = Field(default_factory=list)
    audio_path: Optional[str] = None
    # Décalage manuel appliqué aux marqueurs exportés (ms). Positif = plus tard.
    offset_ms: Optional[float] = 0.0

    def all_paths(self) -> list[str]:
        if self.paths:
            return list(self.paths)
        return [self.audio_path] if self.audio_path else []


class ExportRequest(BaseModel):
    job_id: int
    fmt: str = "fcpxml"  # edl | xml | fcpxml | csv | davinci_script
    fps: float = 24.0
    file_index: Optional[int] = None  # None + lot → ZIP
    ranks: Optional[list[int]] = None  # filtre rangs (1,2,3) — None = tous


# ── Aides estimation / lots ────────────────────────────────────────────────

def _expand(paths, extensions) -> list[str]:
    """Développe fichiers/dossiers en liste de fichiers réels."""
    files = collect_files(paths, extensions=extensions)
    return files


def _file_card(path: str) -> dict:
    """Métadonnées d'un fichier pour les écrans de scan."""
    info = probe(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    duration = float(info.get("duration") or 0.0)
    return {
        "path": path,
        "name": os.path.basename(path),
        "duration": round(duration, 3),
        "duration_human": fmt_duration(duration),
        "size": size,
        "size_str": fmt_size(size),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "kind": info.get("kind"),
    }


def _scan_payload(paths, extensions, module: str, key: str) -> dict:
    """Construit la réponse d'un scan : fichiers + estimation de durée."""
    files = _expand(paths, extensions)
    cards = [_file_card(p) for p in files]
    total = sum(c["duration"] for c in cards)
    est = estimates.estimate(module, key, total, files=max(1, len(cards)))
    est["total_duration"] = round(total, 3)
    est["total_duration_human"] = fmt_duration(total)
    return {
        "files": cards,
        "count": len(cards),
        "total_duration": round(total, 3),
        "total_duration_human": fmt_duration(total),
        "estimate": est,
    }


def _convert_key(target: str, preset: str) -> str:
    return f"{target}:{preset}"


def _subtitle_key(model: str, device: str) -> str:
    dev = "cpu" if device in ("cpu", "auto") else "gpu"
    return f"{model}:{dev}"


def _scene_key(downscale_factor: int) -> str:
    return f"ds{max(1, int(downscale_factor or 1))}"


def _cleanup_partials(paths, target: str = "resolve") -> list[str]:
    """Supprime les fichiers `.part-*` laissés par une conversion interrompue.

    convert.py écrit d'abord `<sortie>/.part-<nom>.mov` puis renomme à la fin.
    Un arrêt brutal laisse donc un gros fichier temporaire : on le retire.
    """
    removed = []
    outdir_name = "CONVERTIS" if target == "resolve" else "TIKTOK"
    for src in paths:
        if not os.path.isfile(src):
            continue
        outdir = os.path.join(os.path.dirname(src), outdir_name)
        if not os.path.isdir(outdir):
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        for name in os.listdir(outdir):
            if not name.startswith(".part-") or stem not in name:
                continue
            try:
                os.unlink(os.path.join(outdir, name))
                removed.append(os.path.join(outdir, name))
            except OSError:
                pass
    return removed


# ── Fonctions de travail (background threads) ──────────────────────────────

def _run_convert_job(job_id: int):
    """Thread de conversion : lance convert.py via subprocess."""
    job = get_job(job_id)
    if not job:
        return
    cfg = job["config"]
    update_job(job_id, status="running", started=time.time())

    def progress_cb(done, total, frac, line, stats):
        if job.get("_cancel").is_set():
            return
        update_job(job_id, progress=frac, done=done, total=total)
        append_log(job_id, line)

    try:
        result = _convert_svc.run(
            paths=cfg["paths"],
            target=cfg.get("target", "resolve"),
            preset=cfg.get("preset", "sq"),
            keep_4k=cfg.get("keep_4k", False),
            force=cfg.get("force", False),
            jobs=cfg.get("jobs", 4),
            cancel_event=job.get("_cancel"),
            progress_callback=progress_cb,
        )
        ended = time.time()
        if result["status"] in ("cancelled", "error"):
            # Un arrêt brutal laisse un `.part-*.mov` de plusieurs Go : on nettoie
            for path in _cleanup_partials(cfg["paths"], cfg.get("target", "resolve")):
                append_log(job_id, f"[NETTOYAGE] fichier partiel supprimé : {path}")
        update_job(job_id,
                   status=result["status"],
                   result=result,
                   error=result.get("error"),
                   ended=ended)
        if result["status"] == "done":
            media = sum(float(probe(p).get("duration") or 0.0)
                        for p in cfg["paths"] if os.path.isfile(p))
            estimates.record("convert",
                             _convert_key(cfg.get("target", "resolve"),
                                          cfg.get("preset", "sq")),
                             media, ended - (job.get("started") or ended))
    except Exception as e:
        update_job(job_id, status="error",
                   error=str(e), ended=time.time())
        append_log(job_id, f"[ERREUR] {e}")
    finally:
        _release_slot("convert", job_id)


def _run_subtitle_job(job_id: int):
    """Thread de sous-titrage : lance transcrire.py via subprocess."""
    job = get_job(job_id)
    if not job:
        return
    cfg = job["config"]
    update_job(job_id, status="running", started=time.time())

    def progress_cb(done, total, frac, line, stats):
        if job.get("_cancel").is_set():
            return
        update_job(job_id, progress=frac, done=done, total=total)
        append_log(job_id, line, limit=2000)

    try:
        result = _subtitle_svc.run(
            paths=cfg["paths"],
            language=cfg.get("language", "auto"),
            model=cfg.get("model", "small"),
            offset=cfg.get("offset", 0.0),
            device=cfg.get("device", "auto"),
            max_chars=cfg.get("max_chars", 42),
            max_lines=cfg.get("max_lines", 2),
            mode=cfg.get("mode", "phrases"),
            words_per_cue=cfg.get("words_per_cue", 1),
            music=cfg.get("music", False),
            prompt=cfg.get("prompt"),
            force=cfg.get("force", False),
            jobs=cfg.get("jobs", 1),
            cancel_event=job.get("_cancel"),
            progress_callback=progress_cb,
        )
        ended = time.time()
        update_job(job_id,
                   status=result["status"],
                   result=result,
                   error=result.get("error"),
                   ended=ended)
        if result["status"] == "done":
            media = sum(float(probe(p).get("duration") or 0.0)
                        for p in cfg["paths"] if os.path.isfile(p))
            estimates.record("subtitles",
                             _subtitle_key(cfg.get("model", "small"),
                                           cfg.get("device", "auto")),
                             media, ended - (job.get("started") or ended))
    except Exception as e:
        update_job(job_id, status="error",
                   error=str(e), ended=time.time())
        append_log(job_id, f"[ERREUR] {e}")
    finally:
        _release_slot("subtitles", job_id)


def _run_scene_job(job_id: int):
    """Thread de détection de plans (lot) : scenedetect directement."""
    job = get_job(job_id)
    if not job:
        return
    cfg = job["config"]
    cancel = job.get("_cancel")
    files = _expand(cfg.get("paths") or [cfg.get("video_path")], VIDEO_EXTS)
    started = time.time()
    update_job(job_id, status="running", started=started, total=len(files))

    ds = int(cfg.get("downscale_factor", 2) or 2)
    entries: list[dict] = []
    durations = [float(probe(f).get("duration") or 0.0) for f in files]
    total_media = sum(durations) or 1.0
    tracker = estimates.EtaTracker(total_media)
    done_media = 0.0

    try:
        if not files:
            raise RuntimeError("Aucun fichier vidéo valide dans la sélection.")

        for idx, path in enumerate(files):
            if cancel.is_set():
                raise JobCancelled("Détection annulée.")
            name = os.path.basename(path)
            media = durations[idx]
            last_log = [0.0]
            append_log(job_id, f"[{idx + 1}/{len(files)}] {name} "
                               f"({fmt_duration(media)})")

            def progress_cb(frac, frame_num, total_frames,
                            _idx=idx, _name=name, _media=media,
                            _base=done_media, _last=last_log):
                if cancel.is_set():
                    return
                frac = max(0.0, min(1.0, float(frac or 0.0)))
                overall = (_idx + frac) / max(1, len(files))
                info = tracker.update(_base + _media * frac)
                update_job(job_id, progress=round(overall, 4), eta=info,
                           done=_idx, total=len(files))
                now = time.time()
                if now - _last[0] >= 2.0 and total_frames:
                    _last[0] = now
                    append_log(
                        job_id,
                        f"    {_name} : {frac * 100:5.1f}% "
                        f"({frame_num}/{total_frames} frames)"
                        + (f" — reste ~{info.get('eta_human')}"
                           if info.get("eta_human") else ""),
                    )

            result = detect_scenes(
                video_path=path,
                threshold=cfg.get("threshold", 30.0),
                min_scene_len=cfg.get("min_scene_len", 1.0),
                downscale_factor=ds,
                progress_cb=progress_cb,
                cancel_event=cancel,
            )
            estimates.record("scenes", _scene_key(ds),
                             result.get("duration") or media,
                             result.get("elapsed") or 0.0)
            entry = dict(result)
            entry.update({"path": path, "name": name})
            entries.append(entry)
            done_media += media
            append_log(job_id, f"    → {result['total_cuts']} plans "
                               f"en {result['elapsed']}s")

        ended = time.time()
        result = {
            "status": "done",
            "files": entries,
            "count": len(entries),
            "total_cuts": sum(e.get("total_cuts", 0) for e in entries),
            "duration": round(sum(float(e.get("duration") or 0.0)
                                  for e in entries), 3),
            "elapsed": round(ended - started, 2),
            # Compatibilité mono-fichier
            "cuts": entries[0]["cuts"] if len(entries) == 1 else [],
            "fps": entries[0]["fps"] if len(entries) == 1 else DEFAULT_FPS,
            "video_path": files[0] if len(files) == 1 else None,
        }
        update_job(job_id, status="done", result=result, progress=1.0,
                   done=len(files), total=len(files), ended=ended)
        append_log(job_id, f"Terminé : {result['total_cuts']} plans sur "
                           f"{result['count']} fichier(s) en "
                           f"{result['elapsed']}s")
    except JobCancelled as e:
        update_job(job_id, status="cancelled", ended=time.time(), error=None)
        append_log(job_id, f"[ANNULÉ] {e}")
    except Exception as e:
        update_job(job_id, status="error", error=str(e), ended=time.time())
        append_log(job_id, f"[ERREUR] {e}")


def _run_audio_job(job_id: int):
    """Thread d'analyse audio (lot) : librosa directement."""
    job = get_job(job_id)
    if not job:
        return
    cfg = job["config"]
    cancel = job.get("_cancel")
    files = _expand(cfg.get("paths") or [cfg.get("audio_path")],
                    AUDIO_EXTS | VIDEO_EXTS)
    started = time.time()
    update_job(job_id, status="running", started=started, total=len(files))

    entries: list[dict] = []
    durations = [float(probe(f).get("duration") or 0.0) for f in files]
    tracker = estimates.EtaTracker(sum(durations) or 1.0)
    done_media = 0.0

    try:
        if not files:
            raise RuntimeError("Aucun fichier audio/vidéo valide dans la sélection.")

        for idx, path in enumerate(files):
            if cancel.is_set():
                raise JobCancelled("Analyse annulée.")
            name = os.path.basename(path)
            media = durations[idx]
            append_log(job_id, f"[{idx + 1}/{len(files)}] {name} "
                               f"({fmt_duration(media)})")

            def progress_cb(frac, label, _idx=idx, _media=media,
                            _base=done_media):
                if cancel.is_set():
                    return
                frac = max(0.0, min(1.0, float(frac or 0.0)))
                overall = (_idx + frac) / max(1, len(files))
                info = tracker.update(_base + _media * frac)
                update_job(job_id, progress=round(overall, 4), eta=info,
                           done=_idx, total=len(files))
                append_log(job_id, f"    {label}")

            result = analyze_audio(
                audio_path=path,
                sr=cfg.get("sample_rate", 22050),
                progress_cb=progress_cb,
                cancel_event=cancel,
                offset=(float(cfg.get("offset_ms") or 0.0) / 1000.0),
            )
            estimates.record("audio", "default",
                             result.get("duration") or media,
                             result.get("elapsed") or 0.0)
            entry = dict(result)
            entry.update({"path": path, "name": name})
            entries.append(entry)
            done_media += media
            append_log(job_id, f"    → {result['total_beats']} beats, "
                               f"{result['tempo']} BPM")

        ended = time.time()
        ranks = {1: 0, 2: 0, 3: 0}
        for e in entries:
            for rank, count in (e.get("ranks") or {}).items():
                ranks[int(rank)] = ranks.get(int(rank), 0) + count
        result = {
            "status": "done",
            "files": entries,
            "count": len(entries),
            "total_beats": sum(e.get("total_beats", 0) for e in entries),
            "duration": round(sum(float(e.get("duration") or 0.0)
                                  for e in entries), 3),
            "ranks": ranks,
            "elapsed": round(ended - started, 2),
            # Compatibilité mono-fichier
            "beats": entries[0]["beats"] if len(entries) == 1 else [],
            "tempo": entries[0]["tempo"] if len(entries) == 1 else None,
            "audio_path": files[0] if len(files) == 1 else None,
        }
        update_job(job_id, status="done", result=result, progress=1.0,
                   done=len(files), total=len(files), ended=ended)
        append_log(job_id, f"Terminé : {result['total_beats']} beats sur "
                           f"{result['count']} fichier(s) en "
                           f"{result['elapsed']}s")
    except JobCancelled as e:
        update_job(job_id, status="cancelled", ended=time.time(), error=None)
        append_log(job_id, f"[ANNULÉ] {e}")
    except Exception as e:
        update_job(job_id, status="error", error=str(e), ended=time.time())
        append_log(job_id, f"[ERREUR] {e}")


def _run_rhythm_job(job_id: int):
    """Thread du montage au rythme : analyse des beats puis plan de coupe."""
    job = get_job(job_id)
    if not job:
        return
    cfg = job["config"]
    cancel = job.get("_cancel")
    video_path = cfg["video_path"]
    audio_path = cfg["audio_path"]
    started = time.time()
    update_job(job_id, status="running", started=started, total=1)

    try:
        vinfo = probe(video_path)
        ainfo = probe(audio_path)
        video_duration = float(vinfo.get("duration") or 0.0)
        audio_duration = float(ainfo.get("duration") or 0.0)
        if video_duration <= 0:
            raise RuntimeError("Durée vidéo inconnue (ffprobe).")
        if audio_duration <= 0:
            raise RuntimeError("Durée audio inconnue (ffprobe).")

        append_log(job_id, f"Vidéo   : {os.path.basename(video_path)} "
                           f"({fmt_duration(video_duration)})")
        append_log(job_id, f"Musique : {os.path.basename(audio_path)} "
                           f"({fmt_duration(audio_duration)})")
        append_log(job_id, f"Rangs   : {', '.join(str(r) for r in cfg['ranks'])} "
                           f"| plans {cfg['min_shot']}–{cfg['max_shot']} s")

        last_log = [0.0]

        def progress_cb(frac, label):
            if cancel.is_set():
                return
            frac = max(0.0, min(1.0, float(frac or 0.0)))
            update_job(job_id, progress=round(frac * 0.9, 4))
            now = time.time()
            if now - last_log[0] >= 1.0:
                last_log[0] = now
                append_log(job_id, f"    {label}")

        analysis = analyze_audio(
            audio_path,
            sr=cfg.get("sample_rate", 22050),
            progress_cb=progress_cb,
            cancel_event=cancel,
        )
        if cancel.is_set():
            raise JobCancelled("Montage annulé.")
        append_log(job_id, f"    → {analysis['total_beats']} beats, "
                           f"{analysis['tempo']} BPM")
        update_job(job_id, progress=0.92)

        plan = build_cut_plan(
            analysis.get("beats") or [],
            video_duration=video_duration,
            audio_duration=audio_duration,
            ranks=cfg["ranks"],
            min_shot=cfg["min_shot"],
            max_shot=cfg["max_shot"],
            loop=cfg["loop"],
            fps=cfg["fps"],
            video_start=cfg.get("video_start", 0.0),
        )
        if not plan["cuts"]:
            raise RuntimeError(
                "Aucun plan généré : vérifiez les rangs cochés et la durée vidéo."
            )

        ended = time.time()
        result = {
            "video_path": video_path,
            "audio_path": audio_path,
            "video": os.path.basename(video_path),
            "audio": os.path.basename(audio_path),
            "fps": cfg["fps"],
            "ranks": cfg["ranks"],
            "beats_total": analysis["total_beats"],
            "beats_ranked": analysis.get("ranks") or {},
            "tempo": analysis.get("tempo"),
            "cuts": plan["cuts"],
            "points": plan["points"],
            "segments": plan["segments"],
            "clips": plan["clips"],
            "beats_used": plan["beats_used"],
            "timeline_duration": plan["timeline_duration"],
            "audio_duration": plan["audio_duration"],
            "video_duration": plan["video_duration"],
            "video_start": plan["video_start"],
            "loop": plan["loop"],
            "truncated": plan["truncated"],
            "min_shot": plan["min_shot"],
            "max_shot": plan["max_shot"],
            "elapsed": round(ended - started, 2),
        }
        update_job(job_id, status="done", result=result, progress=1.0,
                   done=1, total=1, ended=ended)
        append_log(job_id, f"Terminé : {plan['segments']} plans, "
                           f"{plan['timeline_duration']}s de timeline en "
                           f"{result['elapsed']}s")
    except JobCancelled as e:
        update_job(job_id, status="cancelled", ended=time.time(), error=None)
        append_log(job_id, f"[ANNULÉ] {e}")
    except Exception as e:
        update_job(job_id, status="error", error=str(e), ended=time.time())
        append_log(job_id, f"[ERREUR] {e}")


# ── Routes API ─────────────────────────────────────────────────────────────

# 1. Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def index():
    """Sert l'interface web."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return HTMLResponse("<h1>EdittingHub</h1><p>Démarrage…</p>", status_code=200)


# 2. Health check
@app.get("/api/ping")
async def ping():
    return {"ok": True, "timestamp": time.time()}


# 3. Configuration
@app.get("/api/config")
async def get_config():
    tools = check_external_tools()
    return {
        "version": app.version,
        "tools": tools,
        "port": PORT,
        "workspace": str(WORKSPACE_ROOT),
        "convert_script": str(_convert_svc.script) if _convert_svc.available() else None,
        "subtitle_script": _subtitle_svc.script if _subtitle_svc.available() else None,
        "subtitle_python": _subtitle_svc.python,
        "native_picker": _zenity() is not None,
    }


# 3.5. System resource usage
@app.get("/api/system")
async def system():
    """Returns basic system resource usage (CPU load, memory)."""
    import os
    data = {"cpu_load": None, "memory": None, "swap": None, "gpu": None}
    try:
        # Load average (1, 5, 15 min)
        if os.path.exists("/proc/loadavg"):
            with open("/proc/loadavg") as f:
                loads = f.read().split()
                data["cpu_load"] = {
                    "1min": float(loads[0]),
                    "5min": float(loads[1]),
                    "15min": float(loads[2]),
                }
        # Memory info
        if os.path.exists("/proc/meminfo"):
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        mem[key.strip()] = val.strip()
            # Extract MemTotal and MemAvailable (Linux) or MemFree
            mem_total = int(mem.get("MemTotal", "0 kB").split()[0])  # in kB
            mem_avail = int(mem.get("MemAvailable", "0 kB").split()[0])
            if mem_total > 0:
                used = mem_total - mem_avail
                data["memory"] = {
                    "total": mem_total,
                    "used": used,
                    "percent": round(used / mem_total * 100, 1),
                }
            # Swap
            swap_total = int(mem.get("SwapTotal", "0 kB").split()[0])
            swap_free = int(mem.get("SwapFree", "0 kB").split()[0])
            if swap_total > 0:
                used_swap = swap_total - swap_free
                data["swap"] = {
                    "total": swap_total,
                    "used": used_swap,
                    "percent": round(used_swap / swap_total * 100, 1),
                }
    except Exception:
        pass
    # GPU info via nvidia-smi if available
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                if line:
                    util_gpu, util_mem, temp = line.split(", ")
                    gpus.append(
                        {
                            "gpu_util": int(util_gpu),
                            "mem_util": int(util_mem),
                            "temp": int(temp),
                        }
                    )
            data["gpu"] = gpus
    except Exception:
        pass
    return data


# 4. Browser de fichiers intégré (repli si zenity absent)
@app.get("/api/ls")
async def ls_files(path: str = ""):
    result = util_list_dir(path or "")
    # Formate les tailles
    for e in result["entries"]:
        if not e["isdir"]:
            e["size_str"] = fmt_size(e["size"])
    return result


# 5. Sélecteur NATIF (zenity) — « Parcourir » ouvre le vrai explorateur
def _zenity() -> Optional[str]:
    return shutil.which("zenity")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@app.get("/api/pick")
def pick_files(kind: str = "files", multiple: bool = True,
               title: str = "Choisir des fichiers", exts: str = "",
               start: str = ""):
    """Ouvre le sélecteur de fichiers natif du bureau (zenity).

    Endpoint **synchrone** (def) : FastAPI l'exécute dans un threadpool, le
    serveur reste donc réactif pendant que la boîte de dialogue est ouverte.

    Retourne {ok, paths, cancelled, reason}. Si zenity ou l'affichage manque,
    ok=False et le frontend bascule sur son explorateur intégré.
    """
    zenity = _zenity()
    if not zenity:
        return {"ok": False, "reason": "zenity absent", "paths": []}
    if not _has_display():
        return {"ok": False, "reason": "pas de session graphique", "paths": []}

    cmd = [zenity, "--file-selection", f"--title={title}"]
    if multiple:
        cmd += ["--multiple", "--separator=\n"]
    if kind == "folder":
        cmd += ["--directory"]
    if start and os.path.exists(start):
        cmd += [f"--filename={start.rstrip('/')}/"]

    filters = []
    if exts:
        for group in exts.split(";"):
            group = group.strip()
            if not group:
                continue
            label, _, patterns = group.partition("|")
            if patterns:
                cmd += [f"--file-filter={label.strip()}|{patterns.strip()}"]
                filters.append(label)
        if filters:
            cmd += ["--file-filter=Tous les fichiers|*"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": True, "paths": [], "cancelled": True}
    except OSError as e:
        return {"ok": False, "reason": str(e), "paths": []}

    if proc.returncode != 0:
        # 1 = annulé par l'utilisateur
        return {"ok": True, "paths": [], "cancelled": True}

    paths = [p for p in (proc.stdout or "").splitlines() if p.strip()]
    return {"ok": True, "paths": paths, "cancelled": False}


# 6. Jobs
@app.get("/api/job/{job_id}")
async def get_job_status(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return job_public_view(job)


@app.post("/api/job/{job_id}/cancel")
async def cancel_job_endpoint(job_id: int):
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return {"ok": True, "status": "cancelling"}


# ── Hub de Conversion ───────────────────────────────────────────────────────

@app.post("/api/convert/scan")
async def convert_scan(req: ScanRequest):
    """Scanne les fichiers et estime la taille de sortie + la durée."""
    if not _convert_svc.available():
        raise HTTPException(
            status_code=503,
            detail=f"Script convert.py introuvable : {_convert_svc.script}. "
            "Vérifiez que le projet ResolveConvert est présent."
        )
    try:
        plan = _convert_svc.scan(
            paths=req.paths,
            target=req.target,
            preset=req.preset,
            keep_4k=req.keep_4k,
        )
        files = plan.get("files") or []
        total = sum(float(f.get("duration") or 0.0) for f in files)
        plan["estimate"] = estimates.estimate(
            "convert", _convert_key(req.target, req.preset),
            total, files=max(1, len(files)))
        plan["total_duration"] = round(total, 3)
        plan["total_duration_human"] = fmt_duration(total)
        return plan
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/convert/start")
async def convert_start(req: ConvertRequest):
    """Démarre la conversion en arrière-plan."""
    busy = _running_job("convert")
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"Une conversion est déjà en cours (job {busy}). "
                   "Arrêtez-la avant d'en lancer une autre."
        )
    jid = create_job("convert", req.model_dump(), estimate=None)
    _claim_slot("convert", jid)
    threading.Thread(target=_run_convert_job, args=(jid,), daemon=True).start()
    return {"id": jid}


# ── Hub de Sous-titrage ────────────────────────────────────────────────────

@app.post("/api/subtitles/scan")
async def subtitles_scan(req: SubtitleScanRequest):
    """Scanne les fichiers et estime le temps de transcription."""
    if not _subtitle_svc.available():
        raise HTTPException(
            status_code=503,
            detail=f"Script transcrire.py introuvable : {_subtitle_svc.script}. "
            "Vérifiez que le projet ResolveSubs est présent."
        )
    try:
        plan = _subtitle_svc.scan(
            paths=req.paths,
            language=req.language,
            model=req.model,
            offset=req.offset,
        )
        files = plan.get("files") or []
        total = sum(float(f.get("duration") or 0.0) for f in files)
        plan["estimate"] = estimates.estimate(
            "subtitles", _subtitle_key(req.model, req.device),
            total, files=max(1, len(files)))
        plan["total_duration"] = round(total, 3)
        plan["total_duration_human"] = fmt_duration(total)
        return plan
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/subtitles/start")
async def subtitles_start(req: SubtitleRequest):
    """Démarre la transcription en arrière-plan."""
    if not _subtitle_svc.available():
        raise HTTPException(
            status_code=503,
            detail=f"Script transcrire.py introuvable : {_subtitle_svc.script}."
        )
    busy = _running_job("subtitles")
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"Un sous-titrage est déjà en cours (job {busy}). "
                   "Arrêtez-le avant d'en lancer un autre."
        )
    jid = create_job("subtitles", req.model_dump())
    _claim_slot("subtitles", jid)
    threading.Thread(target=_run_subtitle_job, args=(jid,), daemon=True).start()
    return {"id": jid}


# ── Découpage Automatique de Plans ─────────────────────────────────────────

@app.post("/api/scenes/scan")
async def scenes_scan(req: SceneScanRequest):
    """Prépare un lot de détection : fichiers + durée estimée du traitement."""
    try:
        payload = _scan_payload(req.paths, VIDEO_EXTS, "scenes",
                                _scene_key(req.downscale_factor))
        if not payload["files"]:
            raise HTTPException(status_code=400,
                                detail="Aucune vidéo valide dans la sélection.")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/scenes/detect")
async def scenes_detect(req: SceneRequest):
    """Démarre la détection de plans (lot) en arrière-plan."""
    paths = req.all_paths()
    if not paths:
        raise HTTPException(status_code=400, detail="Aucun chemin fourni.")
    files = _expand(paths, VIDEO_EXTS)
    if not files:
        raise HTTPException(
            status_code=404,
            detail="Aucune vidéo valide dans la sélection."
        )
    payload = dict(req.model_dump())
    payload["paths"] = paths
    total = sum(float(probe(f).get("duration") or 0.0) for f in files)
    est = estimates.estimate("scenes", _scene_key(req.downscale_factor),
                             total, files=len(files))
    jid = create_job("scenes", payload, estimate=est)
    threading.Thread(target=_run_scene_job, args=(jid,), daemon=True).start()
    return {"id": jid, "estimate": est, "count": len(files)}


@app.post("/api/scenes/export")
async def scenes_export(req: ExportRequest):
    """Exporte les coupes d'un job de détection (texte, ou ZIP si lot)."""
    job = get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(
            status_code=409,
            detail="Le job n'est pas terminé ou n'a pas de résultats."
        )
    result = job["result"]
    entries = result.get("files") or []
    if not entries and result.get("cuts"):
        entries = [{"cuts": result["cuts"], "fps": result.get("fps", DEFAULT_FPS),
                    "path": job["config"].get("video_path", "video")}]

    ext = {"edl": "edl", "xml": "xml", "fcpxml": "fcpxml", "csv": "csv"}.get(req.fmt, "edl")
    try:
        if req.file_index is not None:
            if not (0 <= req.file_index < len(entries)):
                raise HTTPException(status_code=400, detail="Index de fichier invalide.")
            entries = [entries[req.file_index]]

        if len(entries) == 1:
            entry = entries[0]
            video_path = entry.get("path") or "video"
            name = os.path.splitext(os.path.basename(video_path))[0]
            content = export_cuts(video_path, entry.get("cuts", []),
                                  entry.get("fps") or req.fps, fmt=req.fmt)
            return PlainTextResponse(
                content,
                headers={"Content-Disposition": f'inline; filename="{name}_cuts.{ext}"'},
                media_type="text/plain",
            )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in entries:
                video_path = entry.get("path") or "video"
                name = os.path.splitext(os.path.basename(video_path))[0]
                content = export_cuts(video_path, entry.get("cuts", []),
                                      entry.get("fps") or req.fps, fmt=req.fmt)
                zf.writestr(f"{name}_cuts.{ext}", content)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="plans_{stamp}.zip"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'export : {e}")


# ── Analyseur de Beats Audio ───────────────────────────────────────────────

@app.post("/api/audio/scan")
async def audio_scan(req: AudioScanRequest):
    """Prépare un lot d'analyse audio : fichiers + durée estimée."""
    try:
        payload = _scan_payload(req.paths, AUDIO_EXTS | VIDEO_EXTS, "audio",
                                "default")
        if not payload["files"]:
            raise HTTPException(
                status_code=400,
                detail="Aucun fichier audio/vidéo valide dans la sélection."
            )
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/audio/analyze")
async def audio_analyze(req: AudioRequest):
    """Démarre l'analyse de beats audio (lot) en arrière-plan."""
    paths = req.all_paths()
    if not paths:
        raise HTTPException(status_code=400, detail="Aucun chemin fourni.")
    files = _expand(paths, AUDIO_EXTS | VIDEO_EXTS)
    if not files:
        raise HTTPException(
            status_code=404,
            detail="Aucun fichier audio/vidéo valide dans la sélection."
        )
    payload = dict(req.model_dump())
    payload["paths"] = paths
    total = sum(float(probe(f).get("duration") or 0.0) for f in files)
    est = estimates.estimate("audio", "default", total, files=len(files))
    jid = create_job("audio", payload, estimate=est)
    threading.Thread(target=_run_audio_job, args=(jid,), daemon=True).start()
    return {"id": jid, "estimate": est, "count": len(files)}


@app.post("/api/audio/export")
async def audio_export(req: ExportRequest):
    """Exporte les beats d'un job d'analyse (texte, ou ZIP si lot)."""
    job = get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(
            status_code=409,
            detail="Le job n'est pas terminé ou n'a pas de résultats."
        )
    result = job["result"]
    entries = result.get("files") or []
    if not entries and result.get("beats"):
        entries = [{"beats": result["beats"],
                    "path": job["config"].get("audio_path", "audio")}]

    ext_map = {"csv": "csv", "edl": "edl", "xml": "xml", "fcpxml": "fcpxml", "davinci_script": "py"}
    ext = ext_map.get(req.fmt, "csv")
    try:
        if req.file_index is not None:
            if not (0 <= req.file_index < len(entries)):
                raise HTTPException(status_code=400, detail="Index de fichier invalide.")
            entries = [entries[req.file_index]]

        def _content(entry):
            path = entry.get("path") or "audio"
            name = os.path.splitext(os.path.basename(path))[0]
            text = export_markers(entry.get("beats", []), fps=req.fps,
                                  fmt=req.fmt, title=f"{name}_beats",
                                  asset_path=path, ranks=req.ranks)
            return name, text

        if len(entries) == 1:
            name, content = _content(entries[0])
            return PlainTextResponse(
                content,
                headers={"Content-Disposition":
                         f'inline; filename="{name}_beats.{ext}"'},
                media_type="text/plain",
            )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in entries:
                name, content = _content(entry)
                zf.writestr(f"{name}_beats.{ext}", content)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="beats_{stamp}.zip"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'export : {e}")


# ── Montage au rythme (auto-cut beats) ─────────────────────────────────────

class RhythmScanRequest(BaseModel):
    """Sources du montage : une vidéo + une musique."""
    video_path: str
    audio_path: str


class RhythmRequest(RhythmScanRequest):
    """Paramètres complets du montage au rythme."""
    ranks: list[int] = Field(default_factory=lambda: [1])
    min_shot: float = 0.8
    max_shot: float = 3.0
    loop: bool = True
    fps: float = 24.0
    sample_rate: int = 22050
    video_start: float = 0.0


class RhythmExportRequest(BaseModel):
    """Export d'un montage déjà calculé."""
    job_id: int
    fmt: str = "fcpxml"          # fcpxml | xml | edl | csv
    include_audio: bool = True   # musique sur une piste liée (FCPXML)


def _rhythm_sources(video_path: str, audio_path: str) -> tuple[dict, dict]:
    """Valide les deux sources et retourne leurs infos ffprobe."""
    video_path = os.path.abspath(str(video_path or ""))
    audio_path = os.path.abspath(str(audio_path or ""))
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404,
                            detail=f"Vidéo introuvable : {video_path}")
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=404,
                            detail=f"Musique introuvable : {audio_path}")

    vinfo = probe(video_path)
    ainfo = probe(audio_path)
    if not vinfo.get("has_video"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier vidéo ne contient aucune piste image.",
        )
    if not ainfo.get("has_audio"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier musique ne contient aucune piste audio.",
        )
    if float(vinfo.get("duration") or 0.0) <= 0:
        raise HTTPException(status_code=400,
                            detail="Durée de la vidéo inconnue (ffprobe).")
    if float(ainfo.get("duration") or 0.0) <= 0:
        raise HTTPException(status_code=400,
                            detail="Durée de la musique inconnue (ffprobe).")
    return vinfo, ainfo


@app.post("/api/rhythm/scan")
async def rhythm_scan(req: RhythmScanRequest):
    """Vérifie les sources et renvoie durées, format et estimation."""
    vinfo, ainfo = _rhythm_sources(req.video_path, req.audio_path)
    audio_duration = float(ainfo.get("duration") or 0.0)
    est = estimates.estimate("audio", "default", audio_duration, files=1)
    return {
        "video": {
            "path": vinfo.get("path"),
            "name": os.path.basename(str(vinfo.get("path") or "")),
            "duration": round(float(vinfo.get("duration") or 0.0), 3),
            "width": int(vinfo.get("width") or 0),
            "height": int(vinfo.get("height") or 0),
            "fps": round(float(vinfo.get("fps") or 24.0), 3),
        },
        "audio": {
            "path": ainfo.get("path"),
            "name": os.path.basename(str(ainfo.get("path") or "")),
            "duration": round(audio_duration, 3),
        },
        "estimate": est,
    }


@app.post("/api/rhythm/start")
async def rhythm_start(req: RhythmRequest):
    """Lance l'analyse des beats + le calcul du montage (job)."""
    vinfo, ainfo = _rhythm_sources(req.video_path, req.audio_path)

    ranks = sorted({int(r) for r in (req.ranks or []) if int(r) in (1, 2, 3)})
    if not ranks:
        ranks = [1]
    min_shot = min(max(0.1, float(req.min_shot or 0.8)), 30.0)
    max_shot = min(max(min_shot, float(req.max_shot or 3.0)), 60.0)
    fps = min(max(1.0, float(req.fps or 24.0)), 120.0)

    payload = {
        "video_path": os.path.abspath(req.video_path),
        "audio_path": os.path.abspath(req.audio_path),
        "ranks": ranks,
        "min_shot": round(min_shot, 3),
        "max_shot": round(max_shot, 3),
        "loop": bool(req.loop),
        "fps": round(fps, 3),
        "sample_rate": int(min(max(8000, req.sample_rate or 22050), 48000)),
        "video_start": round(max(0.0, float(req.video_start or 0.0)), 3),
    }
    audio_duration = float(ainfo.get("duration") or 0.0)
    est = estimates.estimate("audio", "default", audio_duration, files=1)
    jid = create_job("rhythm", payload, estimate=est)
    threading.Thread(target=_run_rhythm_job, args=(jid,), daemon=True).start()
    return {
        "id": jid,
        "estimate": est,
        "video_duration": round(float(vinfo.get("duration") or 0.0), 3),
        "audio_duration": round(audio_duration, 3),
    }


@app.post("/api/rhythm/export")
async def rhythm_export(req: RhythmExportRequest):
    """Exporte le montage calculé (FCPXML, EDL ou CSV)."""
    job = get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("type") != "rhythm":
        raise HTTPException(status_code=400,
                            detail="Ce job n'est pas un montage au rythme.")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(
            status_code=409,
            detail="Le montage n'est pas terminé ou n'a pas de résultat.",
        )

    result = job["result"]
    cuts = result.get("cuts") or []
    if not cuts:
        raise HTTPException(status_code=409, detail="Aucun plan à exporter.")

    fmt = (req.fmt or "fcpxml").lower()
    ext = {"fcpxml": "fcpxml", "xml": "fcpxml", "edl": "edl",
           "csv": "csv"}.get(fmt)
    if not ext:
        raise HTTPException(
            status_code=400,
            detail="Format invalide : utilisez fcpxml, edl ou csv.",
        )

    video_path = result.get("video_path") or ""
    stem = os.path.splitext(os.path.basename(video_path))[0] or "montage"
    try:
        content = export_rhythm(
            video_path,
            cuts,
            fps=float(result.get("fps") or 24.0),
            fmt=fmt,
            audio_path=result.get("audio_path"),
            title=f"{stem}_rythme",
            audio_duration=result.get("audio_duration"),
            include_audio=bool(req.include_audio),
        )
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erreur d'export : {e}")

    return PlainTextResponse(
        content,
        headers={"Content-Disposition":
                 f'inline; filename="{stem}_rythme.{ext}"'},
        media_type="text/plain",
    )


# ── DaVinci Resolve : installation du script de marqueurs ──────────────────

class DavinciInstallRequest(BaseModel):
    """Requête d'installation du script dans le menu Scripts de Resolve."""
    job_id: int
    fps: float = 24.0
    file_index: Optional[int] = None
    ranks: Optional[list[int]] = None


def _audio_entries(job) -> list[dict]:
    """Liste normalisée des pistes analysées d'un job audio."""
    result = job.get("result") or {}
    entries = result.get("files") or []
    if not entries and result.get("beats"):
        entries = [{"beats": result["beats"],
                    "path": job["config"].get("audio_path", "audio")}]
    return entries


@app.post("/api/audio/davinci/install")
async def davinci_install(req: DavinciInstallRequest):
    """Installe (ou met à jour) le script dans Workspace → Scripts.

    Le script embarque les marqueurs du job courant : il suffit ensuite de
    lancer *Workspace → Scripts → EdittingHub Beat Markers* dans Resolve.
    """
    from app import davinci
    from app.services.audio_beats import export_markers

    job = get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(
            status_code=409,
            detail="Le job n'est pas terminé ou n'a pas de résultats."
        )

    entries = _audio_entries(job)
    if not entries:
        raise HTTPException(status_code=409, detail="Aucun beat à exporter.")

    index = req.file_index if req.file_index is not None else 0
    if not (0 <= index < len(entries)):
        raise HTTPException(status_code=400, detail="Index de fichier invalide.")
    entry = entries[index]

    path = entry.get("path") or "audio"
    name = os.path.splitext(os.path.basename(path))[0]
    beats = entry.get("beats", [])
    if req.ranks is not None:
        beats = [b for b in beats if int(b.get("rank") or 0) in req.ranks]

    content = export_markers(beats, fps=req.fps, fmt="davinci_script",
                             title=f"{name}_beats", asset_path=path,
                             ranks=req.ranks)
    try:
        info = davinci.install_script(content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Écriture impossible dans le dossier Scripts : {e}"
        )

    info.update({
        "markers": len(beats),
        "source": os.path.basename(path),
        "ranks": req.ranks,
        "resolve_installed": davinci.resolve_installed(),
    })
    return info
