"""Estimations de durée de traitement, auto-calibrées.

Chaque module a un ratio « secondes de traitement par seconde de média ».
Les valeurs par défaut viennent de mesures réelles sur cette machine
(Ryzen 5 5600G + RX 6600) ; après chaque job, le ratio observé est enregistré
et lissé (moyenne pondérée). Les estimations s'affinent donc à l'usage.

Fichier de calibration : outputs/estimates.json (créé automatiquement).
"""
from __future__ import annotations

import json
import threading
import time

from app.config import OUTPUT_ROOT

STORE = OUTPUT_ROOT / "estimates.json"
_LOCK = threading.Lock()

# Ratios par défaut : (secondes de traitement) / (seconde de média)
DEFAULTS: dict[str, dict[str, float]] = {
    # Conversion : key = "target:preset"
    "convert": {
        "tiktok:lb": 0.13,
        "tiktok:sq": 0.25,
        "tiktok:hq": 0.50,
        "resolve:lb": 1.20,
        "resolve:sq": 2.00,
        "resolve:hq": 3.50,
    },
    # Sous-titrage : key = "modele:peripherique"
    "subtitles": {
        "base:cpu": 0.09,
        "small:cpu": 0.15,
        "medium:cpu": 0.40,
        "small:gpu": 0.04,
        "turbo:gpu": 0.06,
        "large-v3:gpu": 0.30,
        "default": 0.15,
    },
    # Détection de plans : key = "ds<facteur>"
    "scenes": {
        "ds1": 0.28,
        "ds2": 0.07,
        "ds3": 0.031,
        "ds4": 0.018,
        "default": 0.07,
    },
    # Analyse audio : key = "default"
    "audio": {
        "default": 0.045,
    },
    # Coupe des blancs : silences + ASR + rendu (1 passe x264)
    "autotrim": {
        "gpu": 0.35,
        "cpu": 0.60,
        "default": 0.60,
    },
    # Vérification de script : ASR + alignement difflib
    "scriptcheck": {
        "gpu": 0.06,
        "cpu": 0.20,
        "default": 0.20,
    },
}

# Coût fixe par fichier (ouverture, extraction, écriture)
OVERHEAD = {
    "convert": 1.0,
    "subtitles": 2.0,
    "scenes": 0.6,
    "audio": 1.2,
    "autotrim": 4.0,
    "scriptcheck": 3.0,
}


def _load() -> dict:
    try:
        with open(STORE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp.replace(STORE)
    except OSError:
        pass


def default_ratio(module: str, key: str | None = None) -> float:
    table = DEFAULTS.get(module, {})
    if key and key in table:
        return float(table[key])
    if "default" in table:
        return float(table["default"])
    return 0.1


def ratio(module: str, key: str | None = None) -> float:
    """Ratio calibré s'il existe, sinon valeur par défaut."""
    with _LOCK:
        data = _load()
    entry = (data.get(module) or {}).get(key or "default")
    if isinstance(entry, dict) and entry.get("ratio"):
        try:
            return float(entry["ratio"])
        except (TypeError, ValueError):
            pass
    if isinstance(entry, (int, float)):
        return float(entry)
    return default_ratio(module, key)


def record(module: str, key: str | None, media_seconds: float, elapsed: float) -> None:
    """Enregistre un ratio observé (moyenne pondérée, poids 0.5)."""
    media_seconds = float(media_seconds or 0.0)
    elapsed = float(elapsed or 0.0)
    if media_seconds <= 1.0 or elapsed <= 0.0:
        return
    observed = elapsed / media_seconds
    # Ignore les mesures aberrantes (plantage, attente utilisateur…)
    if not (0.001 <= observed <= 50.0):
        return

    key = key or "default"
    with _LOCK:
        data = _load()
        module_data = data.setdefault(module, {})
        entry = module_data.get(key)
        if isinstance(entry, dict) and entry.get("ratio"):
            old = float(entry["ratio"])
            count = int(entry.get("n") or 1)
            new = round(old * 0.5 + observed * 0.5, 5)
            module_data[key] = {"ratio": new, "n": count + 1}
        else:
            module_data[key] = {"ratio": round(observed, 5), "n": 1}
        _save(data)


def estimate(module: str, key: str | None, media_seconds: float,
             files: int = 1) -> dict:
    """Estime la durée d'un job.

    Retourne {seconds, human, ratio, media_seconds, files}.
    """
    media_seconds = float(media_seconds or 0.0)
    files = max(1, int(files or 1))
    r = ratio(module, key)
    seconds = media_seconds * r + OVERHEAD.get(module, 1.0) * files
    return {
        "seconds": round(seconds, 1),
        "human": human(seconds),
        "ratio": round(r, 4),
        "media_seconds": round(media_seconds, 2),
        "files": files,
    }


def human(seconds: float) -> str:
    """Durée lisible : '45 s', '3 min 20 s', '1 h 12 min'."""
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{max(1, int(round(seconds)))} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d} s" if sec else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if minutes else f"{hours} h"


class EtaTracker:
    """Suit l'avancement réel pour afficher un temps restant fiable."""

    def __init__(self, total_units: float = 0.0):
        self.started = time.time()
        self.total = float(total_units or 0.0)

    def update(self, done_units: float) -> dict:
        elapsed = time.time() - self.started
        done_units = float(done_units or 0.0)
        if done_units <= 0 or elapsed <= 0:
            return {"elapsed": round(elapsed, 1), "eta": None}
        speed = done_units / elapsed          # unités par seconde
        remaining = max(0.0, self.total - done_units)
        eta = remaining / speed if speed > 0 else None
        return {
            "elapsed": round(elapsed, 1),
            "eta": round(eta, 1) if eta is not None else None,
            "eta_human": human(eta) if eta is not None else None,
        }
