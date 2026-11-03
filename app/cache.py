"""Cache disque des analyses coûteuses (beats, plans).

Pourquoi : analyser une piste de 4 min prend ~10 s, découper une vidéo 4K
plusieurs minutes. Ces résultats ne changent pas tant que le fichier et les
paramètres sont identiques. On les stocke donc dans `outputs/cache/` :

    outputs/cache/audio/<clé>.json
    outputs/cache/scenes/<clé>.json

Clé = empreinte (chemin, taille, mtime, paramètres). Modifier le fichier ou
un paramètre invalide l'entrée automatiquement — aucun risque de résultat
périmé. Le cache est purement local et supprimable à tout moment.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from app.config import OUTPUT_ROOT

CACHE_ROOT = OUTPUT_ROOT / "cache"
_LOCK = threading.Lock()

# Version du format : à incrémenter si l'algorithme change et que les anciens
# résultats ne sont plus comparables.
FORMAT_VERSION = 2


def _kind_dir(kind: str):
    return CACHE_ROOT / kind


def file_fingerprint(path: str) -> dict:
    """Empreinte stable d'un fichier : chemin absolu, taille, mtime arrondi."""
    try:
        st = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": st.st_size,
            "mtime": round(st.st_mtime, 3),
        }
    except OSError:
        return {"path": os.path.abspath(path), "size": 0, "mtime": 0}


def make_key(kind: str, path: str, params: dict | None = None) -> str:
    """Calcule la clé de cache d'un fichier + paramètres."""
    payload = {
        "v": FORMAT_VERSION,
        "kind": kind,
        "file": file_fingerprint(path),
        "params": params or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def get(kind: str, key: str, max_age_days: float = 30) -> dict | None:
    """Lit une entrée de cache (None si absente, illisible ou trop vieille)."""
    path = _kind_dir(kind) / f"{key}.json"
    try:
        if max_age_days and time.time() - path.stat().st_mtime > max_age_days * 86400:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def put(kind: str, key: str, payload: dict) -> bool:
    """Écrit une entrée de cache de façon atomique."""
    directory = _kind_dir(kind)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    target = directory / f"{key}.json"
    tmp = directory / f".tmp-{key}-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, target)
        return True
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def stats() -> dict:
    """Nombre d'entrées et taille par type (pour l'UI)."""
    out: dict[str, dict] = {}
    total_bytes = 0
    total_count = 0
    with _LOCK:
        if not CACHE_ROOT.is_dir():
            return {"kinds": {}, "count": 0, "bytes": 0, "human": "0 o"}
        for directory in sorted(CACHE_ROOT.iterdir()):
            if not directory.is_dir():
                continue
            count = 0
            size = 0
            for entry in directory.glob("*.json"):
                try:
                    size += entry.stat().st_size
                    count += 1
                except OSError:
                    continue
            out[directory.name] = {"count": count, "bytes": size}
            total_bytes += size
            total_count += count
    return {
        "kinds": out,
        "count": total_count,
        "bytes": total_bytes,
        "human": _human(total_bytes),
    }


def clear(kind: str | None = None) -> int:
    """Supprime le cache (d'un type ou de tout). Retourne le nombre de fichiers."""
    removed = 0
    targets = [_kind_dir(kind)] if kind else (
        [d for d in CACHE_ROOT.iterdir() if d.is_dir()] if CACHE_ROOT.is_dir() else []
    )
    for directory in targets:
        if not directory.is_dir():
            continue
        for entry in directory.glob("*.json"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _human(num: int) -> str:
    for unit in ("o", "Kio", "Mio", "Gio"):
        if num < 1024 or unit == "Gio":
            return f"{num:.0f} {unit}" if unit == "o" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} Gio"
