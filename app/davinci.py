"""Intégration DaVinci Resolve : script de marqueurs + installation.

Pourquoi ce module
------------------
L'import FCPXML/EDL ne restitue pas les marqueurs sur DaVinci Resolve **Free
sous Linux** (la timeline est importée, les `<marker>` sont ignorés). Le seul
canal fiable est l'API Python *interne* de Resolve (celle que fournit l'objet
`resolve` de la console et des scripts du menu Workspace → Scripts).

Le script externe (`python3 script.py` + `RESOLVE_SCRIPT_LIB`) renvoie
`resolve = None` sur cette installation : le mode « External scripting » n'est
pas exposé. On ne l'utilise donc plus comme voie principale.

Deux façons d'exécuter le script généré :

1. **Menu Scripts (recommandé)** : le script est copié dans
   ``~/.local/share/DaVinciResolve/Fusion/Scripts/Utility/`` puis lancé par
   *Workspace → Scripts → EdittingHub Beat Markers*.
2. **Console** : *Workspace → Console → onglet Python*, coller le script.

Pour que les deux fonctionnent, **chaque instruction du script tient sur une
seule ligne** (pas de bloc indenté sur plusieurs lignes) : un collage ligne à
ligne dans la console reste alors valide.

Le script n'importe jamais `DaVinciResolveScript` : charger `fusionscript.so`
une seconde fois dans le processus de Resolve provoque un SIGSEGV. Il réutilise
l'objet `resolve` déjà présent (globals → `bmd` → module `fusionscript` déjà
chargé).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

# Nom fixe : l'entrée du menu Resolve reste la même d'une installation à
# l'autre (Resolve ne relit pas le dossier Scripts, mais relit le fichier
# au moment du clic → pas besoin de redémarrer après la 1re installation).
SCRIPT_NAME = "EdittingHub_Beat_Markers.py"
SCRIPT_STEM = "EdittingHub Beat Markers"

# Couleurs de marqueurs Resolve par rang de beat
RANK_COLORS = {1: "Red", 2: "Yellow", 3: "Blue", 4: "Green"}

# Caractères interdits dans l'encodage compact des données
_SEP_FIELD = "|"
_SEP_ITEM = ";"


# ── Chemins DaVinci ────────────────────────────────────────────────────────

def resolve_scripts_dir() -> Path:
    """Dossier des scripts utilisateur de DaVinci Resolve (Linux)."""
    return (Path.home() / ".local" / "share" / "DaVinciResolve"
            / "Fusion" / "Scripts" / "Utility")


def resolve_installed() -> bool:
    """Vrai si une installation DaVinci Resolve est détectée."""
    return (Path("/opt/resolve/bin/resolve").exists()
            or (Path.home() / ".local" / "share" / "DaVinciResolve").is_dir())


def script_path() -> Path:
    """Chemin absolu du script installé dans le menu Resolve."""
    return resolve_scripts_dir() / SCRIPT_NAME


# ── État du processus Resolve ──────────────────────────────────────────────

def _proc_start_epoch(pid: int) -> float | None:
    """Heure de démarrage d'un processus (epoch), via /proc/<pid>/stat."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        # Le champ « comm » (2) peut contenir des espaces et des parenthèses :
        # on repart de la dernière parenthèse fermante.
        fields = raw[raw.rindex(")") + 2:].split()
        start_ticks = int(fields[19])          # 22e champ de /proc/<pid>/stat
        btime = 0
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def resolve_process_start() -> float | None:
    """Date de lancement du processus DaVinci Resolve, ou None s'il est fermé."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            exe = os.readlink(f"/proc/{entry}/exe")
        except OSError:
            continue
        if exe.endswith("/resolve") and "/opt/resolve" in exe:
            return _proc_start_epoch(int(entry))
    return None


# ── Génération du script ───────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Neutralise tout ce qui pourrait casser l'encodage ou le littéral Python."""
    text = str(text or "").replace(_SEP_FIELD, "/").replace(_SEP_ITEM, ",")
    text = re.sub(r'["\'\\]', " ", text)          # guillemets / antislash
    return re.sub(r"\s+", " ", text).strip()


def _pack(beats) -> str:
    """Encode les marqueurs en une seule ligne ``t|rang|nom``.

    Un seul littéral d'une seule ligne : indispensable pour un collage dans la
    console Resolve (pas de liste Python multi-lignes).
    """
    items = []
    for index, beat in enumerate(beats, 1):
        try:
            seconds = float(beat.get("time") or 0.0)
        except (TypeError, ValueError):
            continue
        if seconds < 0:
            continue
        rank = int(beat.get("rank") or 1)
        label = _sanitize(beat.get("label") or "Beat")
        name = _sanitize(f"Beat {index} - {label}")
        items.append(f"{seconds:.6f}{_SEP_FIELD}{rank}{_SEP_FIELD}{name}")
    return _SEP_ITEM.join(items)


def build_script(beats, fps: float = 24.0, title: str | None = None,
                 asset_path: str | None = None,
                 ranks: list[int] | None = None) -> str:
    """Construit le script Python autonome (marqueurs embarqués).

    Args:
        beats : liste de dicts ``{time, rank, label, …}`` déjà filtrés.
        fps : fps de repli si la timeline n'en fournit pas.
        title : nom affiché dans l'en-tête (projet / piste).
        asset_path : média source (informatif).
        ranks : rangs exportés (informatif).

    Returns:
        Le contenu texte du script ``.py``.
    """
    beats = list(beats or [])
    # Filtrage défensif : l'appelant filtre déjà, mais on ne dépend pas de lui.
    if ranks is not None:
        allowed = {int(r) for r in ranks}
        beats = [b for b in beats if int(b.get("rank") or 0) in allowed]
    title = _sanitize(title or "Audio Beat Markers")
    source = _sanitize(os.path.basename(asset_path)) if asset_path else "(inconnu)"
    ranks_txt = ",".join(str(r) for r in (ranks or sorted(
        {int(b.get("rank") or 1) for b in beats}))) or "tous"
    data = _pack(beats)
    stamp = time.strftime("%Y-%m-%d %H:%M")

    lines = [
        "# -*- coding: utf-8 -*-",
        "#",
        "# EdittingHub — marqueurs de beats pour DaVinci Resolve",
        f"# Source    : {source}",
        f"# Projet    : {title}",
        f"# Rangs     : {ranks_txt}",
        f"# Marqueurs : {len(beats)}",
        f"# Généré le : {stamp}",
        "#",
        "# UTILISATION",
        "#   1. Menu Resolve : Workspace → Scripts → « EdittingHub Beat Markers »",
        "#      (le fichier doit être dans",
        "#       ~/.local/share/DaVinciResolve/Fusion/Scripts/Utility/)",
        "#   2. Ou console : Workspace → Console → onglet Python → coller ce script.",
        "#",
        "# Prérequis : un projet ouvert ET une timeline active.",
        "# Chaque instruction tient sur une seule ligne (collage console sûr).",
        "",
        "FPS_FALLBACK = %s" % float(fps or 24.0),
        "CLEAR_FIRST = False  # True = supprime d'abord les marqueurs Rouge/Jaune/Bleu",
        "COLORS = {1: 'Red', 2: 'Yellow', 3: 'Blue', 4: 'Green'}",
        "",
        "# t|rang|nom ; t|rang|nom ; …",
        'DATA = "%s"' % data,
        "",
        "import sys",
        "resolve = globals().get('resolve')",
        "if resolve is None and 'bmd' in globals(): resolve = globals()['bmd'].scriptapp('Resolve')",
        "if resolve is None and 'fusionscript' in sys.modules: resolve = sys.modules['fusionscript'].scriptapp('Resolve')",
        "if resolve is None: print('ERREUR: objet resolve introuvable.'); print('Lancez ce script via Workspace > Scripts, ou collez-le dans Workspace > Console > onglet Python.'); print('(Un terminal externe renvoie resolve = None : ce n est pas la bonne methode.)'); raise SystemExit(1)",
        "",
        "pm = resolve.GetProjectManager()",
        "proj = pm.GetCurrentProject() if pm else None",
        "if not proj: print('ERREUR: aucun projet ouvert dans DaVinci Resolve.'); raise SystemExit(1)",
        "tl = proj.GetCurrentTimeline()",
        "if not tl: print('ERREUR: aucune timeline active. Ouvrez une timeline puis relancez.'); raise SystemExit(1)",
        "",
        "_raw_fps = tl.GetSetting('timelineFrameRate')",
        "FPS = float(_raw_fps) if _raw_fps else FPS_FALLBACK",
        "_safe = lambda v: str(v).encode('ascii', 'replace').decode('ascii')",
        "",
        "if CLEAR_FIRST: tl.DeleteMarkersByColor('Red'); tl.DeleteMarkersByColor('Yellow'); tl.DeleteMarkersByColor('Blue')",
        "",
        "MARKERS = [p.split('%s') for p in DATA.split('%s') if p]" % (_SEP_FIELD, _SEP_ITEM),
        "added = 0",
        "for _p in MARKERS: _f = int(float(_p[0]) * FPS + 0.5); _ok = tl.AddMarker(_f, COLORS.get(int(_p[1]), 'Red'), _p[2], _p[2], 1); added += (1 if _ok else 0)",
        "",
        "print('Timeline  : ' + _safe(tl.GetName()))",
        "print('FPS       : ' + _safe(FPS))",
        "print('Marqueurs : ' + _safe(added) + ' / ' + _safe(len(MARKERS)))",
        "if added < len(MARKERS): print('ATTENTION: certains marqueurs ont ete refuses (position hors timeline ?)')",
        "print('OK - marqueurs ajoutes sur la timeline active.')",
        "",
    ]
    return "\n".join(lines)


# ── Installation dans le menu Scripts de Resolve ───────────────────────────

def install_script(content: str) -> dict:
    """Écrit le script dans le dossier Scripts/Utility de Resolve.

    Returns:
        dict ``{ok, path, created_dir, restart_needed}``.

    Raises:
        FileNotFoundError : DaVinci Resolve n'est pas installé.
    """
    base = Path.home() / ".local" / "share" / "DaVinciResolve"
    if not base.is_dir() and not resolve_installed():
        raise FileNotFoundError(
            "DaVinci Resolve introuvable : dossier ~/.local/share/DaVinciResolve absent."
        )

    directory = resolve_scripts_dir()
    created = not directory.is_dir()
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / SCRIPT_NAME
    existed = target.is_file()
    # mtime AVANT réécriture : détermine si Resolve avait déjà vu le fichier
    # au moment de son démarrage.
    previous_mtime = target.stat().st_mtime if existed else None
    target.write_text(content, encoding="utf-8")
    try:
        target.chmod(0o755)
    except OSError:
        pass

    # Resolve n'énumère le dossier Scripts qu'au démarrage. Le script n'est
    # donc absent du menu que s'il a été écrit APRÈS le lancement du Resolve
    # en cours. Une fois l'entrée créée, le fichier est relu à chaque clic :
    # réinstaller ne demande plus aucun redémarrage.
    started = resolve_process_start()
    running = started is not None
    # Seule la CRÉATION du fichier pendant que Resolve tourne impose un
    # redémarrage : l'entrée du menu n'existe pas encore. Une réécriture
    # ultérieure ne change rien (Resolve relit le fichier à chaque clic).
    needs_restart = running and previous_mtime is None

    return {
        "ok": True,
        "path": str(target),
        "directory": str(directory),
        "created_dir": created,
        "restart_needed": needs_restart,
        "resolve_running": running,
        "menu_label": SCRIPT_STEM,
    }
