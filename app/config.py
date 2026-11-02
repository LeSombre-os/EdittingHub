"""Configuration centrale du hub EdittingHub.

Toutes les constantes, chemins vers les outils externes et paramètres
d'export sont définis ici pour être centralisés et facilement modifiables.
"""
import os
from pathlib import Path

# ── Chemin de base du workspace ───────────────────────────────────────────
# Par défaut : le dossier parent de EdittingHub/ (les projets voisins
# ResolveConvert/ et ResolveSubs/ y sont cherchés). Surchargeable via la
# variable d'environnement EDITTINGHUB_WORKSPACE.
WORKSPACE_ROOT = Path(
    os.environ.get("EDITTINGHUB_WORKSPACE")
    or Path(__file__).resolve().parent.parent.parent
)

# ── Outils externes (scripts existants du workspace) ──────────────────────
# ResolveConvert : convert.py (stdlib uniquement, ffmpeg/ffprobe requis)
RESOLVECONVERT_DIR = WORKSPACE_ROOT / "ResolveConvert"
CONVERT_SCRIPT = RESOLVECONVERT_DIR / "convert.py"

# ResolveSubs : transcrire.py (nécessite faster-whisper dans un venv)
RESOLVESUBS_DIR = WORKSPACE_ROOT / "ResolveSubs"
SUBTITLES_SCRIPT = RESOLVESUBS_DIR / "transcrire.py"
# Venv dédié au sous-titrage (a faster-whisper installé)
SUBTITLES_PYTHON = (
    Path(os.environ.get("HOME", "")) / ".venvs" / "resolvesubs" / "bin" / "python"
)
if not SUBTITLES_PYTHON.exists():
    SUBTITLES_PYTHON = "/usr/bin/python3"  # repli : système (sans whisper)

# Python système pour convert.py (stdlib uniquement)
CONVERT_PYTHON = "/usr/bin/python3"

# ── Paramètres serveur ────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8005
BASE_URL = f"http://{HOST}:{PORT}"

# ── Dossiers de sortie par défaut (relatifs au projet) ────────────────────
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"
SCENES_DIR = OUTPUT_ROOT / "scenes"
BEATS_DIR = OUTPUT_ROOT / "beats"
EDL_DIR = OUTPUT_ROOT / "edl"

# ── Paramètres par défaut pour la détection de plans ──────────────────────
DEFAULT_SCENE_THRESHOLD = 30.0       # seuil de détection de changement de plan (0-100)
DEFAULT_MIN_SCENE_LEN = 15           # durée minimale d'une scène (frames)
DEFAULT_FPS = 24                     # fps de référence pour l'EDL

# ── Paramètres par défaut pour l'analyse audio ────────────────────────────
DEFAULT_AUDIO_SR = 22050             # sample rate pour librosa
DEFAULT_RANK1_PERCENTILE = 66.0      # percentile pour le rang 1 (beats forts)
DEFAULT_RANK2_PERCENTILE = 33.0      # percentile pour le rang 2 (beats moyens)
MIN_BEAT_INTERVAL = 0.15             # intervalle minimum entre deux beats (s)

# Couleurs Resolve pour les marqueurs EDL
RANK_COLORS = {
    1: "Red",       # impacts forts
    2: "Yellow",    # secondaires
    3: "Blue",      # contre-temps / micro-variations
}

# ── Vérification des outils externes ──────────────────────────────────────
def check_external_tools():
    """Vérifie que les outils externes requis sont disponibles.

    Retourne un dict d'état pour le endpoint /api/config.
    """
    import shutil
    tools = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "convert_script": CONVERT_SCRIPT.exists(),
        "subtitles_script": SUBTITLES_SCRIPT.exists(),
        "subtitles_python": str(SUBTITLES_PYTHON),
        "convert_python": CONVERT_PYTHON,
    }
    # Vérifie les paquets locaux
    try:
        import scenedetect
        tools["scenedetect"] = True
        tools["scenedetect_version"] = scenedetect.__version__
    except ImportError:
        tools["scenedetect"] = False

    try:
        import librosa
        tools["librosa"] = True
        tools["librosa_version"] = librosa.__version__
    except ImportError:
        tools["librosa"] = False

    # Crée les dossiers de sortie
    for d in (SCENES_DIR, BEATS_DIR, EDL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    return tools
