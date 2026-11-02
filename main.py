#!/usr/bin/env python3
"""EdittingHub — Point d'entrée unique.

Lance le serveur FastAPI en local et ouvre automatiquement l'interface
dans le navigateur par défaut.

Usage :
    python3 main.py          → lance le serveur + ouvre le navigateur
    ./run.sh                 → même chose via bash (double-clic / menu)
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Utilise le venv du projet si disponible, sinon python3 système
_VENV_PY = Path("/home/lesombre/.venvs/edittinghub/bin/python")
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

HOST = "127.0.0.1"
PORT = 8005
URL = f"http://{HOST}:{PORT}"


def _wait_for_server(timeout=15):
    """Attend que le serveur réponde au ping."""
    import urllib.request
    for _ in range(timeout * 10):
        try:
            urllib.request.urlopen(f"{URL}/api/ping", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def main():
    # Vérifie que le paquet uvicorn est installé
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("uvicorn manquant. Installez les dépendances :")
        print(f"  {_VENV_PY} -m pip install -r requirements.txt")
        sys.exit(1)

    os.chdir(Path(__file__).resolve().parent)

    # Lance le serveur uvicorn
    print(f"🚀 EdittingHub — démarrage sur {URL}")
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "app.server:app",
         "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Attend que le serveur soit prêt
    if _wait_for_server():
        print(f"✅ Serveur prêt — ouverture du navigateur…")
        webbrowser.open(URL)
        print(f"📍 {URL}")
    else:
        print("⚠️  Le serveur n'a pas répondu au ping — vérifiez les logs.")

    # Attend que le serveur se termine (Ctrl+C)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur…")
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
