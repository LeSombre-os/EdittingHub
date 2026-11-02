#!/usr/bin/env bash
# EdittingHub — lanceur unique.
#
#   ./run.sh              → ouvre l'interface web (http://127.0.0.1:8005)
#   python3 main.py       → même chose en Python pur
#
# Le serveur FastAPI tourne en local, 100 % offline.
cd "$(dirname "$0")"
PORT=8005
URL="http://127.0.0.1:$PORT"
PY="/home/lesombre/.venvs/edittinghub/bin/python"
if [ ! -x "$PY" ]; then
    PY="python3"
fi

# Démarrage du serveur si ce n'est pas déjà fait
if ! curl -s --max-time 2 "$URL/api/ping" > /dev/null 2>&1; then
    setsid nohup "$PY" -m uvicorn app.server:app \
        --host 127.0.0.1 --port "$PORT" \
        --log-level warning > /tmp/edittinghub.log 2>&1 < /dev/null &
    disown 2>/dev/null

    # Attend que le serveur réponde réellement (max ~20 s) avant d'ouvrir le navigateur
    for _ in $(seq 1 40); do
        curl -s --max-time 1 "$URL/api/ping" > /dev/null 2>&1 && break
        sleep 0.5
    done

    if ! curl -s --max-time 1 "$URL/api/ping" > /dev/null 2>&1; then
        echo "Le serveur n'a pas démarré. Voir /tmp/edittinghub.log" >&2
        echo "Test manuel : cd \"$(pwd)\" && \"$PY\" -m uvicorn app.server:app --port $PORT" >&2
        exit 1
    fi
fi

# Ouvre le navigateur
if command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$URL" > /dev/null 2>&1 &
    disown 2>/dev/null
else
    echo "Ouvrez $URL dans votre navigateur."
fi
