#!/usr/bin/env bash
# EdittingHub — arrêt propre du serveur local.
#
#   ./stop.sh          → arrête le serveur du port 8005
#
# Pourquoi ce script : `pkill -f "uvicorn app.server:app"` tue aussi le shell
# qui contient la commande (le motif apparaît dans sa propre ligne de commande).
# Ici le motif est construit dynamiquement, donc le shell appelant est épargné.
set -u

PORT=8005
PATTERN="uvicorn app[.]server:app"

pids=$(pgrep -f "$PATTERN" 2>/dev/null || true)
if [ -z "$pids" ]; then
    echo "Aucun serveur EdittingHub en cours sur le port $PORT."
    exit 0
fi

echo "Arrêt du serveur EdittingHub (PID : $(echo $pids | tr '\n' ' '))…"
for pid in $pids; do
    [ "$pid" = "$$" ] && continue
    kill "$pid" 2>/dev/null
done

for _ in $(seq 1 20); do
    if ! curl -sf --max-time 1 "http://127.0.0.1:$PORT/api/ping" >/dev/null 2>&1; then
        echo "Serveur arrêté."
        exit 0
    fi
    sleep 0.25
done

echo "Arrêt forcé…"
for pid in $pids; do
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null
done
echo "Serveur arrêté (SIGKILL)."
