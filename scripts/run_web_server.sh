#!/usr/bin/env sh
# Start the remote web server (FastAPI + WebSockets) for GPU tracking.
# Usage:
#   ./scripts/run_web_server.sh [--host 0.0.0.0] [--port 8000]

set -eu

HOST="0.0.0.0"
PORT="8000"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--host 0.0.0.0] [--port 8000]";
      exit 0;;
    *)
      echo "Unknown arg: $1" 1>&2
      exit 2;;
  esac
done

# Prefer repo venv if present.
PY="python3"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
fi

exec "$PY" -m uvicorn web.server:app --host "$HOST" --port "$PORT"
