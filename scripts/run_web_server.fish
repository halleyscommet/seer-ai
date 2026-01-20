#!/usr/bin/env fish
# Start the remote web server (FastAPI + WebSockets) for GPU tracking.
# Usage:
#   ./scripts/run_web_server.fish --host 0.0.0.0 --port 8000

set -e

argparse 'host=' 'port=' -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_host
  set _flag_host 0.0.0.0
end
if not set -q _flag_port
  set _flag_port 8000
end

set -l py python3
if test -x .venv/bin/python
  set py .venv/bin/python
end

exec $py -m uvicorn web.server:app --host $_flag_host --port $_flag_port
