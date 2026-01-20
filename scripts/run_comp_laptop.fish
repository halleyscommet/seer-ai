#!/usr/bin/env fish
# Competition laptop helper:
#  - starts the capture client (camera -> server)
#  - opens the viewer page in your browser
#
# Usage:
#   ./scripts/run_comp_laptop.fish --server 192.168.1.50 --camera 0

set -e

argparse \
  'server=' \
  'port=' \
  'camera=' \
  'width=' \
  'fps=' \
  'jpeg-quality=' \
  'conf=' \
  'token=' \
  -- $argv
or begin
  echo "Failed to parse arguments." 1>&2
  exit 2
end

if not set -q _flag_server
  echo "Missing --server (IP or hostname of the GPU box)." 1>&2
  exit 2
end

if not set -q _flag_port
  set _flag_port 8000
end
if not set -q _flag_camera
  set _flag_camera 0
end
if not set -q _flag_width
  set _flag_width 640
end
if not set -q _flag_fps
  set _flag_fps 20
end
if not set -q _flag_jpeg_quality
  set _flag_jpeg_quality 75
end
if not set -q _flag_conf
  set _flag_conf 0.5
end

set -l token ""
if set -q _flag_token
  set token $_flag_token
else if set -q STREAM_TOKEN
  set token $STREAM_TOKEN
end

set -l http_url "http://$_flag_server:$_flag_port/"
set -l ws_url "ws://$_flag_server:$_flag_port"

# Open browser best-effort.
if command -q xdg-open
  xdg-open "$http_url" >/dev/null 2>&1 &
else if command -q open
  open "$http_url" >/dev/null 2>&1 &
else
  echo "Open this URL in a browser: $http_url"
end

set -l py python3
if test -x .venv/bin/python
  set py .venv/bin/python
end

function _cleanup --on-signal INT --on-signal TERM
  if set -q client_pid
    kill $client_pid >/dev/null 2>&1
  end
end

if test -n "$token"
  $py web/capture_client.py --server "$ws_url" --token "$token" --camera $_flag_camera --width $_flag_width --fps $_flag_fps --jpeg-quality $_flag_jpeg_quality --conf $_flag_conf &
else
  $py web/capture_client.py --server "$ws_url" --camera $_flag_camera --width $_flag_width --fps $_flag_fps --jpeg-quality $_flag_jpeg_quality --conf $_flag_conf &
end
set -g client_pid $last_pid

echo "Viewer:  $http_url"
echo "Capture: camera=$_flag_camera width=$_flag_width fps=$_flag_fps jpeg=$_flag_jpeg_quality conf=$_flag_conf"
wait $client_pid
