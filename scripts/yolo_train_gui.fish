#!/usr/bin/env fish
# Launch the YOLO training GUI (Tkinter).

set -l root (dirname (dirname (status --current-filename)))
set -l py "$root/.venv/bin/python"

if not test -x $py
  echo "Missing venv python at $py" 1>&2
  echo "Create venv + install deps, then try again." 1>&2
  exit 127
end

exec $py "$root/train_gui.py"
