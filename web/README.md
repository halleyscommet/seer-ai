# Remote Web UI (GPU server)

Goal: run YOLO + BoT-SORT on a strong Ubuntu GPU box, while a weaker laptop sends live camera frames over the network. Any browser can view the annotated stream.

## 1) Start the server (Ubuntu + 3070Ti)

From the repo root:

- Install deps: `pip install -r requirements.txt`
- Optional security: set `STREAM_TOKEN` in your `.env` on the server
- Run the web server:

`./scripts/run_web_server.sh --host 0.0.0.0 --port 8000`

(Or fish: `./scripts/run_web_server.fish --host 0.0.0.0 --port 8000`)

Then open:

- `http://<server-ip>:8000/`

If you set `STREAM_TOKEN`, paste it into the token box in the page.

## 2) Start the capture client (competition laptop)

You only need OpenCV + websockets on the laptop:

- Install deps: `pip install opencv-python websockets`

Run:

`./scripts/run_comp_laptop.sh --server <server-ip> --port 8000 --camera 0 --width 640 --fps 20 --jpeg-quality 75 --conf 0.5`

(Or fish: `./scripts/run_comp_laptop.fish --server <server-ip> --port 8000 --camera 0`)

If you set a token on the server:

`./scripts/run_comp_laptop.sh --server <server-ip> --token <STREAM_TOKEN>`

## Notes

- Latency/bandwidth knobs:
  - lower `--width` (e.g. 512)
  - lower `--jpeg-quality` (e.g. 60)
  - cap `--fps` (e.g. 15)
- This uses WebSockets + JPEG frames for simplicity and reliability; it’s not as efficient as WebRTC, but it’s much easier to deploy.
