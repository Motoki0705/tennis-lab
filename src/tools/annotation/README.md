# Video Annotation Tool (MVP)

Local end-to-end workflow for creating annotations from a single video under `data/tmp/`.

## 1) Prepare input
- Put your video at `data/tmp/input.mp4`.

## 2) Run backend (FastAPI)
The repo already defines an optional dependency group `webui` (FastAPI + Uvicorn).

```bash
uv run --group webui -m src.tools.annotation.backend.app --port 8000 \
  --assist-checkpoint third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar \
  --assist-device cpu
```

Optional assist flags:
- `--assist-model` (`wasb` / `hrcnet`)
- `--assist-batch-size`
- `--assist-score-threshold`
- `--assist-max-disp`

## 3) Run frontend (Next.js)
```bash
cd src/tools/annotation/frontend
npm install
NEXT_PUBLIC_ANNOTATION_BACKEND=http://127.0.0.1:8000 npm run dev
```

## 4) Exports written under `data/tmp/`
- Ball (WASB): `data/tmp/wasb/game_tmp/Clip1/`
- Court: `data/tmp/court_keypoints/`

## How to use (short)

### Mode: `ball` (sequential clip)
- Use `[` to mark clip start and `]` to mark clip end on the current frame, then press `Enter` to apply.
- Click to place the ball point, drag to move, and release to auto-save.
- Use **Run assist** to precompute WASB predictions. Assist points render in a different color and can be applied per-frame.
- **Reset** clears the current frame annotation (falls back to assist if available).
- Keys:
  - `←/→` (or `Shift+←/→`): prev/next frame (10 frames with Shift)
  - `S`: save current frame annotation
  - `E`: export WASB clip to `data/tmp/wasb/...`

### Mode: `court` (sparse frames)
- Set `frame idx` to the frame you want to annotate. Only frames you save are exported.
- Select a keypoint, click to place it. After placing, the UI auto-advances to the next unset keypoint.
- Use **Homography fill** (>=4 manual ground points) to auto-place missing keypoints.
- **Reset** clears the current frame annotation.
- Keys:
  - `Tab` / `Shift+Tab`: next/prev keypoint
  - `N`: jump to next unset keypoint
  - `Backspace`: clear active keypoint
  - `←/→` (or `Shift+←/→`): prev/next frame (10 frames with Shift)
  - `S`: save, `E`: export to `data/tmp/court_keypoints/...`
