# Video Annotation Tool (MVP)

Local end-to-end workflow for creating annotations from a single video under `data/tmp/`.

## 1) Prepare input
- Put your video at `data/tmp/input.mp4`.

## 2) Run backend (FastAPI)
The repo already defines an optional dependency group `webui` (FastAPI + Uvicorn).

```bash
uv run --group webui -m src.tools.annotation.backend.app --reload --port 8000
```

## 3) Run frontend (Next.js)
```bash
cd src/tools/annotation/frontend
npm install
NEXT_PUBLIC_ANNOTATION_BACKEND=http://127.0.0.1:8000 npm run dev
```

## 4) Exports written under `data/tmp/`
- Ball (WASB): `data/tmp/wasb/game_tmp/Clip1/`
- Court: `data/tmp/court_keypoints/`

