# Annotation Frontend (MVP)

This is a minimal Next.js + TypeScript UI for annotating:
- Ball center (single point per frame, sequential clip)
- Court keypoints (CourtKP20, only saves annotated frames)

## Prerequisites
- Node.js 18+ (recommended)
- Backend running locally (see `src/tools/annotation/backend/app.py`)

## Install
```bash
cd src/tools/annotation/frontend
npm install
```

## Run (dev)
```bash
cd src/tools/annotation/frontend
NEXT_PUBLIC_ANNOTATION_BACKEND=http://127.0.0.1:8000 npm run dev
```

