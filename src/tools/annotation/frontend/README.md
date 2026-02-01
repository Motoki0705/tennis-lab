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

## How to use (MVP)

### Common
- `←/→`: previous/next frame (`Shift+←/→` moves by 10)
- `S`: save
- `E`: export (ball -> WASB, court -> court_keypoints)

### Ball mode (sequential clip)
- `[` / `]`: mark clip start/end using the current global frame index
- `Enter`: apply the marked clip
- Click to place the ball point, drag to move (auto-save on mouse up)

### Court mode (sparse frames)
- Only annotated frames are exported (frames saved via `Save` / `S`)
- Click to place the currently selected keypoint; the UI auto-advances to the next unset keypoint
- `Tab` / `Shift+Tab`: next/previous keypoint
- `N`: jump to next unset keypoint
- `Backspace`: clear active keypoint
