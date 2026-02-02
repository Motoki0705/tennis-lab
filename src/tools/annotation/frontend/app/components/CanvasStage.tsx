"use client";

import React, { useEffect, useMemo, useRef } from "react";

import { apiBase } from "../shared/api";
import { clamp } from "../shared/utils";
import type {
  BallFrameAnnotation,
  CourtFrameAnnotation,
  VideoMeta
} from "../shared/types";

type CanvasStageProps = {
  meta: VideoMeta | null;
  mode: "ball" | "court";
  globalFrameIdx: number;
  ballAnn: BallFrameAnnotation;
  ballAssistAnn: BallFrameAnnotation | null;
  courtAnn: CourtFrameAnnotation | null;
  activeKp: number;
  isFrameLoading: boolean;
  resetCacheToken: number;
  onBallAnnChange: (ann: BallFrameAnnotation) => void;
  onCourtAnnChange: (ann: CourtFrameAnnotation) => void;
  onActiveKpChange: (idx: number) => void;
  onBallSave: () => Promise<void>;
  onCourtSave: (ann: CourtFrameAnnotation) => Promise<void>;
  onFrameLoadingChange: (loading: boolean) => void;
  onStatus: (msg: string) => void;
};

function nextUnsetKpIndex(
  ann: CourtFrameAnnotation,
  fromIdx: number
): number | null {
  for (let i = fromIdx + 1; i < ann.keypoints.length; i++) {
    if (ann.keypoints[i].visibility === 0) return i;
  }
  for (let i = 0; i <= fromIdx; i++) {
    if (ann.keypoints[i].visibility === 0) return i;
  }
  return null;
}

function findNearestCourtKp(
  x: number,
  y: number,
  ann: CourtFrameAnnotation,
  radiusPx: number
): number {
  let best = -1;
  let bestD2 = radiusPx * radiusPx;
  for (let i = 0; i < ann.keypoints.length; i++) {
    const kp = ann.keypoints[i];
    if (kp.visibility === 0) continue;
    const dx = kp.x_px - x;
    const dy = kp.y_px - y;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) {
      bestD2 = d2;
      best = i;
    }
  }
  return best;
}

function courtColor(source: string, isActive: boolean): string {
  if (isActive) return "#FFB020";
  if (source === "homography") return "#A855F7";
  if (source === "assist") return "#60A5FA";
  return "#22C55E";
}

export default function CanvasStage({
  meta,
  mode,
  globalFrameIdx,
  ballAnn,
  ballAssistAnn,
  courtAnn,
  activeKp,
  isFrameLoading,
  resetCacheToken,
  onBallAnnChange,
  onCourtAnnChange,
  onActiveKpChange,
  onBallSave,
  onCourtSave,
  onFrameLoadingChange,
  onStatus
}: CanvasStageProps) {
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const renderTokenRef = useRef<number>(0);
  const frameCacheRef = useRef<Map<number, HTMLImageElement>>(new Map());
  const frameLoadRef = useRef<Map<number, Promise<HTMLImageElement>>>(new Map());
  const preloadGenRef = useRef<number>(0);
  const dragRef = useRef<{
    kind: "ball" | "court" | null;
    kpIndex: number;
  }>({ kind: null, kpIndex: -1 });

  const frameUrl = useMemo(() => {
    return (idx: number) => `${apiBase()}/api/frame/${idx}.jpg`;
  }, []);

  useEffect(() => {
    frameCacheRef.current.clear();
    frameLoadRef.current.clear();
    preloadGenRef.current += 1;
  }, [resetCacheToken]);

  useEffect(() => {
    if (!meta) return;
    const frameCanvas = frameCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (frameCanvas) {
      frameCanvas.width = meta.width;
      frameCanvas.height = meta.height;
    }
    if (overlayCanvas) {
      overlayCanvas.width = meta.width;
      overlayCanvas.height = meta.height;
    }
  }, [meta]);

  function getCachedFrame(idx: number): HTMLImageElement | null {
    const cached = frameCacheRef.current.get(idx);
    if (cached && cached.complete && cached.naturalWidth > 0) return cached;
    return null;
  }

  function preloadFrame(idx: number): Promise<HTMLImageElement> {
    const cached = getCachedFrame(idx);
    if (cached) return Promise.resolve(cached);
    const inflight = frameLoadRef.current.get(idx);
    if (inflight) return inflight;

    const img = new Image();
    img.crossOrigin = "anonymous";
    const promise = new Promise<HTMLImageElement>((resolve, reject) => {
      img.onload = () => {
        frameCacheRef.current.set(idx, img);
        frameLoadRef.current.delete(idx);
        resolve(img);
      };
      img.onerror = () => {
        frameLoadRef.current.delete(idx);
        reject(new Error(`failed to load frame ${idx}`));
      };
    });
    frameLoadRef.current.set(idx, promise);
    img.src = frameUrl(idx);
    return promise;
  }

  function pruneFrameCache(centerIdx: number, maxSize: number) {
    const cache = frameCacheRef.current;
    if (cache.size <= maxSize) return;
    const entries = Array.from(cache.keys());
    entries.sort((a, b) => Math.abs(b - centerIdx) - Math.abs(a - centerIdx));
    const toRemove = entries.slice(0, Math.max(0, entries.length - maxSize));
    for (const idx of toRemove) cache.delete(idx);
  }

  function preloadRange(centerIdx: number, radius: number, maxSize: number) {
    const gen = ++preloadGenRef.current;
    const indices: number[] = [centerIdx];
    for (let i = 1; i <= radius; i++) {
      indices.push(centerIdx + i, centerIdx - i);
    }
    for (const idx of indices) {
      if (!meta) return;
      if (idx < 0 || idx >= meta.frame_count) continue;
      if (getCachedFrame(idx)) continue;
      void preloadFrame(idx).catch(() => {
        if (gen !== preloadGenRef.current) return;
      });
    }
    pruneFrameCache(centerIdx, maxSize);
  }

  useEffect(() => {
    const frameCanvas = frameCanvasRef.current;
    if (!frameCanvas || !meta) return;

    const token = ++renderTokenRef.current;
    let canceled = false;
    const ctx = frameCanvas.getContext("2d");
    if (!ctx) return;
    const cached = getCachedFrame(globalFrameIdx);
    if (cached) {
      ctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
      ctx.drawImage(cached, 0, 0, frameCanvas.width, frameCanvas.height);
      onFrameLoadingChange(false);
    } else {
      onFrameLoadingChange(true);
      void preloadFrame(globalFrameIdx)
        .then((img) => {
          if (canceled) return;
          if (token !== renderTokenRef.current) return;
          ctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
          ctx.drawImage(img, 0, 0, frameCanvas.width, frameCanvas.height);
          onFrameLoadingChange(false);
        })
        .catch(() => {
          if (canceled) return;
          if (token !== renderTokenRef.current) return;
          onFrameLoadingChange(false);
          onStatus("failed to load frame image (check backend / video / frame idx)");
        });
    }

    preloadRange(globalFrameIdx, 6, 48);

    return () => {
      canceled = true;
    };
  }, [meta, globalFrameIdx, onFrameLoadingChange, onStatus]);

  useEffect(() => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas || !meta) return;
    const ctx = overlayCanvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (mode === "ball") {
      if (ballAssistAnn && ballAssistAnn.visibility > 0) {
        ctx.strokeStyle = "#A855F7";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(ballAssistAnn.x_px, ballAssistAnn.y_px, 7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (ballAnn.visibility > 0) {
        ctx.fillStyle = "#00E5FF";
        ctx.strokeStyle = "#001018";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(ballAnn.x_px, ballAnn.y_px, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      return;
    }

    if (mode === "court" && courtAnn) {
      for (let i = 0; i < courtAnn.keypoints.length; i++) {
        const kp = courtAnn.keypoints[i];
        if (kp.visibility === 0) continue;
        ctx.fillStyle = courtColor(kp.source, i === activeKp);
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(kp.x_px, kp.y_px, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }, [meta, mode, globalFrameIdx, ballAnn, ballAssistAnn, courtAnn, activeKp]);

  function toCanvasXY(e: React.MouseEvent<HTMLCanvasElement>): {
    x: number;
    y: number;
  } {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    return { x, y };
  }

  return (
    <div className="canvasWrap">
      <div
        className="canvasStage"
        style={{
          aspectRatio: meta ? `${meta.width} / ${meta.height}` : "16 / 9"
        }}
      >
        <div className={`loadingBadge ${isFrameLoading ? "show" : ""}`}>
          Loading...
        </div>
        <canvas ref={frameCanvasRef} className="frameCanvas" />
        <canvas
          ref={overlayCanvasRef}
          className="overlayCanvas"
          onMouseDown={(e) => {
            if (!meta) return;
            const { x, y } = toCanvasXY(e);
            if (mode === "ball") {
              const dx = ballAnn.x_px - x;
              const dy = ballAnn.y_px - y;
              const near = ballAnn.visibility > 0 && dx * dx + dy * dy <= 12 * 12;
              if (near) {
                dragRef.current = { kind: "ball", kpIndex: -1 };
              } else {
                onBallAnnChange({
                  ...ballAnn,
                  visibility: 1,
                  x_px: x,
                  y_px: y,
                  source: "manual"
                });
                dragRef.current = { kind: "ball", kpIndex: -1 };
              }
            } else if (mode === "court" && courtAnn) {
              const nearest = findNearestCourtKp(x, y, courtAnn, 10);
              if (nearest >= 0) {
                dragRef.current = { kind: "court", kpIndex: nearest };
                onActiveKpChange(nearest);
              } else {
                const autoNext = nextUnsetKpIndex(courtAnn, activeKp);
                const next: CourtFrameAnnotation = {
                  ...courtAnn,
                  keypoints: courtAnn.keypoints.map((kp, i) =>
                    i === activeKp
                      ? {
                          ...kp,
                          visibility: 1,
                          x_px: x,
                          y_px: y,
                          source: "manual"
                        }
                      : kp
                  )
                };
                onCourtAnnChange(next);
                dragRef.current = { kind: "court", kpIndex: activeKp };
                if (autoNext !== null && autoNext !== activeKp)
                  onActiveKpChange(autoNext);
              }
            }
          }}
          onMouseMove={(e) => {
            if (!meta) return;
            const drag = dragRef.current;
            if (!drag.kind) return;
            const { x, y } = toCanvasXY(e);
            if (drag.kind === "ball") {
              onBallAnnChange({
                ...ballAnn,
                visibility: 1,
                x_px: x,
                y_px: y,
                source: "manual"
              });
            } else if (drag.kind === "court" && courtAnn) {
              const idx = clamp(drag.kpIndex, 0, courtAnn.keypoints.length - 1);
              const next: CourtFrameAnnotation = {
                ...courtAnn,
                keypoints: courtAnn.keypoints.map((kp, i) =>
                  i === idx
                    ? {
                        ...kp,
                        visibility: 1,
                        x_px: x,
                        y_px: y,
                        source: "manual"
                      }
                    : kp
                )
              };
              onCourtAnnChange(next);
            }
          }}
          onMouseUp={async () => {
            const drag = dragRef.current;
            dragRef.current = { kind: null, kpIndex: -1 };
            if (drag.kind === "ball") await onBallSave();
            if (drag.kind === "court" && courtAnn) await onCourtSave(courtAnn);
          }}
          onMouseLeave={() => {
            dragRef.current = { kind: null, kpIndex: -1 };
          }}
        />
      </div>
    </div>
  );
}
