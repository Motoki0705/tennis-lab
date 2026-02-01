"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

type VideoMeta = {
  fps: number;
  frame_count: number;
  width: number;
  height: number;
};

type BallClipConfig = {
  start_frame: number;
  clip_length: number;
};

type BallFrameAnnotation = {
  visibility: 0 | 1 | 2;
  x_px: number;
  y_px: number;
  score: number;
  source: "manual" | "assist" | "unknown";
};

type CourtKeypoint = {
  x_px: number;
  y_px: number;
  visibility: 0 | 1;
  source: "manual" | "assist" | "homography" | "unknown";
};

type CourtFrameAnnotation = {
  frame_idx: number;
  keypoints: CourtKeypoint[];
};

type ExportResult = { output_dir: string };

function apiBase(): string {
  return process.env.NEXT_PUBLIC_ANNOTATION_BACKEND ?? "http://127.0.0.1:8000";
}

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [meta, setMeta] = useState<VideoMeta | null>(null);
  const [mode, setMode] = useState<"ball" | "court">("ball");
  const [status, setStatus] = useState<string>("");
  const statusTimeoutRef = useRef<number | null>(null);

  // Ball (sequential clip)
  const [ballCfg, setBallCfg] = useState<BallClipConfig>({
    start_frame: 0,
    clip_length: 300
  });
  const [ballLocalIdx, setBallLocalIdx] = useState<number>(0);
  const [ballAnn, setBallAnn] = useState<BallFrameAnnotation>({
    visibility: 0,
    x_px: 0,
    y_px: 0,
    score: 0,
    source: "manual"
  });
  const [ballClipMarkStart, setBallClipMarkStart] = useState<number | null>(null);
  const [ballClipMarkEnd, setBallClipMarkEnd] = useState<number | null>(null);

  // Court (sparse frames)
  const [courtFrameIdx, setCourtFrameIdx] = useState<number>(0);
  const [kpNames, setKpNames] = useState<string[]>([]);
  const [activeKp, setActiveKp] = useState<number>(0);
  const [courtAnn, setCourtAnn] = useState<CourtFrameAnnotation | null>(null);

  const globalFrameIdx = useMemo(() => {
    if (mode === "ball") return ballCfg.start_frame + ballLocalIdx;
    return courtFrameIdx;
  }, [mode, ballCfg.start_frame, ballLocalIdx, courtFrameIdx]);

  const dragRef = useRef<{
    kind: "ball" | "court" | null;
    kpIndex: number;
  }>({ kind: null, kpIndex: -1 });

  function setStatusWithTimeout(msg: string, ms: number = 1200) {
    setStatus(msg);
    if (statusTimeoutRef.current) {
      window.clearTimeout(statusTimeoutRef.current);
    }
    statusTimeoutRef.current = window.setTimeout(() => {
      setStatus("");
      statusTimeoutRef.current = null;
    }, ms);
  }

  function isTypingInField(): boolean {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName.toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select";
  }

  useEffect(() => {
    (async () => {
      try {
        const m = await apiGet<VideoMeta>("/api/meta");
        setMeta(m);
        const cfg = await apiGet<BallClipConfig>("/api/ball/clip_config");
        setBallCfg(cfg);
        setBallLocalIdx(0);
        const names = await apiGet<string[]>("/api/court/kp_names");
        setKpNames(names);
        setStatus("");
      } catch (e) {
        setStatus(String(e));
      }
    })();
  }, []);

  // Load annotations when frame changes
  useEffect(() => {
    (async () => {
      try {
        if (mode === "ball") {
          const ann = await apiGet<BallFrameAnnotation>(
            `/api/ball/annotations/${ballLocalIdx}`
          );
          setBallAnn(ann);
        } else {
          const ann = await apiGet<CourtFrameAnnotation>(
            `/api/court/annotations/${courtFrameIdx}`
          );
          setCourtAnn(ann);
        }
        setStatus("");
      } catch (e) {
        setStatus(String(e));
      }
    })();
  }, [mode, ballLocalIdx, courtFrameIdx]);

  // Draw frame + overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !meta) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `${apiBase()}/api/frame/${globalFrameIdx}.jpg`;
    img.onload = () => {
      canvas.width = meta.width;
      canvas.height = meta.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // overlays
      if (mode === "ball") {
        if (ballAnn.visibility > 0) {
          ctx.fillStyle = "#00E5FF";
          ctx.strokeStyle = "#001018";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(ballAnn.x_px, ballAnn.y_px, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      } else if (mode === "court" && courtAnn) {
        for (let i = 0; i < courtAnn.keypoints.length; i++) {
          const kp = courtAnn.keypoints[i];
          if (kp.visibility === 0) continue;
          ctx.fillStyle = i === activeKp ? "#FFB020" : "#22C55E";
          ctx.strokeStyle = "#111827";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(kp.x_px, kp.y_px, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    };
  }, [meta, mode, globalFrameIdx, ballAnn, courtAnn, activeKp]);

  async function saveBall() {
    try {
      const saved = await apiPut<BallFrameAnnotation>(
        `/api/ball/annotations/${ballLocalIdx}`,
        ballAnn
      );
      setBallAnn(saved);
      setStatusWithTimeout("saved");
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function saveCourt(next: CourtFrameAnnotation) {
    try {
      const saved = await apiPut<CourtFrameAnnotation>(
        `/api/court/annotations/${courtFrameIdx}`,
        next
      );
      setCourtAnn(saved);
      setStatusWithTimeout("saved");
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function exportCurrentMode() {
    try {
      if (mode === "ball") {
        const r = await apiPost<ExportResult>("/api/export/wasb");
        setStatus(`exported: ${r.output_dir}`);
        return;
      }
      const r = await apiPost<ExportResult>("/api/export/court");
      setStatus(`exported: ${r.output_dir}`);
    } catch (e) {
      setStatus(String(e));
    }
  }

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

  async function applyBallClipMarks() {
    if (!meta) return;
    if (ballClipMarkStart === null || ballClipMarkEnd === null) {
      setStatus("set both clip start/end first ([ and ])");
      return;
    }
    const start = clamp(
      Math.min(ballClipMarkStart, ballClipMarkEnd),
      0,
      meta.frame_count - 1
    );
    const end = clamp(
      Math.max(ballClipMarkStart, ballClipMarkEnd),
      0,
      meta.frame_count - 1
    );
    const clip_length = end - start + 1;
    try {
      const saved = await apiPut<BallClipConfig>("/api/ball/clip_config", {
        start_frame: start,
        clip_length
      });
      setBallCfg(saved);
      setBallLocalIdx(0);
      setStatusWithTimeout(`clip set: ${start}..${end}`);
    } catch (e) {
      setStatus(String(e));
    }
  }

  // Keyboard shortcuts (common + per-mode)
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!meta) return;
      if (isTypingInField()) return;

      // Common
      if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        if (mode === "ball") void saveBall();
        if (mode === "court" && courtAnn) void saveCourt(courtAnn);
        return;
      }
      if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        void exportCurrentMode();
        return;
      }

      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowLeft") {
        if (mode === "ball") {
          setBallLocalIdx((v) => clamp(v - step, 0, ballCfg.clip_length - 1));
        } else {
          setCourtFrameIdx((v) => clamp(v - step, 0, meta.frame_count - 1));
        }
      }
      if (e.key === "ArrowRight") {
        if (mode === "ball") {
          setBallLocalIdx((v) => clamp(v + step, 0, ballCfg.clip_length - 1));
        } else {
          setCourtFrameIdx((v) => clamp(v + step, 0, meta.frame_count - 1));
        }
      }

      // Ball: clip marking from current frame
      if (mode === "ball") {
        if (e.key === "[") {
          e.preventDefault();
          setBallClipMarkStart(globalFrameIdx);
          setStatusWithTimeout(`clip start = ${globalFrameIdx}`);
          return;
        }
        if (e.key === "]") {
          e.preventDefault();
          setBallClipMarkEnd(globalFrameIdx);
          setStatusWithTimeout(`clip end = ${globalFrameIdx}`);
          return;
        }
        if (e.key === "Enter") {
          e.preventDefault();
          void applyBallClipMarks();
          return;
        }
      }

      // Court: keypoint navigation / edit
      if (mode === "court" && courtAnn) {
        if (e.key === "Tab") {
          e.preventDefault();
          setActiveKp((v) => {
            const next = e.shiftKey ? v - 1 : v + 1;
            return ((next % courtAnn.keypoints.length) + courtAnn.keypoints.length) % courtAnn.keypoints.length;
          });
          return;
        }
        if (e.key === "n" || e.key === "N") {
          e.preventDefault();
          const next = nextUnsetKpIndex(courtAnn, activeKp);
          if (next !== null) setActiveKp(next);
          return;
        }
        if (e.key === "Backspace") {
          e.preventDefault();
          const next = {
            ...courtAnn,
            keypoints: courtAnn.keypoints.map((kp, i) =>
              i === activeKp
                ? { ...kp, visibility: 0, x_px: 0, y_px: 0, source: "manual" }
                : kp
            )
          };
          setCourtAnn(next);
          void saveCourt(next);
          return;
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    meta,
    mode,
    globalFrameIdx,
    ballCfg.clip_length,
    ballLocalIdx,
    ballAnn,
    courtAnn,
    activeKp,
    ballClipMarkStart,
    ballClipMarkEnd
  ]);

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

  return (
    <div className="root">
      <div className="canvasWrap">
        <canvas
          ref={canvasRef}
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
                setBallAnn({
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
                setActiveKp(nearest);
              } else {
                const autoNext = nextUnsetKpIndex(courtAnn, activeKp);
                const next = {
                  ...courtAnn,
                  keypoints: courtAnn.keypoints.map((kp, i) =>
                    i === activeKp
                      ? { ...kp, visibility: 1, x_px: x, y_px: y, source: "manual" }
                      : kp
                  )
                };
                setCourtAnn(next);
                dragRef.current = { kind: "court", kpIndex: activeKp };
                if (autoNext !== null && autoNext !== activeKp) setActiveKp(autoNext);
              }
            }
          }}
          onMouseMove={(e) => {
            if (!meta) return;
            const drag = dragRef.current;
            if (!drag.kind) return;
            const { x, y } = toCanvasXY(e);
            if (drag.kind === "ball") {
              setBallAnn({
                ...ballAnn,
                visibility: 1,
                x_px: x,
                y_px: y,
                source: "manual"
              });
            } else if (drag.kind === "court" && courtAnn) {
              const idx = drag.kpIndex;
              if (idx < 0) return;
              const next = {
                ...courtAnn,
                keypoints: courtAnn.keypoints.map((kp, i) =>
                  i === idx
                    ? { ...kp, visibility: 1, x_px: x, y_px: y, source: "manual" }
                    : kp
                )
              };
              setCourtAnn(next);
            }
          }}
          onMouseUp={async () => {
            const drag = dragRef.current;
            dragRef.current = { kind: null, kpIndex: -1 };
            if (drag.kind === "ball") await saveBall();
            if (drag.kind === "court" && courtAnn) await saveCourt(courtAnn);
          }}
          onMouseLeave={() => {
            dragRef.current = { kind: null, kpIndex: -1 };
          }}
        />
      </div>

      <div className="panel">
        <div className="row">
          <label>Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "ball" | "court")}
          >
            <option value="ball">ball (sequential clip)</option>
            <option value="court">court (sparse frames)</option>
          </select>
        </div>

        <div className="row">
          <div className="small">
            frame: {globalFrameIdx}
            {meta ? ` / ${meta.frame_count - 1}` : ""}
          </div>
        </div>

        {mode === "ball" ? (
          <>
            <div className="row">
              <label>clip start</label>
              <input
                type="number"
                value={ballCfg.start_frame}
                onChange={(e) =>
                  setBallCfg({
                    ...ballCfg,
                    start_frame: Number(e.target.value)
                  })
                }
              />
              <label>clip length</label>
              <input
                type="number"
                value={ballCfg.clip_length}
                onChange={(e) =>
                  setBallCfg({
                    ...ballCfg,
                    clip_length: Number(e.target.value)
                  })
                }
              />
              <button
                className="primary"
                onClick={async () => {
                  if (!meta) return;
                  try {
                    const next = {
                      start_frame: clamp(ballCfg.start_frame, 0, meta.frame_count - 1),
                      clip_length: clamp(
                        ballCfg.clip_length,
                        1,
                        meta.frame_count
                      )
                    };
                    const saved = await apiPut<BallClipConfig>(
                      "/api/ball/clip_config",
                      next
                    );
                    setBallCfg(saved);
                    setBallLocalIdx(0);
                    setStatusWithTimeout("clip config saved");
                  } catch (e) {
                    setStatus(String(e));
                  }
                }}
              >
                Set clip
              </button>
            </div>

            <div className="row">
              <button
                onClick={() => {
                  setBallClipMarkStart(globalFrameIdx);
                  setStatusWithTimeout(`clip start = ${globalFrameIdx}`);
                }}
              >
                Mark start [
              </button>
              <button
                onClick={() => {
                  setBallClipMarkEnd(globalFrameIdx);
                  setStatusWithTimeout(`clip end = ${globalFrameIdx}`);
                }}
              >
                Mark end ]
              </button>
              <div className="small">
                marked: {ballClipMarkStart ?? "—"} .. {ballClipMarkEnd ?? "—"}
              </div>
              <button className="primary" onClick={applyBallClipMarks}>
                Apply (Enter)
              </button>
            </div>

            <div className="row">
              <label>local idx</label>
              <input
                type="number"
                value={ballLocalIdx}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setBallLocalIdx(clamp(v, 0, ballCfg.clip_length - 1));
                }}
              />
              <button className="primary" onClick={saveBall}>
                Save
              </button>
              <button
                onClick={exportCurrentMode}
              >
                Export WASB
              </button>
            </div>

            <div className="row">
              <div className="small">
                click to place; drag to move; auto-save on mouse up
              </div>
            </div>
            <div className="row">
              <div className="small">
                keys: [ ] mark clip, Enter apply, S save, E export, ←/→ navigate
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="row">
              <label>frame idx</label>
              <input
                type="number"
                value={courtFrameIdx}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (!meta) return;
                  setCourtFrameIdx(clamp(v, 0, meta.frame_count - 1));
                }}
              />
              <button
                className="primary"
                onClick={async () => {
                  if (courtAnn) await saveCourt(courtAnn);
                }}
              >
                Save
              </button>
              <button
                onClick={exportCurrentMode}
              >
                Export Court
              </button>
            </div>

            <div className="small">
              Select a keypoint, then click to place. Drag existing points to move.
            </div>
            <div className="small">
              keys: Tab/Shift+Tab next/prev kp, N next unset, Backspace clear, S save, E export
            </div>

            <div className="kpList">
              {(kpNames.length ? kpNames : Array.from({ length: 20 }, (_, i) => `kp_${i}`)).map(
                (name, i) => (
                  <div
                    key={i}
                    className={`kpItem ${i === activeKp ? "active" : ""}`}
                    onClick={() => setActiveKp(i)}
                  >
                    <div className="small">
                      {i}: {name}
                    </div>
                    <div className="small">
                      {courtAnn?.keypoints?.[i]?.visibility ? "set" : "—"}
                    </div>
                  </div>
                )
              )}
            </div>
          </>
        )}

        <div className="row">
          <div className="small">{status}</div>
        </div>
      </div>
    </div>
  );
}
