"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

import { apiDelete, apiGet, apiPost, apiPut } from "./shared/api";
import { clamp, isTypingInField } from "./shared/utils";
import type {
  BallAssistAll,
  BallAssistRunResult,
  BallAssistSummary,
  BallClipConfig,
  BallFrameAnnotation,
  CourtFrameAnnotation,
  ExportResult,
  VideoMeta
} from "./shared/types";
import BallPanel from "./components/BallPanel";
import CanvasStage from "./components/CanvasStage";
import CourtPanel from "./components/CourtPanel";
import SeekBar from "./components/SeekBar";

export default function Page() {
  const [meta, setMeta] = useState<VideoMeta | null>(null);
  const [mode, setMode] = useState<"ball" | "court">("ball");
  const [status, setStatus] = useState<string>("");
  const [isFrameLoading, setIsFrameLoading] = useState<boolean>(false);
  const statusTimeoutRef = useRef<number | null>(null);
  const [seekPreviewIdx, setSeekPreviewIdx] = useState<number | null>(null);
  const [isSeeking, setIsSeeking] = useState<boolean>(false);
  const [resetCacheToken, setResetCacheToken] = useState<number>(0);

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
  const [ballAssistSummary, setBallAssistSummary] = useState<
    BallAssistSummary | null
  >(null);
  const [ballAssistMap, setBallAssistMap] = useState<
    Map<number, BallFrameAnnotation>
  >(new Map());
  const [ballAssistLoading, setBallAssistLoading] = useState<boolean>(false);
  const [ballAnnotatedFrames, setBallAnnotatedFrames] = useState<Set<number>>(
    new Set()
  );

  // Court (sparse frames)
  const [courtFrameIdx, setCourtFrameIdx] = useState<number>(0);
  const [kpNames, setKpNames] = useState<string[]>([]);
  const [activeKp, setActiveKp] = useState<number>(0);
  const [courtAnn, setCourtAnn] = useState<CourtFrameAnnotation | null>(null);

  const globalFrameIdx = useMemo(() => {
    if (mode === "ball") return ballCfg.start_frame + ballLocalIdx;
    return courtFrameIdx;
  }, [mode, ballCfg.start_frame, ballLocalIdx, courtFrameIdx]);

  const ballAssistAnn = useMemo(() => {
    return ballAssistMap.get(ballLocalIdx) ?? null;
  }, [ballAssistMap, ballLocalIdx]);

  const manualCount = useMemo(() => {
    let count = 0;
    for (const idx of ballAnnotatedFrames) {
      if (idx >= 0 && idx < ballCfg.clip_length) count += 1;
    }
    return count;
  }, [ballAnnotatedFrames, ballCfg.clip_length]);

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

  async function refreshBallAssistSummary(): Promise<BallAssistSummary | null> {
    try {
      const summary = await apiGet<BallAssistSummary>("/api/ball/assist/summary");
      setBallAssistSummary(summary);
      return summary;
    } catch (e) {
      setBallAssistSummary(null);
      return null;
    }
  }

  async function loadBallAssistAll(): Promise<void> {
    const data = await apiGet<BallAssistAll>("/api/ball/assist/all");
    const next = new Map<number, BallFrameAnnotation>();
    for (const [k, v] of Object.entries(data.annotations)) {
      next.set(Number(k), v);
    }
    setBallAssistMap(next);
  }

  async function refreshBallAnnotatedFrames(): Promise<void> {
    try {
      const frames = await apiGet<number[]>("/api/ball/annotated_frames");
      setBallAnnotatedFrames(new Set(frames));
    } catch (e) {
      setBallAnnotatedFrames(new Set());
    }
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
        await refreshBallAnnotatedFrames();
        const summary = await refreshBallAssistSummary();
        if (summary?.available && summary.clip_matches_current) {
          await loadBallAssistAll();
        } else {
          setBallAssistMap(new Map());
        }
        setStatus("");
      } catch (e) {
        setStatus(String(e));
      }
    })();
  }, []);

  function setGlobalFrameIdx(next: number) {
    if (!meta) return;
    const clamped = clamp(next, 0, meta.frame_count - 1);
    if (mode === "ball") {
      const local = clamp(
        clamped - ballCfg.start_frame,
        0,
        ballCfg.clip_length - 1
      );
      setBallLocalIdx(local);
    } else {
      setCourtFrameIdx(clamped);
    }
  }

  // Load annotations when frame changes
  useEffect(() => {
    (async () => {
      try {
        if (mode === "ball") {
          setBallAnn({
            visibility: 0,
            x_px: 0,
            y_px: 0,
            score: 0,
            source: "manual"
          });
          const ann = await apiGet<BallFrameAnnotation>(
            `/api/ball/annotations/${ballLocalIdx}`
          );
          setBallAnn(ann);
        } else {
          setCourtAnn(null);
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

  async function saveBall() {
    try {
      const saved = await apiPut<BallFrameAnnotation>(
        `/api/ball/annotations/${ballLocalIdx}`,
        ballAnn
      );
      setBallAnn(saved);
      setBallAnnotatedFrames((prev) => {
        const next = new Set(prev);
        next.add(ballLocalIdx);
        return next;
      });
      setStatusWithTimeout("saved");
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function resetBall() {
    try {
      await apiDelete<{ ok: boolean }>(`/api/ball/annotations/${ballLocalIdx}`);
      setBallAnn({
        visibility: 0,
        x_px: 0,
        y_px: 0,
        score: 0,
        source: "manual"
      });
      setBallAnnotatedFrames((prev) => {
        const next = new Set(prev);
        next.delete(ballLocalIdx);
        return next;
      });
      setStatusWithTimeout("reset");
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

  async function resetCourt() {
    try {
      await apiDelete<{ ok: boolean }>(`/api/court/annotations/${courtFrameIdx}`);
      const ann = await apiGet<CourtFrameAnnotation>(
        `/api/court/annotations/${courtFrameIdx}`
      );
      setCourtAnn(ann);
      setStatusWithTimeout("reset");
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
      await refreshBallAnnotatedFrames();
      await refreshBallAssistSummary();
      setBallAssistMap(new Map());
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function runBallAssist() {
    setBallAssistLoading(true);
    try {
      const result = await apiPost<BallAssistRunResult>("/api/ball/assist/run");
      setStatusWithTimeout(`assist done: ${result.count} frames`);
      const summary = await refreshBallAssistSummary();
      if (summary?.available && summary.clip_matches_current) {
        await loadBallAssistAll();
      }
    } catch (e) {
      setStatus(String(e));
    } finally {
      setBallAssistLoading(false);
    }
  }

  async function applyAssistCurrent() {
    if (!ballAssistAnn) {
      setStatus("assist not available for current frame");
      return;
    }
    const next: BallFrameAnnotation = {
      ...ballAssistAnn,
      source: "assist"
    };
    setBallAnn(next);
    try {
      const saved = await apiPut<BallFrameAnnotation>(
        `/api/ball/annotations/${ballLocalIdx}`,
        next
      );
      setBallAnn(saved);
      setBallAnnotatedFrames((prev) => {
        const updated = new Set(prev);
        updated.add(ballLocalIdx);
        return updated;
      });
      setStatusWithTimeout("assist applied");
    } catch (e) {
      setStatus(String(e));
    }
  }

  async function runCourtHomography() {
    if (!courtAnn) return;
    try {
      const filled = await apiPost<CourtFrameAnnotation>(
        "/api/court/homography",
        courtAnn
      );
      setCourtAnn(filled);
      setStatusWithTimeout("homography filled");
    } catch (e) {
      setStatus(String(e));
    }
  }

  useEffect(() => {
    (async () => {
      const summary = await refreshBallAssistSummary();
      if (summary?.available && summary.clip_matches_current) {
        try {
          await loadBallAssistAll();
        } catch (e) {
          setBallAssistMap(new Map());
        }
        return;
      }
      setBallAssistMap(new Map());
    })();
  }, [ballCfg.start_frame, ballCfg.clip_length]);

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
            return (
              ((next % courtAnn.keypoints.length) + courtAnn.keypoints.length) %
              courtAnn.keypoints.length
            );
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
          const next: CourtFrameAnnotation = {
            ...courtAnn,
            keypoints: courtAnn.keypoints.map((kp, i) =>
              i === activeKp
                ? {
                    ...kp,
                    visibility: 0 as const,
                    x_px: 0,
                    y_px: 0,
                    source: "manual" as const
                  }
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

  return (
    <div className="root">
      <CanvasStage
        meta={meta}
        mode={mode}
        globalFrameIdx={globalFrameIdx}
        ballAnn={ballAnn}
        ballAssistAnn={ballAssistAnn}
        courtAnn={courtAnn}
        activeKp={activeKp}
        isFrameLoading={isFrameLoading}
        resetCacheToken={resetCacheToken}
        onBallAnnChange={setBallAnn}
        onCourtAnnChange={setCourtAnn}
        onActiveKpChange={setActiveKp}
        onBallSave={saveBall}
        onCourtSave={saveCourt}
        onFrameLoadingChange={setIsFrameLoading}
        onStatus={setStatus}
      />

      <div className="panel">
        <SeekBar
          meta={meta}
          globalFrameIdx={globalFrameIdx}
          seekPreviewIdx={seekPreviewIdx}
          isSeeking={isSeeking}
          onSeekStart={() => setIsSeeking(true)}
          onSeekPreviewChange={(idx) => {
            setSeekPreviewIdx(idx);
            if (!isSeeking) setIsSeeking(true);
          }}
          onSeekCommit={(target) => {
            const jump = Math.abs(target - globalFrameIdx);
            if (jump > 18) setResetCacheToken((v) => v + 1);
            setGlobalFrameIdx(target);
            setIsSeeking(false);
            setSeekPreviewIdx(null);
          }}
        />

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
          <div className="small">
            {isFrameLoading ? "loading frame..." : "ready"}
          </div>
        </div>

        {mode === "ball" ? (
          <BallPanel
            meta={meta}
            ballCfg={ballCfg}
            ballLocalIdx={ballLocalIdx}
            ballClipMarkStart={ballClipMarkStart}
            ballClipMarkEnd={ballClipMarkEnd}
            assistSummary={ballAssistSummary}
            assistLoading={ballAssistLoading}
            assistAnn={ballAssistAnn}
            manualCount={manualCount}
            onBallCfgChange={setBallCfg}
            onSetClip={async () => {
              if (!meta) return;
              try {
                const next = {
                  start_frame: clamp(ballCfg.start_frame, 0, meta.frame_count - 1),
                  clip_length: clamp(ballCfg.clip_length, 1, meta.frame_count)
                };
                const saved = await apiPut<BallClipConfig>(
                  "/api/ball/clip_config",
                  next
                );
                setBallCfg(saved);
                setBallLocalIdx(0);
                setStatusWithTimeout("clip config saved");
                await refreshBallAnnotatedFrames();
                await refreshBallAssistSummary();
                setBallAssistMap(new Map());
              } catch (e) {
                setStatus(String(e));
              }
            }}
            onMarkStart={() => {
              setBallClipMarkStart(globalFrameIdx);
              setStatusWithTimeout(`clip start = ${globalFrameIdx}`);
            }}
            onMarkEnd={() => {
              setBallClipMarkEnd(globalFrameIdx);
              setStatusWithTimeout(`clip end = ${globalFrameIdx}`);
            }}
            onApplyClipMarks={applyBallClipMarks}
            onBallLocalIdxChange={setBallLocalIdx}
            onSaveBall={saveBall}
            onResetBall={resetBall}
            onExportBall={exportCurrentMode}
            onRunAssist={runBallAssist}
            onApplyAssist={applyAssistCurrent}
          />
        ) : (
          <CourtPanel
            meta={meta}
            courtFrameIdx={courtFrameIdx}
            courtAnn={courtAnn}
            kpNames={kpNames}
            activeKp={activeKp}
            onCourtFrameIdxChange={setCourtFrameIdx}
            onActiveKpChange={setActiveKp}
            onSaveCourt={() => {
              if (courtAnn) void saveCourt(courtAnn);
            }}
            onResetCourt={resetCourt}
            onExportCourt={exportCurrentMode}
            onRunHomography={runCourtHomography}
          />
        )}

        <div className="row">
          <div className="small">{status}</div>
        </div>
      </div>
    </div>
  );
}
