"use client";

import React from "react";

import type {
  BallAssistSummary,
  BallClipConfig,
  BallFrameAnnotation,
  VideoMeta
} from "../shared/types";
import { clamp, formatAssistMetaSummary } from "../shared/utils";

type BallPanelProps = {
  meta: VideoMeta | null;
  ballCfg: BallClipConfig;
  ballLocalIdx: number;
  ballClipMarkStart: number | null;
  ballClipMarkEnd: number | null;
  assistSummary: BallAssistSummary | null;
  assistLoading: boolean;
  assistAnn: BallFrameAnnotation | null;
  manualCount: number | null;
  onBallCfgChange: (cfg: BallClipConfig) => void;
  onSetClip: () => void;
  onMarkStart: () => void;
  onMarkEnd: () => void;
  onApplyClipMarks: () => void;
  onBallLocalIdxChange: (idx: number) => void;
  onSaveBall: () => void;
  onResetBall: () => void;
  onExportBall: () => void;
  onRunAssist: () => void;
  onApplyAssist: () => void;
};

export default function BallPanel({
  meta,
  ballCfg,
  ballLocalIdx,
  ballClipMarkStart,
  ballClipMarkEnd,
  assistSummary,
  assistLoading,
  assistAnn,
  manualCount,
  onBallCfgChange,
  onSetClip,
  onMarkStart,
  onMarkEnd,
  onApplyClipMarks,
  onBallLocalIdxChange,
  onSaveBall,
  onResetBall,
  onExportBall,
  onRunAssist,
  onApplyAssist
}: BallPanelProps) {
  const assistAvailable = Boolean(
    assistSummary && assistSummary.available && assistSummary.clip_matches_current
  );
  const assistCount = assistSummary?.count ?? 0;
  const manualText =
    manualCount === null
      ? "manual: —"
      : `manual: ${manualCount}/${ballCfg.clip_length}`;

  return (
    <>
      <div className="row">
        <label>clip start</label>
        <input
          type="number"
          value={ballCfg.start_frame}
          onChange={(e) =>
            onBallCfgChange({
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
            onBallCfgChange({
              ...ballCfg,
              clip_length: Number(e.target.value)
            })
          }
        />
        <button className="primary" onClick={onSetClip}>
          Set clip
        </button>
      </div>

      <div className="row">
        <button onClick={onMarkStart}>Mark start [</button>
        <button onClick={onMarkEnd}>Mark end ]</button>
        <div className="small">
          marked: {ballClipMarkStart ?? "—"} .. {ballClipMarkEnd ?? "—"}
        </div>
        <button className="primary" onClick={onApplyClipMarks}>
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
            onBallLocalIdxChange(
              clamp(v, 0, Math.max(0, ballCfg.clip_length - 1))
            );
          }}
        />
        <button className="primary" onClick={onSaveBall}>
          Save
        </button>
        <button onClick={onResetBall}>Reset</button>
        <button onClick={onExportBall}>Export WASB</button>
      </div>

      <div className="row">
        <button className="primary" onClick={onRunAssist} disabled={assistLoading}>
          {assistLoading ? "Running assist..." : "Run assist"}
        </button>
        <button
          onClick={onApplyAssist}
          disabled={!assistAvailable || !assistAnn}
        >
          Apply assist (current)
        </button>
      </div>

      <div className="row">
        <div className={`badge ${assistAvailable ? "ok" : "warn"}`}>
          assist: {assistAvailable ? "ready" : "not ready"}
        </div>
        <div className="small">
          assist frames: {assistCount}/{ballCfg.clip_length}
        </div>
      </div>
      <div className="row">
        <div className="small">{manualText}</div>
      </div>
      <div className="row">
        <div className="small">{formatAssistMetaSummary(assistSummary?.meta ?? null)}</div>
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
  );
}
