"use client";

import React from "react";

import type { CourtFrameAnnotation, VideoMeta } from "../shared/types";
import { clamp } from "../shared/utils";

type CourtPanelProps = {
  meta: VideoMeta | null;
  courtFrameIdx: number;
  courtAnn: CourtFrameAnnotation | null;
  kpNames: string[];
  activeKp: number;
  onCourtFrameIdxChange: (idx: number) => void;
  onActiveKpChange: (idx: number) => void;
  onSaveCourt: () => void;
  onResetCourt: () => void;
  onExportCourt: () => void;
  onRunHomography: () => void;
};

export default function CourtPanel({
  meta,
  courtFrameIdx,
  courtAnn,
  kpNames,
  activeKp,
  onCourtFrameIdxChange,
  onActiveKpChange,
  onSaveCourt,
  onResetCourt,
  onExportCourt,
  onRunHomography
}: CourtPanelProps) {
  const listNames =
    kpNames.length > 0
      ? kpNames
      : Array.from({ length: 20 }, (_, i) => `kp_${i}`);

  return (
    <>
      <div className="row">
        <label>frame idx</label>
        <input
          type="number"
          value={courtFrameIdx}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (!meta) return;
            onCourtFrameIdxChange(clamp(v, 0, meta.frame_count - 1));
          }}
        />
        <button className="primary" onClick={onSaveCourt} disabled={!courtAnn}>
          Save
        </button>
        <button onClick={onResetCourt} disabled={!courtAnn}>
          Reset
        </button>
        <button onClick={onExportCourt}>Export Court</button>
      </div>

      <div className="row">
        <button onClick={onRunHomography} disabled={!courtAnn}>
          Homography fill
        </button>
      </div>

      <div className="small">
        Select a keypoint, then click to place. Drag existing points to move.
      </div>
      <div className="small">
        keys: Tab/Shift+Tab next/prev kp, N next unset, Backspace clear, S save, E
        export
      </div>

      <div className="kpList">
        {listNames.map((name, i) => {
          const kp = courtAnn?.keypoints?.[i];
          const status = kp?.visibility ? kp.source : "—";
          return (
            <div
              key={i}
              className={`kpItem ${i === activeKp ? "active" : ""}`}
              onClick={() => onActiveKpChange(i)}
            >
              <div className="small">
                {i}: {name}
              </div>
              <div className="small kpStatus">{status}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}
