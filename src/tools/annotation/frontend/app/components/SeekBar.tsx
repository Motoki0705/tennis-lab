"use client";

import React from "react";

import type { VideoMeta } from "../shared/types";

type SeekBarProps = {
  meta: VideoMeta | null;
  globalFrameIdx: number;
  seekPreviewIdx: number | null;
  isSeeking: boolean;
  onSeekStart: () => void;
  onSeekPreviewChange: (idx: number) => void;
  onSeekCommit: (idx: number) => void;
};

export default function SeekBar({
  meta,
  globalFrameIdx,
  seekPreviewIdx,
  isSeeking,
  onSeekStart,
  onSeekPreviewChange,
  onSeekCommit
}: SeekBarProps) {
  const previewValue = seekPreviewIdx ?? globalFrameIdx;
  const max = meta ? meta.frame_count - 1 : 0;

  return (
    <div className="row">
      <label>seek</label>
      <input
        type="range"
        min={0}
        max={max}
        value={previewValue}
        onMouseDown={onSeekStart}
        onTouchStart={onSeekStart}
        onChange={(e) => onSeekPreviewChange(Number(e.target.value))}
        onMouseUp={() => onSeekCommit(previewValue)}
        onTouchEnd={() => onSeekCommit(previewValue)}
      />
      <div className="small">
        {previewValue}
        {meta ? ` / ${max}` : ""}
        {isSeeking ? " (seeking)" : ""}
      </div>
    </div>
  );
}
