"""Dataset for cached DINOv3 patch embeddings and target heatmaps."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

_AUG_RE = re.compile(r"^(?P<clip>.+?)(?:_aug(?P<aug>\d+))?$")


@dataclass(frozen=True)
class PatchEmbeddingSample:
    """Single windowed sample from a clip (optionally with augmentation)."""

    embedding_path: Path | None
    heatmap_path: Path | None
    match: str
    clip: str
    aug_idx: int | None
    start_idx: int


def _resolve_matches(root_dir: Path, matches: Iterable[str]) -> list[str]:
    match_list = list(matches)
    if match_list:
        return match_list
    return sorted([p.name for p in root_dir.iterdir() if p.is_dir()])


def _parse_embedding_name(filename: str) -> tuple[str, int | None] | None:
    if filename.endswith("_heatmaps.npy"):
        return None
    stem = Path(filename).stem
    match = _AUG_RE.match(stem)
    if not match:
        return None
    clip = match.group("clip")
    aug_raw = match.group("aug")
    aug_idx = int(aug_raw) if aug_raw is not None else None
    return clip, aug_idx


class PatchEmbeddingsDataset(Dataset):
    """Dataset for cached patch embeddings and heatmaps."""

    def __init__(
        self,
        root_dir: str | Path,
        embeddings_dir: str | Path | None = None,
        heatmaps_dir: str | Path | None = None,
        matches: Sequence[str] = (),
        include_embeddings: bool = True,
        include_heatmaps: bool = True,
        frames_in: int = 8,
        frames_out: int = 1,
        step: int = 1,
    ) -> None:
        if not (include_embeddings or include_heatmaps):
            raise ValueError("include_embeddings or include_heatmaps must be true.")
        if frames_out > frames_in:
            raise ValueError("frames_out cannot exceed frames_in")
        if frames_in < 1:
            raise ValueError("frames_in must be >= 1")
        if step < 1:
            raise ValueError("step must be >= 1")

        self.root_dir = Path(root_dir)
        self.embeddings_dir = (
            Path(embeddings_dir) if embeddings_dir is not None else self.root_dir / "patch_embeddings"
        )
        self.heatmaps_dir = (
            Path(heatmaps_dir) if heatmaps_dir is not None else self.embeddings_dir
        )
        self._memmap_cache: dict[Path, np.ndarray] = {}
        self.include_embeddings = include_embeddings
        self.include_heatmaps = include_heatmaps
        self.frames_in = int(frames_in)
        self.frames_out = int(frames_out)
        self.step = int(step)
        self._meta_embeddings: dict[str, int] = {}
        self._meta_heatmaps: dict[str, int] = {}
        self._load_meta()

        match_list = _resolve_matches(self.embeddings_dir, matches)
        self.samples = self._build_index(match_list)
        if not self.samples:
            raise RuntimeError(f"No patch embedding samples found under {self.embeddings_dir}")

    def _build_index(self, matches: Sequence[str]) -> list[PatchEmbeddingSample]:
        samples: list[PatchEmbeddingSample] = []
        for match in matches:
            match_dir = self.embeddings_dir / match
            if not match_dir.exists():
                LOGGER.warning("Match directory missing, skipping: %s", match_dir)
                continue
            for entry in sorted(match_dir.iterdir()):
                if not entry.is_file() or entry.suffix.lower() != ".npy":
                    continue
                parsed = _parse_embedding_name(entry.name)
                if parsed is None:
                    continue
                clip, aug_idx = parsed
                embedding_path = entry if self.include_embeddings else None
                heatmap_path = None
                if self.include_heatmaps:
                    heatmap_path = self.heatmaps_dir / match / f"{clip}_heatmaps.npy"
                    if not heatmap_path.exists():
                        raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")
                length = self._resolve_sequence_length(embedding_path, heatmap_path)
                if length < self.frames_in:
                    continue
                max_start = length - self.frames_in
                for start_idx in range(0, max_start + 1, self.step):
                    samples.append(
                        PatchEmbeddingSample(
                            embedding_path=embedding_path,
                            heatmap_path=heatmap_path,
                            match=match,
                            clip=clip,
                            aug_idx=aug_idx,
                            start_idx=start_idx,
                        )
                    )
        return samples

    def _load_meta(self) -> None:
        meta_path = self.embeddings_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse meta.json: %s", exc)
            return
        self._meta_embeddings = {
            str(k): int(v) for k, v in payload.get("embeddings", {}).items()
        }
        self._meta_heatmaps = {
            str(k): int(v) for k, v in payload.get("heatmaps", {}).items()
        }

    def _meta_length_for(self, path: Path, *, kind: str) -> int | None:
        rel = str(path.relative_to(self.embeddings_dir))
        if kind == "embeddings":
            return self._meta_embeddings.get(rel)
        return self._meta_heatmaps.get(rel)

    def _resolve_sequence_length(
        self, embedding_path: Path | None, heatmap_path: Path | None
    ) -> int:
        length = None
        if embedding_path is not None:
            meta_len = self._meta_length_for(embedding_path, kind="embeddings")
            if meta_len is not None:
                length = meta_len
            else:
                length = self._length_from_file(embedding_path, expected_dims=3, kind="embeddings")
        if heatmap_path is not None:
            meta_len = self._meta_length_for(heatmap_path, kind="heatmaps")
            if meta_len is not None:
                hm_len = meta_len
            else:
                hm_len = self._length_from_file(heatmap_path, expected_dims=3, kind="heatmaps")
            if length is None:
                length = hm_len
            elif hm_len != length:
                raise RuntimeError(
                    f"Embedding/heatmap length mismatch: {embedding_path} vs {heatmap_path}"
                )
        if length is None:
            raise RuntimeError("Unable to resolve sequence length for sample.")
        return length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        output: dict[str, object] = {
            "match": sample.match,
            "clip": sample.clip,
            "aug_idx": sample.aug_idx,
            "start_idx": sample.start_idx,
        }
        if self.include_embeddings:
            if sample.embedding_path is None:
                raise RuntimeError("Embedding path missing for sample.")
            start = sample.start_idx
            end = start + self.frames_in
            embeddings = self._load_memmap(sample.embedding_path)
            window = embeddings[start:end]
            output["embeddings"] = torch.from_numpy(np.ascontiguousarray(window))
        if self.include_heatmaps:
            if sample.heatmap_path is None:
                raise RuntimeError("Heatmap path missing for sample.")
            start = sample.start_idx
            end = start + self.frames_in
            heatmaps = self._load_memmap(sample.heatmap_path)
            window = heatmaps[start:end]
            output["target_heatmaps"] = torch.from_numpy(
                np.ascontiguousarray(window[-self.frames_out :])
            )
        return output

    def _load_memmap(self, path: Path) -> np.ndarray:
        mmap = self._memmap_cache.get(path)
        if mmap is None:
            mmap = np.load(path, mmap_mode="r")
            self._memmap_cache[path] = mmap
        return mmap

    def _length_from_file(self, path: Path, expected_dims: int, *, kind: str) -> int:
        mmap = np.load(path, mmap_mode="r")
        if mmap.ndim != expected_dims:
            raise RuntimeError(
                f"Expected {kind} {expected_dims}D, got {tuple(mmap.shape)}"
            )
        return int(mmap.shape[0])

    def sample_aug_indices(self) -> list[int | None]:
        return [s.aug_idx for s in self.samples]
