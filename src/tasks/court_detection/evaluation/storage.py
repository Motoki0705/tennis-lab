"""Replayable run artifacts: manifest, prediction cache, and provenance.

The manifest is written once and never silently rewritten, both models read
the same file, and a resumed run refuses to continue when any stored
fingerprint, index line, or cached array disagrees with the requested run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import quote

import numpy as np
import torch

from src.tasks.court_detection.evaluation.contracts import (
    MANIFEST_SCHEMA,
    PREDICTION_SCHEMA,
    DomainName,
    KeypointPrediction,
    ModelPrediction,
    SampleRef,
)

# Sample ids are opaque identities, not file names.  Real YouTube-derived ids can
# legitimately begin with ``-`` and Synthetic Court ids contain ``:``, so the
# cache encodes every id instead of accepting only an alphanumeric subset.
_SAMPLE_FILENAME_PREFIX = "sample-"
# A file name component must stay within the common 255-byte limit.  ``.npz`` is
# reserved before checking the encoded id.
_SAMPLE_FILENAME_SUFFIX = ".npz"
_MAX_FILENAME_BYTES = 255
_FORBIDDEN_ID_CHARACTERS = ("/", "\\", "\x00")
_PREDICTION_ARRAYS = frozenset({"keypoints_xy", "scores", "valid"})


class BenchmarkArtifactError(RuntimeError):
    """Raised when a stored artifact disagrees with the requested run."""


def canonical_json(value: object) -> str:
    """Render deterministic JSON that rejects non-finite numbers."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    """Publish a JSON document through a same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_sample_filename(sample_id: str) -> str:
    """Map any accepted sample id to one portable, collision-free file stem.

    Contract:

    * The id must be a non-empty ``str`` that is neither ``.`` nor ``..`` and
      contains none of ``/``, ``\\``, or NUL.  These are never part of a valid
      benchmark identity and are rejected outright, so path traversal cannot be
      expressed at all.  The leading ``-`` of YouTube-derived ids, ``:`` in
      Synthetic Court ids, underscores, spaces, and non-ASCII text are all
      legitimate and are encoded rather than refused.
    * The stem is ``sample-`` plus the id percent-encoded with an empty ``safe``
      set, so only ``A-Za-z0-9_.-~`` stay literal and every other byte becomes
      ``%XX``.  A literal ``%`` therefore always encodes to ``%25`` and the
      mapping is injective: distinct ids can never share a file name.  (The
      previous ``:`` -> ``__`` substitution was ambiguous because ``a:b`` and a
      literal ``a__b`` both mapped to ``a__b``.)
    * The fixed prefix keeps the name from starting with ``.`` or ``-``.
    * The encoded stem must fit the 255-byte component limit; an over-long id
      fails closed instead of being silently truncated into a different sample.
    """
    if type(sample_id) is not str:
        raise TypeError(f"Sample id must be a string, got {type(sample_id).__name__}.")
    if not sample_id or sample_id != sample_id.strip():
        raise ValueError(
            f"Sample id {sample_id!r} must be non-empty and trimmed to be used "
            "as a portable file name."
        )
    if sample_id in {".", ".."}:
        raise ValueError(
            f"Sample id {sample_id!r} cannot be used as a portable file name."
        )
    forbidden = [char for char in _FORBIDDEN_ID_CHARACTERS if char in sample_id]
    if forbidden:
        raise ValueError(
            f"Sample id {sample_id!r} cannot be used as a portable file name: it "
            "contains a path separator or NUL."
        )
    encoded = quote(sample_id, safe="")
    stem = f"{_SAMPLE_FILENAME_PREFIX}{encoded}"
    name_bytes = len(stem.encode("utf-8")) + len(_SAMPLE_FILENAME_SUFFIX)
    if name_bytes > _MAX_FILENAME_BYTES:
        raise ValueError(
            f"Sample id {sample_id!r} encodes to {name_bytes} bytes, which "
            f"exceeds the {_MAX_FILENAME_BYTES}-byte file name limit."
        )
    return stem


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """The manifest exactly as published on disk."""

    schema: str
    settings_fingerprint: str
    quality: Mapping[str, object]
    selection: Mapping[str, object]
    domains: Mapping[str, Mapping[str, object]]
    samples: tuple[SampleRef, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "settings_fingerprint": self.settings_fingerprint,
            "quality": dict(self.quality),
            "selection": dict(self.selection),
            "domains": {key: dict(value) for key, value in self.domains.items()},
            "samples": [sample.to_json() for sample in self.samples],
        }

    def samples_for(self, domain: DomainName) -> tuple[SampleRef, ...]:
        return tuple(sample for sample in self.samples if sample.domain == domain)


def write_manifest_once(
    path: Path,
    *,
    settings_fingerprint: str,
    quality: Mapping[str, object],
    selection: Mapping[str, object],
    domains: Mapping[str, Mapping[str, object]],
    samples: Sequence[SampleRef],
) -> ManifestDocument:
    """Create the manifest, or refuse to continue if one already disagrees."""
    document = ManifestDocument(
        schema=MANIFEST_SCHEMA,
        settings_fingerprint=settings_fingerprint,
        quality=dict(quality),
        selection=dict(selection),
        domains={key: dict(value) for key, value in domains.items()},
        samples=tuple(samples),
    )
    payload = document.to_json()
    if path.exists():
        observed = read_json(path)
        if canonical_json(observed) != canonical_json(payload):
            raise BenchmarkArtifactError(
                "The stored benchmark manifest disagrees with this run's settings; "
                f"refusing to mix sample sets ({path})."
            )
        return document
    write_json_atomic(path, payload)
    return document


def read_manifest(path: Path) -> ManifestDocument:
    payload = read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != MANIFEST_SCHEMA:
        raise BenchmarkArtifactError(f"Unsupported benchmark manifest: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise BenchmarkArtifactError("Benchmark manifest must list its samples.")
    return ManifestDocument(
        schema=MANIFEST_SCHEMA,
        settings_fingerprint=str(payload["settings_fingerprint"]),
        quality=cast("Mapping[str, object]", payload["quality"]),
        selection=cast("Mapping[str, object]", payload["selection"]),
        domains=cast("Mapping[str, Mapping[str, object]]", payload["domains"]),
        samples=tuple(SampleRef.from_json(entry) for entry in samples),
    )


def read_json(path: Path) -> object:
    if not path.is_file():
        raise BenchmarkArtifactError(f"Required benchmark artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BenchmarkArtifactError(f"Malformed JSON artifact: {path}") from error


@dataclass(frozen=True, slots=True)
class PredictionStore:
    """One model's durable, resumable per-sample prediction cache.

    Each model owns its own index and array directory, so adding a model to an
    existing run can never invalidate the other model's cached predictions.
    """

    root: Path
    domain: DomainName
    model: str
    manifest_fingerprint: str
    model_fingerprint: str

    @property
    def model_dir(self) -> Path:
        return self.root / "predictions" / self.domain / self.model

    @property
    def index_path(self) -> Path:
        return self.model_dir / "index.jsonl"

    def sample_path(self, sample_id: str) -> Path:
        return self.model_dir / f"{_safe_sample_filename(sample_id)}.npz"

    def completed(self) -> dict[str, float]:
        """Return already stored sample ids and their elapsed seconds."""
        if not self.index_path.is_file():
            return {}
        header: Mapping[str, object] | None = None
        completed: dict[str, float] = {}
        filenames: dict[str, str] = {}
        for line_number, line in enumerate(
            self.index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, Mapping):
                raise BenchmarkArtifactError(
                    f"Prediction index line {line_number} is not an object."
                )
            if header is None:
                if entry.get("type") != "header":
                    raise BenchmarkArtifactError(
                        "Prediction index must start with its header line."
                    )
                self._validate_header(entry)
                header = entry
                continue
            if entry.get("type") != "sample":
                raise BenchmarkArtifactError(
                    f"Prediction index line {line_number} is not a sample entry."
                )
            sample_id = entry.get("sample_id")
            if not isinstance(sample_id, str):
                raise BenchmarkArtifactError(
                    f"Prediction index line {line_number} has no sample id."
                )
            if sample_id in completed:
                raise BenchmarkArtifactError(
                    f"Prediction index repeats sample {sample_id!r}."
                )
            filename = _safe_sample_filename(sample_id)
            previous = filenames.setdefault(filename, sample_id)
            if previous != sample_id:
                raise BenchmarkArtifactError(
                    f"Samples {previous!r} and {sample_id!r} share the cached "
                    f"file name {filename!r}."
                )
            path = self._resolve_entry_path(entry, sample_id=sample_id)
            self._verify_cached(path, entry)
            completed[sample_id] = float(cast("float", entry["elapsed_seconds"]))
        if header is None:
            raise BenchmarkArtifactError(
                "Prediction index exists without its header line."
            )
        return completed

    def ensure_header(self) -> None:
        if self.index_path.exists():
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._append(
            {
                "type": "header",
                "schema": PREDICTION_SCHEMA,
                "domain": self.domain,
                "model": self.model,
                "manifest_fingerprint": self.manifest_fingerprint,
                "model_fingerprint": self.model_fingerprint,
            }
        )

    def record(self, sample_id: str, prediction: ModelPrediction) -> None:
        """Persist one prediction, then append its index entry."""
        if prediction.model != self.model:
            raise BenchmarkArtifactError(
                f"Prediction store for {self.model!r} received "
                f"{prediction.model!r} predictions."
            )
        path = self.sample_path(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "keypoints_xy": prediction.keypoints.keypoints_xy.astype(np.float64),
            "scores": prediction.keypoints.scores.astype(np.float64),
            "valid": prediction.keypoints.valid.astype(np.uint8),
        }
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._append(
            {
                "type": "sample",
                "sample_id": sample_id,
                "npz": str(path.relative_to(self.model_dir)),
                "sha256": file_sha256(path),
                "elapsed_seconds": float(prediction.elapsed_seconds),
                "extras": dict(prediction.extras),
            }
        )

    def load(self, sample_id: str) -> ModelPrediction:
        """Read one cached prediction previously verified by :meth:`completed`."""
        entry = self._sample_entry(sample_id)
        path = self.sample_path(sample_id)
        if not path.is_file():
            raise BenchmarkArtifactError(
                f"Cached prediction is missing for {sample_id!r}: {path}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _PREDICTION_ARRAYS:
                raise BenchmarkArtifactError(
                    f"Cached prediction arrays changed for {sample_id!r}."
                )
            keypoints = np.asarray(archive["keypoints_xy"], dtype=np.float64)
            scores = np.asarray(archive["scores"], dtype=np.float64)
            valid = np.asarray(archive["valid"], dtype=np.uint8).astype(bool)
        extras = entry.get("extras")
        if not isinstance(extras, Mapping):
            raise BenchmarkArtifactError(
                f"Cached prediction entry for {sample_id!r} has no mapping of "
                "inference extras; refusing to resume without them."
            )
        try:
            canonical_json(dict(extras))
        except (TypeError, ValueError) as error:
            raise BenchmarkArtifactError(
                f"Cached prediction extras for {sample_id!r} are not valid JSON."
            ) from error
        return ModelPrediction(
            model=cast("object", self.model),  # type: ignore[arg-type]
            keypoints=KeypointPrediction(
                keypoints_xy=keypoints, scores=scores, valid=valid
            ),
            elapsed_seconds=float(cast("float", entry["elapsed_seconds"])),
            extras=dict(extras),
        )

    def entries(self) -> dict[str, Mapping[str, object]]:
        if not self.index_path.is_file():
            return {}
        entries: dict[str, Mapping[str, object]] = {}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "sample":
                entries[str(entry["sample_id"])] = entry
        return entries

    def _sample_entry(self, sample_id: str) -> Mapping[str, object]:
        entry = self.entries().get(sample_id)
        if entry is None:
            raise BenchmarkArtifactError(
                f"Prediction index has no entry for sample {sample_id!r}."
            )
        return entry

    def _append(self, entry: Mapping[str, object]) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        line = canonical_json(entry)
        with self.index_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _validate_header(self, header: Mapping[str, object]) -> None:
        if header.get("schema") != PREDICTION_SCHEMA:
            raise BenchmarkArtifactError("Prediction index schema changed.")
        if header.get("domain") != self.domain:
            raise BenchmarkArtifactError("Prediction index domain changed.")
        if header.get("model") != self.model:
            raise BenchmarkArtifactError(
                f"Prediction index belongs to model {header.get('model')!r}, "
                f"not {self.model!r}."
            )
        if header.get("manifest_fingerprint") != self.manifest_fingerprint:
            raise BenchmarkArtifactError(
                "Prediction index was produced from a different sample manifest."
            )
        if header.get("model_fingerprint") != self.model_fingerprint:
            raise BenchmarkArtifactError(
                "Stored predictions were produced with different model "
                "provenance; refusing to resume."
            )

    def _resolve_entry_path(
        self, entry: Mapping[str, object], *, sample_id: str
    ) -> Path:
        relative = entry.get("npz")
        if not isinstance(relative, str) or not relative:
            raise BenchmarkArtifactError(
                f"Prediction index entry for {sample_id!r} has no array path."
            )
        candidate = (self.model_dir / relative).resolve(strict=False)
        if not candidate.is_relative_to(self.model_dir.resolve(strict=False)):
            raise BenchmarkArtifactError(
                f"Prediction index entry for {sample_id!r} escapes its model directory."
            )
        expected = self.sample_path(sample_id).resolve(strict=False)
        if candidate != expected:
            raise BenchmarkArtifactError(
                f"Prediction index entry for {sample_id!r} points at an unexpected "
                "file."
            )
        return candidate

    def _verify_cached(self, path: Path, entry: Mapping[str, object]) -> None:
        if not path.is_file():
            raise BenchmarkArtifactError(
                f"Prediction index references a missing array file: {path}"
            )
        expected = entry.get("sha256")
        if not isinstance(expected, str):
            raise BenchmarkArtifactError("Prediction index entry has no checksum.")
        if file_sha256(path) != expected:
            raise BenchmarkArtifactError(f"Cached prediction checksum mismatch: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_provenance(
    *,
    root: Path | None = None,
    code_root: Path | None = None,
) -> dict[str, object]:
    """Capture the runtime facts a reader needs to reproduce a run."""
    payload: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cv2": _module_version("cv2"),
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": torch.cuda.is_available(),
        "pid": os.getpid(),
    }
    if root is not None:
        payload["repo_root"] = str(root)
        payload["repo_commit"] = _repo_commit(root)
    if code_root is not None:
        payload["code_root"] = str(code_root)
        payload["code_commit"] = _repo_commit(code_root)
    return payload


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:  # pragma: no cover - cv2 presence is validated upstream
        return None
    return str(getattr(module, "__version__", "unknown"))


def _repo_commit(root: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def timed() -> float:
    return time.perf_counter()


__all__ = [
    "BenchmarkArtifactError",
    "ManifestDocument",
    "PredictionStore",
    "canonical_json",
    "environment_provenance",
    "file_sha256",
    "fingerprint",
    "read_json",
    "read_manifest",
    "timed",
    "write_json_atomic",
    "write_manifest_once",
]
