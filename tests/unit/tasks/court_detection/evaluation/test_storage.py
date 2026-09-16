"""Resume, integrity, and fail-closed contracts for benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation import storage
from src.tasks.court_detection.evaluation.contracts import (
    MANIFEST_SCHEMA,
    KeypointPrediction,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.storage import (
    BenchmarkArtifactError,
    PredictionStore,
    fingerprint,
    write_manifest_once,
)
from src.tasks.court_detection.evaluation.storage import (
    _safe_sample_filename as _sample_filename,
)


def _ref(sample_id: str, *, domain: str = "real_validation") -> SampleRef:
    return SampleRef(
        domain=domain,  # type: ignore[arg-type]
        sample_id=sample_id,
        scene_id="B00" if domain == "synthetic_test" else "tennis_court_detector",
        trajectory_group_id="group-a" if domain == "synthetic_test" else None,
        split="test" if domain == "synthetic_test" else "val",
        source_target_sha256="b" * 64,
        width=960,
        height=540,
    )


def _prediction(model: str = "ours") -> ModelPrediction:
    points: NDArray[np.float64] = np.zeros((14, 2), dtype=np.float64)
    points[:, 0] = np.arange(14, dtype=np.float64)
    return ModelPrediction(
        model=model,  # type: ignore[arg-type]
        keypoints=KeypointPrediction(
            keypoints_xy=points,
            scores=np.ones(14, dtype=np.float64),
            valid=np.ones(14, dtype=bool),
        ),
        elapsed_seconds=1.5,
        extras={"preprocessing": "test"},
    )


def _store(
    root: Path, *, domain: str = "real_validation", model: str = "ours"
) -> PredictionStore:
    return PredictionStore(
        root=root,
        domain=domain,  # type: ignore[arg-type]
        model=model,
        manifest_fingerprint="f" * 64,
        model_fingerprint="1" * 64,
    )


def _rewrite_sample_entry(
    store: PredictionStore,
    sample_id: str,
    *,
    drop: tuple[str, ...] = (),
    **overrides: object,
) -> None:
    """Rewrite one index entry in place to simulate a tampered cache entry."""
    lines = store.index_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("type") == "sample" and entry.get("sample_id") == sample_id:
            for key in drop:
                entry.pop(key, None)
            entry.update(overrides)
        rewritten.append(json.dumps(entry))
    store.index_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _manifest(root: Path, *, samples: tuple[SampleRef, ...] | None = None):
    return write_manifest_once(
        root / "manifest.json",
        settings_fingerprint="f" * 64,
        quality={"pck_fractions": [0.02]},
        selection={"seed": 7, "max_samples_per_domain": 2},
        domains={"real_validation": {"display_name": "real_validation"}},
        samples=samples if samples is not None else (_ref("s1"), _ref("s2")),
    )


def test_manifest_is_reused_when_identical_and_rejected_when_it_differs(
    tmp_path: Path,
) -> None:
    first = _manifest(tmp_path)

    assert _manifest(tmp_path).samples == first.samples

    with pytest.raises(BenchmarkArtifactError, match="disagrees with this run"):
        write_manifest_once(
            tmp_path / "manifest.json",
            settings_fingerprint="f" * 64,
            quality={"pck_fractions": [0.02]},
            selection={"seed": 7, "max_samples_per_domain": 2},
            domains={"real_validation": {"display_name": "real_validation"}},
            samples=(_ref("s1"), _ref("s3")),
        )


def test_completed_returns_previously_recorded_samples_and_their_timing(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())

    completed = store.completed()

    assert completed == {"s1": 1.5}
    restored = store.load("s1")
    np.testing.assert_array_equal(restored.keypoints.valid, np.ones(14, dtype=bool))
    assert restored.keypoints.keypoints_xy[3, 0] == 3.0
    assert restored.extras == {"preprocessing": "test"}
    assert restored.elapsed_seconds == 1.5


def test_a_resumed_store_returns_the_recorded_per_sample_extras(
    tmp_path: Path,
) -> None:
    """A resume on a fresh process must reproduce the original extras."""
    first = _store(tmp_path)
    first.ensure_header()
    prediction = ModelPrediction(
        model="ours",
        keypoints=KeypointPrediction(
            keypoints_xy=np.zeros((14, 2), dtype=np.float64),
            scores=np.ones(14, dtype=np.float64),
            valid=np.ones(14, dtype=bool),
        ),
        elapsed_seconds=2.25,
        extras={"preprocessing": "training_geometry", "resize_side": 512},
    )
    first.record("s1", prediction)

    reopened = _store(tmp_path)
    assert reopened.completed() == {"s1": 2.25}

    restored = reopened.load("s1")

    assert restored.extras == prediction.extras
    assert restored.elapsed_seconds == 2.25


def test_a_stored_entry_without_extras_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())
    _rewrite_sample_entry(store, "s1", drop=("extras",))

    with pytest.raises(BenchmarkArtifactError, match="no mapping of inference extras"):
        store.load("s1")


def test_a_stored_entry_with_non_mapping_extras_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())
    _rewrite_sample_entry(store, "s1", extras="not-a-mapping")

    with pytest.raises(BenchmarkArtifactError, match="no mapping of inference extras"):
        store.load("s1")


def test_loading_a_sample_that_is_not_in_the_index_fails_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.ensure_header()

    with pytest.raises(BenchmarkArtifactError, match="no entry for sample"):
        store.load("missing-sample")


def test_a_different_model_fingerprint_stops_the_resume(tmp_path: Path) -> None:
    _store(tmp_path).ensure_header()

    other = PredictionStore(
        root=tmp_path,
        domain="real_validation",
        model="ours",
        manifest_fingerprint="f" * 64,
        model_fingerprint="2" * 64,
    )

    with pytest.raises(BenchmarkArtifactError, match="model provenance"):
        other.completed()


def test_a_different_manifest_fingerprint_stops_the_resume(tmp_path: Path) -> None:
    _store(tmp_path).ensure_header()

    other = PredictionStore(
        root=tmp_path,
        domain="real_validation",
        model="ours",
        manifest_fingerprint="9" * 64,
        model_fingerprint="1" * 64,
    )

    with pytest.raises(BenchmarkArtifactError, match="sample manifest"):
        other.completed()


def test_a_corrupted_cached_array_stops_the_resume(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())
    store.sample_path("s1").write_bytes(b"not an npz")

    with pytest.raises(BenchmarkArtifactError, match="checksum mismatch"):
        store.completed()


def test_a_deleted_cached_array_stops_the_resume(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())
    store.sample_path("s1").unlink()

    with pytest.raises(BenchmarkArtifactError, match="missing array file"):
        store.completed()


def test_a_repeated_sample_entry_stops_the_resume(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    store.record("s1", _prediction())
    with store.index_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "sample",
                    "sample_id": "s1",
                    "npz": "s1.npz",
                    "sha256": "0" * 64,
                    "elapsed_seconds": 0.0,
                }
            )
            + "\n"
        )

    with pytest.raises(BenchmarkArtifactError, match="repeats sample"):
        store.completed()


def test_synthetic_sample_ids_with_a_colon_are_stored_under_a_safe_name(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, domain="synthetic_test")
    store.ensure_header()
    store.record("B00:court-sample-000001", _prediction())

    assert store.sample_path("B00:court-sample-000001").name == (
        "sample-B00%3Acourt-sample-000001.npz"
    )
    assert store.completed() == {"B00:court-sample-000001": 1.5}


def test_an_unsafe_sample_id_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="portable file name"):
        store.sample_path("../escape")


def test_the_header_is_written_once_and_reused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ensure_header()
    first = store.index_path.read_text(encoding="utf-8")
    store.ensure_header()

    assert store.index_path.read_text(encoding="utf-8") == first
    header = json.loads(first.splitlines()[0])
    assert header["schema"] == MANIFEST_SCHEMA.replace("manifest", "predictions")


def test_fingerprints_are_order_independent_for_mappings(tmp_path: Path) -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


# --- sample-id -> file name encoding contract -------------------------------

_LEADING_HYPHEN_IDS = ("-0M6ixK7aIU_1600", "-5zNAhwRoPE_2650", "-_5ljBK4HnI_500")


@pytest.mark.parametrize("sample_id", _LEADING_HYPHEN_IDS)
def test_youtube_ids_may_begin_with_a_hyphen(sample_id: str) -> None:
    """The production crash: real ids legitimately start with ``-``."""
    stem = _sample_filename(sample_id)

    assert stem == f"sample-{sample_id}"
    assert stem.startswith("sample--")


def test_synthetic_ids_are_encoded_rather_than_substituted() -> None:
    assert _sample_filename("B02:court-sample-001144") == (
        "sample-B02%3Acourt-sample-001144"
    )


def test_colon_and_literal_double_underscore_no_longer_collide() -> None:
    """``a:b`` and a literal ``a__b`` used to share one file name."""
    colon = _sample_filename("a:b")
    literal = _sample_filename("a__b")

    assert colon != literal
    assert colon == "sample-a%3Ab"
    assert literal == "sample-a__b"


def test_synthetic_ids_do_not_collide_with_a_literal_lookalike() -> None:
    encoded = _sample_filename("B02:court-sample-001144")
    lookalike = _sample_filename("B02__court-sample-001144")

    assert encoded != lookalike


def test_a_literal_percent_is_escaped_so_the_mapping_stays_injective() -> None:
    assert _sample_filename("100%") == "sample-100%25"
    assert _sample_filename("100%3A") != _sample_filename("100:")
    assert _sample_filename("100%3A") == "sample-100%253A"


def test_the_encoding_is_injective_over_a_mixed_corpus() -> None:
    corpus = (
        *_LEADING_HYPHEN_IDS,
        "B02:court-sample-001144",
        "B02__court-sample-001144",
        "a:b",
        "a__b",
        "a%3Ab",
        "plain_id",
        "-",
        "a.b",
        "a b",
        "A-B_c.d~e",
        "\u65e5\u672c\u8a9eID",
        "x" * 200,
    )

    stems = [_sample_filename(sample_id) for sample_id in corpus]

    assert len(set(stems)) == len(corpus)


@pytest.mark.parametrize("sample_id", ["a/b", "../escape", "..\\escape", "a\x00b"])
def test_path_separators_and_nul_are_refused(sample_id: str) -> None:
    with pytest.raises(ValueError, match="path separator or NUL|portable file name"):
        _sample_filename(sample_id)


@pytest.mark.parametrize("sample_id", ["", ".", "..", " padded ", "\ttrim"])
def test_empty_dotty_or_untrimmed_ids_are_refused(sample_id: str) -> None:
    with pytest.raises(ValueError, match="portable file name"):
        _sample_filename(sample_id)


def test_a_non_string_id_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        _sample_filename(1234)  # type: ignore[arg-type]


def test_an_over_long_id_fails_closed_instead_of_being_truncated() -> None:
    with pytest.raises(ValueError, match="file name limit"):
        _sample_filename("x" * 400)


def test_encoded_file_names_are_single_portable_components() -> None:
    for sample_id in (*_LEADING_HYPHEN_IDS, "B02:court-sample-001144", "a b", "\u65e5"):
        name = f"{_sample_filename(sample_id)}.npz"

        assert "/" not in name and "\\" not in name
        assert name not in {".", ".."}
        assert not name.startswith(".")
        assert len(name.encode("utf-8")) <= 255


def test_distinct_ids_are_written_to_distinct_files_on_disk(tmp_path: Path) -> None:
    store = _store(tmp_path, domain="synthetic_test")
    store.ensure_header()
    for sample_id in ("a:b", "a__b", "-0M6ixK7aIU_1600", "B02:court-sample-001144"):
        store.record(sample_id, _prediction())

    assert store.completed().keys() == {
        "a:b",
        "a__b",
        "-0M6ixK7aIU_1600",
        "B02:court-sample-001144",
    }
    assert len(list(store.model_dir.glob("*.npz"))) == 4


def test_a_non_injective_encoder_is_still_caught_when_reading_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index re-checks the invariant, so a future encoder cannot collide."""
    store = _store(tmp_path)
    store.ensure_header()

    # Simulate a future encoder that maps two distinct ids onto one file name.
    monkeypatch.setattr(
        storage, "_safe_sample_filename", lambda sample_id: "sample-collapsed"
    )
    collapsed = store.model_dir / "sample-collapsed.npz"
    with collapsed.open("wb") as stream:
        np.savez_compressed(
            stream,
            keypoints_xy=np.zeros((14, 2), dtype=np.float64),
            scores=np.ones(14, dtype=np.float64),
            valid=np.ones(14, dtype=np.uint8),
        )
    digest = storage.file_sha256(collapsed)
    with store.index_path.open("a", encoding="utf-8") as stream:
        for sample_id in ("a:b", "a__b"):
            stream.write(
                json.dumps(
                    {
                        "type": "sample",
                        "sample_id": sample_id,
                        "npz": "sample-collapsed.npz",
                        "sha256": digest,
                        "elapsed_seconds": 0.0,
                        "extras": {"preprocessing": "test"},
                    }
                )
                + "\n"
            )

    with pytest.raises(BenchmarkArtifactError, match="share the cached file name"):
        store.completed()
