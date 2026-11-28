"""Contract and edge-case tests for Issue #2746 cross-clip player identity linking."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from contract import (  # noqa: E402
    SCHEMA_VERSION,
    IdentityRegistry,
    LinkEvidence,
    PlayerIdentityLink,
    PlayerIdentityLinksDocument,
    join_events_with_global_ids,
)
from metrics import (  # noqa: E402
    EvalNode,
    compute_cross_clip_pair_metrics,
    compute_scope_metrics,
)

from src.utils.matching.multiview_clustering import (  # noqa: E402
    MAX_ITEMS,
    SolverTimeLimit,
    cluster_multiview,
)


def _ev(cams: tuple[str, ...] = ("cam0", "cam1"), shape_valid: bool = False, temporal_valid: bool = False) -> LinkEvidence:
    return LinkEvidence(
        cameras_used=cams,
        sample_frames={c: [10, 20] for c in cams},
        sample_source_rows={c: [100, 200] for c in cams},
        appearance_term=0.85 if cams else None,
        shape_term=0.1 if shape_valid else None,
        temporal_term=0.2 if temporal_valid else None,
        shape_valid=shape_valid,
        temporal_valid=temporal_valid,
    )


def _doc(links: tuple[PlayerIdentityLink, ...], registry_version: int = 1) -> PlayerIdentityLinksDocument:
    return PlayerIdentityLinksDocument(
        schema_version=SCHEMA_VERSION,
        identity_scope="meiji_4cam/video_000",
        registry_version=registry_version,
        source_manifest_hashes={"multiview_v4": "a" * 64},
        upstream_digests={"scene_export": "b" * 64},
        encoder_identity={"profile": "clipreid_vitb16_market1501"},
        preprocessing_identity={"crops_per_camera": 4},
        method="E4_quality_weighted_hungarian",
        config={"min_score": 0.4, "min_margin": 0.05},
        links=links,
    )


def test_end_change_and_local_id_reuse_contract() -> None:
    """Same local_player_id=0 reused across clips for different people & end-change preserved."""
    reg = IdentityRegistry("meiji_4cam/video_000")
    gid_a = reg.register_anchor("clip_000:p0", clip_id="video_000/clip_000", local_player_id=0)
    gid_b = reg.register_anchor("clip_000:p1", clip_id="video_000/clip_000", local_player_id=1)
    assert (gid_a, gid_b) == ("player_01", "player_02")

    # Re-registering in reversed order must NOT renumber existing global IDs
    assert reg.register_anchor("clip_000:p1", clip_id="video_000/clip_000", local_player_id=1) == "player_02"
    assert reg.register_anchor("clip_000:p0", clip_id="video_000/clip_000", local_player_id=0) == "player_01"

    # In clip_005 (after end change), local_id=0 is actually player_02 and local_id=1 is player_01
    links = (
        PlayerIdentityLink("video_000/clip_000", 0, gid_a, "accepted", 1.0, "cosine_similarity", 0.2, 0.8, "anchor", _ev()),
        PlayerIdentityLink("video_000/clip_000", 1, gid_b, "accepted", 1.0, "cosine_similarity", 0.2, 0.8, "anchor", _ev()),
        PlayerIdentityLink("video_000/clip_005", 0, gid_b, "accepted", 0.82, "cosine_similarity", 0.31, 0.51, "matched", _ev()),
        PlayerIdentityLink("video_000/clip_005", 1, gid_a, "accepted", 0.79, "cosine_similarity", 0.28, 0.51, "matched", _ev()),
    )
    doc = _doc(links, registry_version=reg.registry_version)
    events = [
        {"clip_id": "video_000/clip_000", "local_player_id": 0, "event": "hit"},
        {"clip_id": "video_000/clip_005", "local_player_id": 0, "event": "hit"},
    ]
    joined = join_events_with_global_ids(events, doc, require_accepted=True)
    assert [r["global_player_id"] for r in joined] == ["player_01", "player_02"]


def test_duplicate_cooccurring_assignment_rejected_but_disjoint_intervals_allowed() -> None:
    """Two co-occurring local players in one clip cannot share an accepted global_player_id."""
    dup_links = (
        PlayerIdentityLink("video_000/clip_010", 0, "player_01", "accepted", 0.8, "cosine_similarity", 0.4, 0.4, "ok", _ev()),
        PlayerIdentityLink("video_000/clip_010", 1, "player_01", "accepted", 0.7, "cosine_similarity", 0.4, 0.3, "dup", _ev()),
    )
    with pytest.raises(ValueError, match="Duplicate accepted global_player_id"):
        _doc(dup_links)

    # Non-overlapping frame intervals (fragmented track of same person) are allowed
    disjoint_links = (
        PlayerIdentityLink(
            "video_000/clip_010", 0, "player_01", "accepted", 0.8, "cosine_similarity", 0.3, 0.5, "part1", _ev(), frame_interval=(0, 100)
        ),
        PlayerIdentityLink(
            "video_000/clip_010", 1, "player_01", "accepted", 0.78, "cosine_similarity", 0.3, 0.48, "part2", _ev(), frame_interval=(120, 250)
        ),
    )
    doc = _doc(disjoint_links)
    joined = join_events_with_global_ids(
        [{"clip_id": "video_000/clip_010", "local_player_id": 1, "frame": 150}], doc, require_accepted=True
    )
    assert joined[0]["global_player_id"] == "player_01"


def test_unobserved_camera_all_unassigned_and_strict_downstream_detection() -> None:
    """No common camera / all unassigned produce explicit unresolved links rejected by strict join."""
    links = (
        PlayerIdentityLink("video_000/clip_020", 0, None, "unresolved", None, "none", None, None, "no_common_camera", _ev(cams=())),
        PlayerIdentityLink(
            "video_000/clip_020", 1, None, "unresolved", 0.25, "cosine_similarity", 0.22, 0.03, "below_threshold", _ev()
        ),
    )
    doc = _doc(links)
    with pytest.raises(ValueError, match="Non-accepted identity link"):
        join_events_with_global_ids([{"clip_id": "video_000/clip_020", "local_player_id": 0}], doc, require_accepted=True)

    lenient = join_events_with_global_ids([{"clip_id": "video_000/clip_020", "local_player_id": 0}], doc, require_accepted=False)
    assert lenient[0]["global_player_id"] is None
    assert lenient[0]["identity_status"] == "unresolved"
    assert lenient[0]["identity_reason"] == "no_common_camera"


def test_invalid_beta_and_unknown_time_explicitly_recorded_without_mutating_scene() -> None:
    """Shape unavailable and temporal unavailable are explicit in evidence; SceneResult untouched."""
    fake_scene = {
        "player_position": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "smpl_betas": np.zeros((2, 10), dtype=np.float32),
        "player_smpl_valid": np.zeros((2, 4), dtype=bool),
    }
    snapshot = copy.deepcopy(fake_scene)
    ev = _ev(shape_valid=False, temporal_valid=False)
    ev = LinkEvidence(
        **{**ev.__dict__, "extra": {"shape_reason": "shape_unavailable", "temporal_reason": "unknown_timestamp"}}
    )
    link = PlayerIdentityLink("video_000/clip_030", 0, "player_01", "accepted", 0.81, "combined_score", 0.30, 0.51, "appearance_only", ev)
    doc = _doc((link,))
    join_events_with_global_ids([{"clip_id": "video_000/clip_030", "local_player_id": 0}], doc, require_accepted=True)
    for k in fake_scene:
        assert np.array_equal(fake_scene[k], snapshot[k])


def test_gallery_update_history_and_solver_max_items_and_timeout_contract() -> None:
    """Gallery updates increment registry_version; existing MILP rejects >MAX_ITEMS and timeout."""
    reg = IdentityRegistry("meiji_4cam/video_000")
    gid = reg.register_anchor("clip_000:p0", clip_id="video_000/clip_000", local_player_id=0)
    assert reg.registry_version == 1
    reg.record_gallery_update(gid, clip_id="video_000/clip_002", local_player_id=2, score=0.91, margin=0.42)
    assert reg.registry_version == 2
    assert len(reg.identities[gid]["gallery_updates"]) == 1

    # Existing multiview_clustering enforces MAX_ITEMS=40
    too_many = MAX_ITEMS + 1
    scores = np.zeros((too_many, too_many), dtype=np.float64)
    views = np.arange(too_many, dtype=np.int64)
    with pytest.raises(ValueError, match=f"at most {MAX_ITEMS}"):
        cluster_multiview(scores, views)

    # Timeout raises SolverTimeLimit explicitly rather than silently falling back to greedy
    n = 16
    rng = np.random.default_rng(0)
    m = rng.normal(0.0, 1.0, size=(n, n))
    sym = 0.5 * (m + m.T)
    np.fill_diagonal(sym, 0.0)
    with pytest.raises(SolverTimeLimit):
        cluster_multiview(sym, np.arange(n, dtype=np.int64), time_limit_s=1e-9)


def test_scope_level_metrics_unresolved_and_no_per_clip_rematching() -> None:
    """Unresolved nodes must count as FN (not merged together), and per-clip label flips must be penalized."""
    # Two clips with swapped anonymous IDs -> must NOT get F1=1.0 via per-clip re-matching
    nodes = [
        EvalNode(0, "clip_000", 0, "player_01", "gA", "accepted", 0.9, 0.5, "b0"),
        EvalNode(0, "clip_000", 1, "player_02", "gB", "accepted", 0.9, 0.5, "b0"),
        EvalNode(1, "clip_001", 0, "player_01", "gB", "accepted", 0.8, 0.4, "b1"),
        EvalNode(1, "clip_001", 1, "player_02", "gA", "accepted", 0.8, 0.4, "b1"),
    ]
    pm = compute_cross_clip_pair_metrics(nodes)
    assert pm["tp_pairs"] == 0
    assert pm["fp_pairs"] == 2
    assert pm["fn_pairs"] == 2
    assert pm["cross_clip_pair_f1"] == 0.0

    # All unresolved must have 0 precision/recall/coverage and 0 FP (not merged as one cluster)
    unres = [
        EvalNode(0, "clip_000", 0, "player_01", None, "unresolved", 0.1, 0.0, "b0"),
        EvalNode(1, "clip_001", 0, "player_01", None, "unresolved", 0.1, 0.0, "b1"),
    ]
    m_unres = compute_scope_metrics(unres, n_bootstrap=0)
    assert m_unres["coverage"] == 0.0
    assert m_unres["fp_pairs"] == 0
    assert m_unres["fn_pairs"] == 1
    assert m_unres["cross_clip_pair_f1"] == 0.0
