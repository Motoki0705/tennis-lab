"""Contract and persistent registry for cross-clip player_identity_links (Issue #2746).

Preserves existing clip-local SceneResult arrays (P axis, coordinates, local_player_id)
untouched and provides a versioned downstream mapping from
(identity_scope, clip_id, local_player_id[, frame_interval]) -> global_player_id.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

LinkStatus = Literal["accepted", "unresolved", "new_candidate", "excluded"]
ScoreKind = Literal["calibrated_llr", "cosine_similarity", "combined_score", "none"]
VALID_STATUSES: frozenset[str] = frozenset({"accepted", "unresolved", "new_candidate", "excluded"})
VALID_SCORE_KINDS: frozenset[str] = frozenset({"calibrated_llr", "cosine_similarity", "combined_score", "none"})


@dataclass(frozen=True)
class LinkEvidence:
    cameras_used: tuple[str, ...]
    sample_frames: dict[str, list[int]]
    sample_source_rows: dict[str, list[int]]
    appearance_term: float | None = None
    shape_term: float | None = None
    temporal_term: float | None = None
    shape_valid: bool = False
    temporal_valid: bool = False
    gallery_update_sources: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerIdentityLink:
    clip_id: str
    local_player_id: int
    global_player_id: str | None
    status: LinkStatus
    score: float | None
    score_kind: ScoreKind
    runner_up_score: float | None
    assignment_margin: float | None
    reason: str
    evidence: LinkEvidence
    frame_interval: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must be non-empty")
        if self.local_player_id < 0:
            raise ValueError(f"local_player_id must be >= 0, got {self.local_player_id}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid link status: {self.status!r}")
        if self.score_kind not in VALID_SCORE_KINDS:
            raise ValueError(f"Invalid score_kind: {self.score_kind!r}")
        if self.status == "accepted" and not self.global_player_id:
            raise ValueError("accepted link must have a non-empty global_player_id")
        if self.status in {"unresolved", "excluded"} and self.global_player_id is not None:
            raise ValueError(f"{self.status} link must have global_player_id=None")
        if self.frame_interval is not None:
            s, e = self.frame_interval
            if s < 0 or e <= s:
                raise ValueError(f"Invalid frame_interval {self.frame_interval}")
        if not self.reason:
            raise ValueError("reason must be non-empty")


@dataclass
class IdentityRegistry:
    """Persistent anonymous global_player_id registry with immutable ID assignment."""

    identity_scope: str
    registry_version: int = 1
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    anchor_to_global_id: dict[str, str] = field(default_factory=dict)
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    def _next_id(self) -> str:
        used = {int(gid.split("_")[-1]) for gid in self.identities if gid.startswith("player_")}
        nxt = max(used, default=0) + 1
        return f"player_{nxt:02d}"

    def register_anchor(self, anchor_key: str, *, clip_id: str, local_player_id: int, metadata: dict[str, Any] | None = None) -> str:
        """Idempotently assign or return the existing anonymous global_player_id for an anchor.

        Re-ordering or re-running clips never renumbers already-registered global IDs.
        """
        if anchor_key in self.anchor_to_global_id:
            return self.anchor_to_global_id[anchor_key]
        gid = self._next_id()
        self.anchor_to_global_id[anchor_key] = gid
        self.identities[gid] = {
            "global_player_id": gid,
            "anchor_key": anchor_key,
            "origin_clip_id": clip_id,
            "origin_local_player_id": local_player_id,
            "gallery_updates": [],
            "metadata": dict(metadata or {}),
        }
        self.revision_history.append(
            {
                "registry_version": self.registry_version,
                "action": "register_anchor",
                "global_player_id": gid,
                "anchor_key": anchor_key,
                "clip_id": clip_id,
                "local_player_id": local_player_id,
            }
        )
        return gid

    def record_gallery_update(self, global_player_id: str, *, clip_id: str, local_player_id: int, score: float, margin: float) -> None:
        if global_player_id not in self.identities:
            raise KeyError(f"Unknown global_player_id: {global_player_id}")
        self.registry_version += 1
        entry = {
            "registry_version": self.registry_version,
            "clip_id": clip_id,
            "local_player_id": local_player_id,
            "score": float(score),
            "margin": float(margin),
        }
        self.identities[global_player_id]["gallery_updates"].append(entry)
        self.revision_history.append({"action": "gallery_update", "global_player_id": global_player_id, **entry})

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_scope": self.identity_scope,
            "registry_version": self.registry_version,
            "identities": copy.deepcopy(self.identities),
            "anchor_to_global_id": dict(self.anchor_to_global_id),
            "revision_history": copy.deepcopy(self.revision_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityRegistry:
        return cls(
            identity_scope=str(data["identity_scope"]),
            registry_version=int(data["registry_version"]),
            identities=copy.deepcopy(data["identities"]),
            anchor_to_global_id=dict(data["anchor_to_global_id"]),
            revision_history=copy.deepcopy(data.get("revision_history", [])),
        )


@dataclass(frozen=True)
class PlayerIdentityLinksDocument:
    schema_version: int
    identity_scope: str
    registry_version: int
    source_manifest_hashes: dict[str, str]
    upstream_digests: dict[str, str]
    encoder_identity: dict[str, Any]
    preprocessing_identity: dict[str, Any]
    method: str
    config: dict[str, Any]
    links: tuple[PlayerIdentityLink, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if not self.identity_scope or not self.method:
            raise ValueError("identity_scope and method must be non-empty")
        validate_no_cooccurring_duplicate_global_ids(self.links)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_scope": self.identity_scope,
            "registry_version": self.registry_version,
            "source_manifest_hashes": dict(self.source_manifest_hashes),
            "upstream_digests": dict(self.upstream_digests),
            "encoder_identity": copy.deepcopy(self.encoder_identity),
            "preprocessing_identity": copy.deepcopy(self.preprocessing_identity),
            "method": self.method,
            "config": copy.deepcopy(self.config),
            "links": [asdict(link) for link in self.links],
        }

    def canonical_sha256(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _intervals_overlap(a: tuple[int, int] | None, b: tuple[int, int] | None) -> bool:
    if a is None or b is None:
        return True
    return a[0] < b[1] and b[0] < a[1]


def validate_no_cooccurring_duplicate_global_ids(links: tuple[PlayerIdentityLink, ...] | list[PlayerIdentityLink]) -> None:
    """Enforce one-to-one constraint: two co-occurring distinct local players in the same clip
    cannot share an accepted global_player_id.
    Non-overlapping frame_intervals of the same global_player_id are permitted.
    """
    by_clip_gid: dict[tuple[str, str], list[PlayerIdentityLink]] = {}
    for link in links:
        if link.status != "accepted" or link.global_player_id is None:
            continue
        key = (link.clip_id, link.global_player_id)
        for existing in by_clip_gid.get(key, []):
            if existing.local_player_id != link.local_player_id and _intervals_overlap(
                existing.frame_interval, link.frame_interval
            ):
                raise ValueError(
                    f"Duplicate accepted global_player_id={link.global_player_id!r} in clip {link.clip_id!r} "
                    f"for co-occurring local_player_ids {existing.local_player_id} and {link.local_player_id}"
                )
        by_clip_gid.setdefault(key, []).append(link)


def join_events_with_global_ids(
    events: list[dict[str, Any]],
    doc: PlayerIdentityLinksDocument,
    *,
    require_accepted: bool = True,
) -> list[dict[str, Any]]:
    """Downstream join example for #1722/#2545 event/stats tables without mutating SceneResult.

    Each event row must contain 'clip_id' and 'local_player_id' (and optionally 'frame').
    When require_accepted=True, unresolved/new_candidate/excluded/missing mappings raise ValueError
    so downstream consumers cannot silently mix or drop ambiguous identities.
    """
    index: dict[tuple[str, int], list[PlayerIdentityLink]] = {}
    for link in doc.links:
        index.setdefault((link.clip_id, link.local_player_id), []).append(link)

    joined: list[dict[str, Any]] = []
    for ev in events:
        clip_id = str(ev["clip_id"])
        local_id = int(ev["local_player_id"])
        frame = ev.get("frame")
        candidates = index.get((clip_id, local_id), [])
        matched: PlayerIdentityLink | None = None
        if len(candidates) == 1 and candidates[0].frame_interval is None:
            matched = candidates[0]
        elif frame is not None:
            f_int = int(frame)
            for cand in candidates:
                if cand.frame_interval is None or (cand.frame_interval[0] <= f_int < cand.frame_interval[1]):
                    matched = cand
                    break
        elif candidates:
            raise ValueError(
                f"Ambiguous interval-split mapping for ({clip_id}, local_id={local_id}); event must supply 'frame'"
            )

        if matched is None:
            if require_accepted:
                raise ValueError(f"No identity link found for ({clip_id}, local_id={local_id}, frame={frame})")
            row = dict(ev)
            row.update({"global_player_id": None, "identity_status": "missing", "identity_reason": "no_link"})
            joined.append(row)
            continue

        if matched.status != "accepted" and require_accepted:
            raise ValueError(
                f"Non-accepted identity link ({matched.status}: {matched.reason}) for ({clip_id}, local_id={local_id})"
            )
        row = dict(ev)
        row.update(
            {
                "global_player_id": matched.global_player_id,
                "identity_status": matched.status,
                "identity_reason": matched.reason,
                "identity_registry_version": doc.registry_version,
            }
        )
        joined.append(row)
    return joined
