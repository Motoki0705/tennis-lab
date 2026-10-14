"""Cross-campaign source identity, bounded admission, and incremental task registration."""

from __future__ import annotations

import json
import sqlite3
from fractions import Fraction
from typing import Any

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef, ClipManifest, SourceIdentity, content_digest
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, save_plan
from ..registry import Registry
from .contracts import (
    AcquisitionReceipt,
    HistoricalSource,
    PreparationReceipt,
    RegistrationResult,
    SelectionReceipt,
    SourceCandidate,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
  id TEXT PRIMARY KEY, digest TEXT NOT NULL, plan TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
  provider TEXT NOT NULL, channel_id TEXT NOT NULL, campaign_id TEXT REFERENCES campaigns(id),
  PRIMARY KEY (provider, channel_id)
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, provider TEXT NOT NULL, channel_id TEXT NOT NULL, video_id TEXT NOT NULL,
  campaign_id TEXT REFERENCES campaigns(id), candidate TEXT NOT NULL, state TEXT NOT NULL,
  acquisition TEXT, selection TEXT, preparation TEXT, reason TEXT,
  UNIQUE (provider, video_id)
);
CREATE TABLE IF NOT EXISTS source_content (
  digest TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id)
);
CREATE TABLE IF NOT EXISTS clips (
  id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
  campaign_id TEXT NOT NULL REFERENCES campaigns(id), manifest TEXT NOT NULL,
  media_digest TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS annotation_tasks (
  id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id),
  task TEXT NOT NULL, clip_id TEXT NOT NULL REFERENCES clips(id),
  model_digest TEXT NOT NULL, inference_digest TEXT NOT NULL, state TEXT NOT NULL,
  UNIQUE (task, clip_id)
);
CREATE TABLE IF NOT EXISTS collection_observations (
  id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id),
  candidate TEXT NOT NULL, outcome TEXT NOT NULL
);
"""


class Backpressure(RuntimeError):
    """No new acquisition until downstream preparation consumes the lead."""


class CollectionLedger:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("collection.v1", SCHEMA)

    def register_campaign(self, plan: CampaignPlan) -> None:
        save_plan(self.layout, plan)
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT digest FROM campaigns WHERE id=?", (plan.campaign_id,)
            ).fetchone()
            if row is not None and row["digest"] != plan.sha256:
                raise ValueError("campaign already registered with a different plan")
            database.execute(
                "INSERT OR IGNORE INTO campaigns VALUES (?,?,?)",
                (
                    plan.campaign_id,
                    plan.sha256,
                    plan.canonical_bytes().decode(),
                ),
            )

    @staticmethod
    def _require_plan(database: sqlite3.Connection, plan: CampaignPlan) -> None:
        row = database.execute(
            "SELECT digest FROM campaigns WHERE id=?", (plan.campaign_id,)
        ).fetchone()
        if row is None or row["digest"] != plan.sha256:
            raise ValueError(
                "register the exact campaign plan before collecting sources"
            )

    def register_source(
        self, plan: CampaignPlan, candidate: SourceCandidate
    ) -> RegistrationResult:
        verify_reference(self.layout, candidate.evidence)
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            existing = database.execute(
                "SELECT campaign_id FROM sources WHERE id=?", (candidate.source_id,)
            ).fetchone()
            channel = database.execute(
                "SELECT campaign_id FROM channels WHERE provider=? AND channel_id=?",
                (
                    candidate.provider,
                    candidate.channel_id,
                ),
            ).fetchone()
            if existing is not None:
                result = RegistrationResult(
                    source_id=candidate.source_id,
                    disposition="existing"
                    if existing["campaign_id"] == plan.campaign_id
                    else "duplicate_source",
                    owner_campaign=existing["campaign_id"],
                    reason="source identity already registered",
                )
            elif channel is not None and channel["campaign_id"] != plan.campaign_id:
                result = RegistrationResult(
                    source_id=candidate.source_id,
                    disposition="duplicate_channel",
                    owner_campaign=channel["campaign_id"],
                    reason="channel used by another campaign or history",
                )
            else:
                n_channels = database.execute(
                    "SELECT count(*) FROM channels WHERE campaign_id=?",
                    (plan.campaign_id,),
                ).fetchone()[0]
                n_videos = database.execute(
                    """SELECT count(*) FROM sources WHERE campaign_id=?
                    AND provider=? AND channel_id=? AND state NOT IN ('excluded','failed')""",
                    (
                        plan.campaign_id,
                        candidate.provider,
                        candidate.channel_id,
                    ),
                ).fetchone()[0]
                if (
                    channel is None and n_channels >= plan.counts.channels
                ) or n_videos >= plan.counts.videos_per_channel:
                    result = RegistrationResult(
                        source_id=candidate.source_id,
                        disposition="limit",
                        owner_campaign=None,
                        reason="requested channel or videos-per-channel limit reached",
                    )
                else:
                    database.execute(
                        "INSERT OR IGNORE INTO channels VALUES (?,?,?)",
                        (
                            candidate.provider,
                            candidate.channel_id,
                            plan.campaign_id,
                        ),
                    )
                    database.execute(
                        """INSERT INTO sources
                        (id,provider,channel_id,video_id,campaign_id,candidate,state) VALUES (?,?,?,?,?,?,'selected')""",
                        (
                            candidate.source_id,
                            candidate.provider,
                            candidate.channel_id,
                            candidate.video_id,
                            plan.campaign_id,
                            candidate.canonical_bytes().decode(),
                        ),
                    )
                    result = RegistrationResult(
                        source_id=candidate.source_id,
                        disposition="registered",
                        owner_campaign=plan.campaign_id,
                        reason=None,
                    )
            observation_id = content_digest(
                [plan.campaign_id, candidate.sha256, result.model_dump()]
            )
            database.execute(
                "INSERT OR IGNORE INTO collection_observations VALUES (?,?,?,?)",
                (
                    observation_id,
                    plan.campaign_id,
                    candidate.canonical_bytes().decode(),
                    result.canonical_bytes().decode(),
                ),
            )
            return result

    def import_history(
        self, source: HistoricalSource | SourceIdentity, evidence: ArtifactRef
    ) -> None:
        """Seed read-only inventory before planning; never claim historical media as new."""
        verify_reference(self.layout, evidence)
        if source.media is not None:
            verify_reference(self.layout, source.media)
        with self.registry.transaction() as database:
            existing = database.execute(
                "SELECT state,candidate,reason FROM sources WHERE id=?",
                (source.canonical_id,),
            ).fetchone()
            if existing is not None:
                if existing["state"] != "historical":
                    raise ValueError("cannot rewrite a campaign source as historical")
                if (existing["candidate"], existing["reason"]) != (
                    source.canonical_bytes().decode(),
                    evidence.canonical_bytes().decode(),
                ):
                    raise ValueError("historical source or evidence changed")
                return
            database.execute(
                "INSERT OR IGNORE INTO channels VALUES (?,?,NULL)",
                (source.provider, source.channel_id),
            )
            database.execute(
                """INSERT INTO sources
                (id,provider,channel_id,video_id,candidate,state,reason) VALUES (?,?,?,?,?,'historical',?)""",
                (
                    source.canonical_id,
                    source.provider,
                    source.channel_id,
                    source.video_id,
                    source.canonical_bytes().decode(),
                    evidence.canonical_bytes().decode(),
                ),
            )
            if source.media is not None:
                database.execute(
                    "INSERT OR IGNORE INTO source_content VALUES (?,?)",
                    (source.media.digest, source.canonical_id),
                )

    def source(self, plan: CampaignPlan, source_id: str) -> dict[str, Any]:
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            row = database.execute(
                "SELECT * FROM sources WHERE id=? AND campaign_id=?",
                (source_id, plan.campaign_id),
            ).fetchone()
            if row is None:
                raise ValueError("source does not belong to this campaign")
            return dict(row)

    def begin_acquisition(
        self, plan: CampaignPlan, source_id: str, limit: int, *, retry: bool = False
    ) -> bool:
        if limit < 1:
            raise ValueError("acquisition lead limit must be positive")
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            row = database.execute(
                "SELECT state FROM sources WHERE id=? AND campaign_id=?",
                (source_id, plan.campaign_id),
            ).fetchone()
            if row is None:
                raise ValueError("unregistered source")
            state = row["state"]
            if state in {"acquired", "screened", "prepared"}:
                return False
            if state == "excluded" or (state == "failed" and not retry):
                raise ValueError(
                    "excluded/failed source needs an explicit decision or retry"
                )
            # An interrupted acquiring source already owns its lead slot. The
            # caller holds its process-scoped source lock before reaching here.
            if state != "acquiring":
                ahead = database.execute(
                    "SELECT count(*) FROM sources WHERE state IN ('acquiring','acquired','screened')"
                ).fetchone()[0]
                if ahead >= limit:
                    raise Backpressure("unprepared source lead is full")
            database.execute(
                "UPDATE sources SET state='acquiring', reason=NULL WHERE id=?",
                (source_id,),
            )
            return True

    def record_acquisition(self, plan: CampaignPlan, reference: ArtifactRef) -> bool:
        receipt = read_document(self.layout, reference, AcquisitionReceipt)
        verify_reference(self.layout, receipt.source.media)
        source_id = receipt.source.canonical_id
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            row = database.execute(
                "SELECT * FROM sources WHERE id=? AND campaign_id=?",
                (source_id, plan.campaign_id),
            ).fetchone()
            if (
                row is None
                or content_digest(json.loads(row["candidate"]))
                != receipt.candidate_digest
            ):
                raise ValueError("acquisition candidate identity mismatch")
            if (
                receipt.configuration != plan.preparation
                or receipt.source.channel_id != row["channel_id"]
            ):
                raise ValueError("acquisition differs from the campaign inputs")
            if row["acquisition"] is not None:
                if row["acquisition"] != reference.canonical_bytes().decode():
                    raise ValueError("source acquisition is immutable")
                return bool(row["state"] != "excluded")
            owner = database.execute(
                "SELECT source_id FROM source_content WHERE digest=?",
                (receipt.source.media.digest,),
            ).fetchone()
            duplicate = owner is not None and owner["source_id"] != source_id
            database.execute(
                "UPDATE sources SET acquisition=?, state=?, reason=? WHERE id=?",
                (
                    reference.canonical_bytes().decode(),
                    "excluded" if duplicate else "acquired",
                    f"duplicate content of {owner['source_id']}" if duplicate else None,
                    source_id,
                ),
            )
            if not duplicate:
                database.execute(
                    "INSERT OR IGNORE INTO source_content VALUES (?,?)",
                    (receipt.source.media.digest, source_id),
                )
            return not duplicate

    def record_selection(self, plan: CampaignPlan, reference: ArtifactRef) -> None:
        receipt = read_document(self.layout, reference, SelectionReceipt)
        verify_reference(self.layout, receipt.evidence)
        row = self.source(plan, receipt.source_id)
        if row["acquisition"] is None:
            raise ValueError("selection requires an acquired source")
        acquired = read_document(
            self.layout,
            ArtifactRef.model_validate_json(row["acquisition"]),
            AcquisitionReceipt,
        )
        if (
            receipt.source_digest != acquired.source.media.digest
            or receipt.policy != plan.selection
        ):
            raise ValueError("selection source/policy does not match the plan")
        if receipt.requested_maximum != plan.counts.max_clips_per_video:
            raise ValueError("selection limit differs from the plan")
        with self.registry.transaction() as database:
            current = database.execute(
                "SELECT state,selection FROM sources WHERE id=?", (receipt.source_id,)
            ).fetchone()
            if current["selection"] is not None:
                if current["selection"] != reference.canonical_bytes().decode():
                    raise ValueError("source selection is immutable")
                return
            if current["state"] != "acquired":
                raise ValueError("only acquired sources can be selected")
            database.execute(
                "UPDATE sources SET selection=?,state=?,reason=? WHERE id=?",
                (
                    reference.canonical_bytes().decode(),
                    "screened" if receipt.windows else "excluded",
                    receipt.shortage_reason,
                    receipt.source_id,
                ),
            )

    def record_preparation(self, plan: CampaignPlan, reference: ArtifactRef) -> None:
        from ..media import check_clip, probe_video
        from .contracts import PreparationConfig
        from .preparation import preparation_identity, window_ranges

        receipt = read_document(self.layout, reference, PreparationReceipt)
        selection = read_document(self.layout, receipt.selection, SelectionReceipt)
        manifests = [
            read_document(self.layout, ref, ClipManifest) for ref in receipt.clips
        ]
        if not manifests or len(manifests) != len(selection.windows):
            raise ValueError(
                "prepared clips must cover the complete nonempty selection"
            )
        if (
            receipt.configuration != plan.preparation
            or receipt.source_digest != selection.source_digest
            or receipt.source_id != selection.source_id
        ):
            raise ValueError("preparation input mismatch")
        registered = self.source(plan, receipt.source_id)
        if registered["acquisition"] is None:
            raise ValueError("preparation requires a registered acquisition")
        acquired = read_document(
            self.layout,
            ArtifactRef.model_validate_json(registered["acquisition"]),
            AcquisitionReceipt,
        )
        config = read_document(self.layout, plan.preparation, PreparationConfig)
        timeline = probe_video(verify_reference(self.layout, acquired.source.media))
        for manifest, window in zip(manifests, selection.windows, strict=True):
            if manifest.source != acquired.source:
                raise ValueError("prepared manifest source mismatch")
            media_range, target_range = window_ranges(timeline, window)
            if (
                manifest.source_range != media_range
                or manifest.target_range != target_range
                or len(manifest.frames) < config.minimum_clip_frames
                or manifest.preparation_digest
                != preparation_identity(
                    acquired.source, plan.preparation, receipt.selection, window
                )
            ):
                raise ValueError(
                    "prepared manifest differs from the selected interval/configuration"
                )
            if (
                Fraction(manifest.time_base) != timeline.time_base
                or Fraction(manifest.nominal_fps) != timeline.rate
            ):
                raise ValueError("prepared manifest has a different source clock")
            for frame in manifest.frames:
                if (
                    frame.source_pts != timeline.pts[frame.source_frame_index]
                    or frame.duration_pts
                    != timeline.durations[frame.source_frame_index]
                ):
                    raise ValueError(
                        "prepared frame mapping differs from the real source"
                    )
            check_clip(verify_reference(self.layout, manifest.media), manifest)
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            row = database.execute(
                "SELECT * FROM sources WHERE id=? AND campaign_id=?",
                (receipt.source_id, plan.campaign_id),
            ).fetchone()
            if (
                row is None
                or row["selection"] != receipt.selection.canonical_bytes().decode()
            ):
                raise ValueError("preparation does not use the registered selection")
            if row["preparation"] is not None:
                if row["preparation"] != reference.canonical_bytes().decode():
                    raise ValueError("preparation receipt is immutable")
                return
            if row["state"] != "screened":
                raise ValueError("source is not ready for preparation")
            for manifest, ref in zip(manifests, receipt.clips, strict=True):
                database.execute(
                    "INSERT INTO clips VALUES (?,?,?,?,?)",
                    (
                        manifest.clip_id,
                        receipt.source_id,
                        plan.campaign_id,
                        ref.canonical_bytes().decode(),
                        manifest.media.digest,
                    ),
                )
                for task in plan.tasks:
                    task_id = content_digest([task.task, manifest.clip_id])
                    database.execute(
                        "INSERT INTO annotation_tasks VALUES (?,?,?,?,?,?,'pending')",
                        (
                            task_id,
                            plan.campaign_id,
                            task.task,
                            manifest.clip_id,
                            task.model.sha256,
                            task.inference.digest,
                        ),
                    )
            database.execute(
                "UPDATE sources SET state='prepared',preparation=? WHERE id=?",
                (reference.canonical_bytes().decode(), receipt.source_id),
            )

    def acquisition_failed(
        self, plan: CampaignPlan, source_id: str, reason: str
    ) -> None:
        if not reason.strip():
            raise ValueError("failure must have a reason")
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            updated = database.execute(
                "UPDATE sources SET state='failed',reason=? WHERE id=? AND campaign_id=? AND state='acquiring'",
                (reason, source_id, plan.campaign_id),
            )
            if updated.rowcount != 1:
                raise ValueError("only in-progress acquisition can fail")

    def summary(self, plan: CampaignPlan) -> dict[str, Any]:
        with self.registry.transaction() as database:
            self._require_plan(database, plan)
            states = dict(
                database.execute(
                    "SELECT state,count(*) FROM sources WHERE campaign_id=? GROUP BY state",
                    (plan.campaign_id,),
                ).fetchall()
            )
            n_channels = database.execute(
                "SELECT count(*) FROM channels WHERE campaign_id=?", (plan.campaign_id,)
            ).fetchone()[0]
            n_clips = database.execute(
                "SELECT count(*) FROM clips WHERE campaign_id=?", (plan.campaign_id,)
            ).fetchone()[0]
            n_tasks = database.execute(
                "SELECT count(*) FROM annotation_tasks WHERE campaign_id=?",
                (plan.campaign_id,),
            ).fetchone()[0]
        return {
            "requested": plan.counts.model_dump(),
            "reserved_channels": n_channels,
            "sources_by_state": states,
            "prepared_clips": n_clips,
            "annotation_tasks": n_tasks,
            "missing_prepared_videos": max(
                0, plan.counts.videos - states.get("prepared", 0)
            ),
        }
