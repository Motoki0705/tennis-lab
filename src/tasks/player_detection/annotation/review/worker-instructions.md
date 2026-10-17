Review one clip for player bbox teachers using the frozen target definition, bbox
definition and numerical policy below. Raw DINO boxes are unreviewed proposals,
including frames with no detections. Do not infer track IDs, player identities,
poses, publication decisions or model adoption from this review.

Write only in your attempt's `work/`. Call the fixed parent service command below
with `--operation NAME --parameters /absolute/path/in/work/params.json` and the
stated PYTHONPATH. Keep full JSON artifacts out of tool messages.

1. `review_handoff` takes `scope: {start,stop}` (half-open, at most 64 frames),
   optional `expected_revision`, `after_row`, and `previous_revision`.
   `review_rows` takes `expected_revision` and `scope` (at most 8 frames, 64 KiB
   response limit). It returns current `rows` and aligned immutable raw `proposals`
   with original observation IDs, bboxes and confidence. Read these bounded pages
   and exact row digests before replacing them. Use a fresh
   handoff after revision conflicts. Preserve original and added observation IDs.
   `image_proof=missing` requires re-viewing evidence and explicitly replacing
   the decision. The handoff's `image_proof_reason` and the row page's aligned
   `image_proof_reasons` explain missing prior evidence, including incomplete
   model metadata or conflicting response IDs. `pending_current_attempt` is checked after exit; `verified`
   denotes actual image bytes already verified for the prior reviewer.
2. `review_image` takes `expected_revision`, `drawing` (`plain`, `prediction`,
   `revision`), `tiles` (up to 16 `{frame_index,crop}`), `columns` (1..4),
   `tile_width`, `tile_height` (32..1024), and `reason`. Native crop edges are
   integer `{left,top,right,bottom}`. The parent chooses overlay boxes; a page
   supports at most 32 overlays per tile. Request plain images if there are more.
   Open the returned `view_path` with the actual `view_image` tool. Rendering or
   listing the file does not count as seeing it. Every claim of complete target
   inventory, including empty target, requires full-frame evidence. Bbox decisions
   additionally require a detailed crop containing the entire box. The two scale
   limits are independent; use both a full frame and local crops when necessary.
3. `review_edit` takes `expected_revision`, `changes` (ArtifactRef to a JSON file
   in your own work directory, role `annotation`, exact SHA256/byte size), and
   `reason`. Changes conform to `PlayerEdits` in
   `src.tasks.player_detection.annotation.review.contracts`: `operations` plus
   returned `RenderedImage` objects in `images`.
   - `confirm_predictions`: explicit `scope` and `evidence` IDs. Only use when
     every candidate is a target player, every proposed bbox is observed and
     correct, and the full-frame target inventory is complete. The frozen policy
     must permit it. Empty raw frames need `confirm_empty` instead.
   - `confirm_empty`: explicit `scope` and evidence for genuinely empty target
     frames with no raw candidates. Do not confuse failed detection with absence.
   - `replace`: `frame_index`, `expected_frame_digest`, complete `FrameDecision`.
     Account for every original candidate: `player`, `non_player`, `duplicate`
     (name the retained player with `duplicate_of`), or `unresolved`. Do not
     silently drop rejected candidates or previously added observations.
     Correct/add/remove bbox by replacing the corresponding observation decision.
     Removal needs detailed evidence of the latest corrected bbox, including
     observations that originated from a model proposal.
     Original observations keep `observation_id=proposal_id`, `manual_id=null`.
     Added observations use `proposal_id=null`, a stable local `manual_id`, and
     `manual_observation_id(clip_id, frame_index, manual_id)` from the contracts.
     Sort observations by ID. Observed/estimated boxes use native xyxy edges up to
     `(width,height)`, unresolved boxes are null, rejected candidates use
     `box_state=not_applicable` and no bbox. Manual corrections/additions/rejections
     use `origin=manual`, `score_origin=missing`, `confidence=0`; zero is a mask,
     not a detector score. Never copy a raw confidence onto a changed box.
     Reviewed coverage can be unknown/partial/complete. Preserve blocking notes
     until resolved from images; uncertainty and estimated boxes are not teachers.

Every reviewed frame needs its own image evidence. A review flag or image claim
must not be copied across frames or workers. Only target frames with complete,
resolved, observed bbox decisions become positive/empty-negative teachers;
context, partial inventory, unresolved or estimated boxes are excluded as a whole
frame. Do not claim that visible individual boxes make an incomplete frame safe
as detector supervision. Teacher masks are derived by the parent.

Finish with `work/result.json`: `status` (`complete` or `deferred`), current
`revision` ArtifactRef, and a nonempty `reason`. Defer uncertainty or unmet policy.
The parent verifies actual image bytes and per-attempt authorship, then freezes
the exact bbox revision for downstream ID review. Completion does not seal or
adopt teachers; final ID corrections must be applied before final adoption.
