Annotate the ONE live ball being played on the court this camera faces. Handheld,
spare and adjacent-court balls are not this target. Work from the supplied plain
images of your assigned camera only. Other cameras, model predictions, overlays
and reference answers are unavailable and must not be sought.

Choose your own exploration and viewing order. The same tools support chronological
overview pages, arbitrary temporal anchors, native crops/zoom, interpolation and
adaptive revisits. Use preceding/following frames and informed inference to leave
as few unknowns as possible. Preserve honest provenance; never call an inferred
position an observation. Estimated coordinates are useful and need no additional
independent visual review or human QA.

States: in_frame is the live ball in this image (with position, or known occlusion
without position). out_of_frame means a live ball exists outside the image.
no_active_ball means no live ball is currently being played, even when spare balls
are visible. unknown is a processed final judgment that cannot be resolved after
using context/inference. unprocessed means work remains, and only these frames
will be carried into the next attempt. Unknown may remain in a complete annotation.

Use native pixel-centre coordinates (0 <= x <= width-1, 0 <= y <= height-1), original
source frame indices and exact supplied PTS. Do not clamp coordinates. Out-of-frame
has no position. observed requires directly visible position and not_occluded;
interpolated needs two enclosing same-camera support_frames; estimated and
occlusion_estimated retain their correct kind/support and notes. The row schema
and all supplied frame identities are in ../input/. Only supplied frames can be
requested or cited as support. You may correct your own earlier rows explicitly.

Tools: write a JSON parameters file inside work, then run
`/usr/bin/python3 -I -S ../input/tools.py OPERATION --parameters PARAMETERS.json`.
The parent saves each accepted edit as an atomic revision before replying. Keep
the returned revision for the next edit; a stale expected_revision is rejected.

- video_handoff: {camera_id, after_frame?: null, limit?: 128}. Returns remaining
  unprocessed source indices and counts. Follow next_after_frame for more; you are
  free to select any remaining frame. Final unknowns are excluded from remaining.
- video_rows: {camera_id, frames: [source indices]}. At most 128 rows, any order.
- video_image: {camera_id, tiles: [{frame_index, crop: {left, top, right, bottom}}],
  columns, tile_width, tile_height, reason}. At most 16 tiles, columns 1..4, tile
  dimensions 32..1024. Half-open crops must fit native bounds. The plain image page
  and its crop/scale/PTS metadata are returned. Use view_image on image_path.
- video_edit: {camera_id, expected_revision, rows: [complete v3 frame rows], reason}.
  Up to 128 rows per atomic save. Use actual input PTS and honest provenance.

Save useful edits frequently. Before the attempt deadline, write work/result.json
with {status: "complete" or "deferred", revision: THE_CURRENT_REVISION, reason}.
Choose complete only when unprocessed=0. Deferred keeps saved progress; it does
not erase final unknowns. Final prose does not save edits or complete the task.
