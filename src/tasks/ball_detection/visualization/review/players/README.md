# Player overlays

`catalog.py` discovers `data_root/ball_detection/*/manifest.json` with the
`ball_detection_player_poses.v1` schema. The public API accepts enumerated IDs,
not file paths. Discovery errors remain visible in the catalog. Campaign files
must remain under `project_root/outputs`; adopted artifacts remain under their
dataset directory. Catalog refresh rebuilds all status and artifact caches.

The existing `PlayerPoseStore` verifies adopted NPZ/review hashes against its
pinned ball store. `readers.py` validates raw `generation.json`, `input.json` and
`tracks.npz`: checksums, input JPEG shard, clip identity, frame index/PTS, array
axes and observation masks. Raw arrays are person × frame; adopted arrays are
frame × player. Both become frame-aligned payloads without changing the files.
Only `observed` / `detection_rows >= 0` observations are sent. Raw interpolated
boxes cannot manufacture pose observations; raw keypoint scores are unchanged.

An appended public ball store is joined to the pinned input by stable `clip_id`,
geometry, frame count, FPS/time base, media identity, frame index/PTS and JPEG
offset/length. JPEG shards must be the same inode or have identical SHA256.
Joining never relaxes the original pose store's global hashes. The comparison
is done on clip cache misses; cached entries include both shard stats and pose /
review / raw-input artifact stats. A changed manifest or source index requires
catalog refresh, rather than serving stale labels.

Per source, at most two decoded clip arrays / 128 MiB are cached behind a lock.
Larger clips can be read but are not cached. Preview requests contain at most
64 frames. `frame_people` serializes observations and their trailing paths;
the display semantics are documented in the user guide below.

The service supplies status (`approved`, `needs_review`, `review_pending`,
`not_generated`, `skipped`, `failed`) independently of the chosen display mode.
Unavailable results have `available=false`; an observed-empty frame in an
available result has `available=true, people=[]`.

User operation is documented in the [Ball UI guide](../../README.md).
Tests: `tests/unit/tasks/ball_detection/visualization/review/players/`.
