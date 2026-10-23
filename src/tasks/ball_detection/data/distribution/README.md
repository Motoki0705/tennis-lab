# Portable frame-store distributions

`prepare_distribution` creates an explicitly versioned JPEG-backed v1/v2 subset.
It writes a reduced `index.npz`, `metadata.json`, original Chat annotations, caller
supplied files below `inputs/`, and `distribution.json`. Clip indices and instance
offsets are dense in the new store. Frame order, native PTS, labels, splits, image
bytes and explicit teacher masks are preserved. Video-backed v3 input is rejected.

Image payloads are **hashed in place**. Preparation never copies or links them.
The source store stays read-only, including inode ctime. The returned manifest
maps clip IDs to both source and distribution indices and payload paths. A new
dataset `version` is mandatory; the frame-table schema keeps its original meaning.
Source metadata/index hashes preserve provenance without source absolute paths.
TrackNet/Meiji source CSVs are not required to consume the normalized store.

```python
from src.tasks.ball_detection.data.distribution import prepare_distribution
from src.tasks.ball_detection.data.distribution.manifest import distribution_files

prepare_distribution(
    source_directory, staging_directory, version="selected-train-val-v1",
    clip_ids=selected_clip_ids, inputs={"inputs/split.json": split_file},
)
files = distribution_files(staging_directory, source_directory=source_directory)
```

All directories and source input files in this API must have absolute paths.
`files` maps each **relative destination** to its local upload source. Transfer
these exact files with the [Drive skill](../../../../../.agents/skills/tennis-drive/SKILL.md),
then transfer `distribution.json`. An interrupted copy remains incomplete until
all entries match. Keep transfer receipts outside the distribution. Never mark a
partial directory as a completed asset or update a published version in place.

`verify_distribution(directory, source_directory=source_directory)` verifies the
small prepared files plus every original payload without allocating payload copies.
`source_payloads` resolves exactly the pinned source metadata/index and clip mapping.
For local sample reads, pass that complete mapping explicitly to
`BallFrameStore(directory, external_payloads=...)`. This mode never guesses a path
or accepts a partial payload set. It is a local source view, not a portable asset.

After transfer, `verify_distribution(directory)` checks every file's size/SHA-256,
the exact file set, store structure, selected clips and original Chat annotation
checksums. Normal `BallFrameStore(directory)` requires all payloads declared by the
reduced metadata. Omitted source clips are no longer declared; missing selected
shards still fail. Symlinks and paths escaping the root are rejected.
`workers` defaults to one; an explicit larger value uses the existing process-shared
file verifier to hash independent files concurrently. Verification has no GPU work.

Chat annotation paths declare `annotation_path_base="store"` and are relative to
the store root. Consumers must use `store.annotation_path(clip)` before checking
the original annotation hash/notes. Legacy paths without that declaration must
be absolute; an unknown base or missing annotation fails. The distribution layer
does not change the teacher policy or generate windows.

`sample_windows(store, [(clip_id, start, length), ...])` produces small receipts
for unaugmented RGB, JPEG bytes, actual PTS, all teacher inputs and original Chat
annotations. It ignores store-global row/index offsets. Compare receipts before
and after relocation with the same OpenCV version; a different decoder version
is explicit and cannot be silently treated as the same RGB result.

Fixed test data uses the same API with a separate clip selection, version and
destination. Window rules, campaign-specific inputs, transfer batching and
measurement commands belong in the experiment's `knowledge/runs/` bundle.
