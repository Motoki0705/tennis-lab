# Verified artifact snapshots

`RcloneSnapshotStore` publishes an immutable generation of a paused local run
directory. The caller owns the completion boundary and stops all writers while
`publish` runs. A training runner can call it synchronously after atomically
publishing a checkpoint and its latest pointer. Exceptions propagate to the
runner; there is no background retry or backend substitution.

```python
from pathlib import Path
from src.utils.artifact_store import RcloneSnapshotStore

store = RcloneSnapshotStore(
    Path("/absolute/run"), remote="gdrive",
    remote_root="tennis_lab/outputs/task/train/experiment/run-id",
    config_path=Path("/private/rclone.conf"),
)
receipt = store.publish("checkpoint-unique-generation")
```

Each generation contains the original relative files and `_SNAPSHOT.json`.
Publication checks the exact remote file set, sizes and MD5 values. The marker is
written last and checked again; its SHA-256 is returned only after verification.
Local SHA-256 uses the repository's independent-provider verifier. Missing hashes,
duplicate paths, changed local files or failed communication abort publication.
Reusing a complete or partial generation is rejected. Select a fresh generation
name after inspecting a failed attempt; no remote file is deleted or overwritten.

Save the receipt outside the published directory. To restore into a new, absent
local directory, construct a store for that directory and call:

```python
receipt = store.restore(
    "checkpoint-unique-generation", manifest_sha256="<saved-marker-sha256>",
)
```

Restore requires the pinned marker hash from the saved receipt, verifies its
schema and paths, checks remote content, downloads into a sibling temporary
directory, and verifies local SHA-256 before exposing the tree by rename. Existing
local trees are rejected. The snapshot API does not relocate paths inside files:
strict training resume must restore inputs and outputs at their original paths.

Use a single owner for a run's persistence. `publish(..., include=["relative/file", ...])`
can select an explicit file set. A checkpoint owner can include its latest
checkpoint/weights, both manifests, configuration and logs, while retaining older
checkpoints in their previous immutable generations. Omit `include` to publish
the entire tree. Selection never deletes prior generations or local files and
is recorded exactly in the marker. Publication and restoration receipts include
elapsed time; no retention policy is inferred.
