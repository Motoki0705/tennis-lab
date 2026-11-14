# Local D:/E: storage tiers

[日本語](README.ja.md) · [简体中文](README.zh-CN.md)

This is the source of truth for local storage placement and retention. D: is the
WSL root ext4 filesystem (`relatime`), containing working copies. E: is the ext4
HDD mounted at `/mnt/wsl/tennis-data` (`noatime`), containing authoritative mirrors.
[Google Drive policy](../../skills/tennis-drive/references/storage-policy.md) remains separate.
Task-relative names come from [OUTPUTS.md](../../../src/tasks/OUTPUTS.md).

## Placement and boundaries

For main checkout `R`, the E: counterpart of `R/<relative>` is
`/mnt/wsl/tennis-data/<R without its leading slash>/<relative>`.

| Location | D: | E: | Eviction unit |
|---|---|---|---|
| Main `data/` | Real directory | Same relative paths | Dataset version |
| Main `outputs/` | Real directory | Same relative paths | Task/purpose/experiment/run-id |
| Worktree `outputs/` | Real directory on D: | `.claude/worktrees/<name>/outputs/...` | Task/purpose/experiment/run-id |
| Main `ckpt/` | Entire real directory, always hydrated | Verified mirror | Never evicted |
| Worktree `data/`, `ckpt/`, `.venv` | Links to main checkout | No independent copies | Not independently scanned |

[config.toml](config.toml) declares dataset **containers**, whose direct children
are versions, and exceptional **unit roots**. Overlapping declarations are rejected.
New task namespaces must be declared there. Raw assets and ambiguous legacy layouts
are reported as `unmanaged`, without recursive guesses. Output discovery uses the
standard four-component run hierarchy and known purposes. An experiment directory
containing files is treated as ambiguous legacy output, not as a collection of runs.
E: worktree output directories remain discoverable after their Git registrations disappear.
Directory symlinks in managed root paths, nested mounts, special files and all
symlinks inside a unit are rejected. Linux readlink updates old symlink atimes even
with an O_PATH descriptor; rejecting them before readlink keeps dry-runs read-only.
Use ordinary files/hardlinks for managed payloads; this tool never rewrites existing links.

Hardlinks work within D: and within E:, never across them. Copies preserve hardlinks
**within** one unit, but use independent inodes between units and tiers. Eviction
removes only that unit's names. The audit separates logical bytes, unique allocated
bytes, external hardlinks, estimated reclaimable bytes and the observed filesystem
free-space delta. External links can keep all blocks allocated. Hydration creates
fresh D: inodes; it does not recreate links into other units.

## CLI

Run from the checkout containing the desired implementation. The finite
[checkout launcher](../lib/README.md#既存cliと新規jobの接続) is mandatory, including dry-runs.

```bash
export PYTHONPATH="$PWD/.agents/ops"
export PYTHONDONTWRITEBYTECODE=1
S=(.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier)
"${S[@]}" scan
"${S[@]}" mirror data/ball_detection/example-v1 --dry-run
"${S[@]}" mirror data/ball_detection/example-v1
"${S[@]}" verify data/ball_detection/example-v1 --dry-run
"${S[@]}" index data/ball_detection/example-v1 --dry-run
"${S[@]}" hydrate data/ball_detection/example-v1
"${S[@]}" evict data/ball_detection/example-v1 --dry-run
"${S[@]}" cycle --dry-run
```

`--repo`, `--config` and `--queue-dir` precede the subcommand. `--repo` selects the
same Git common directory as the launcher; the main checkout is derived from Git.
Explicit unit arguments may name a child inside exactly one discovered unit.
Without arguments, mirror/verify/evict/cycle use local units. Index/hydrate require
explicit units. Use explicit arguments for real-machine probes as well.
`scan` never hashes payloads. `index --dry-run` hashes only the selected E: units
and reports byte/file counts, a tree digest and whether atime was preserved.
Dry-runs create no lock, manifests, directories, audit records or copies.
Exit codes: 0 success or expected retention; 1 failed safety/verification; 2 syntax;
75 missing checkout launcher. Missing E: fails even in dry-run.

After migration, run `activate` to publish `.storage-tier/active.json` locally.
An actual automatic `cycle` without explicit units requires that activation.
The common `PathResolver` and non-Hydra input boundaries then reject archived
DATA/CHECKPOINT/ARTIFACT references with an explicit hydrate command. They stat E:
only to diagnose the missing input; they never read or fetch its payload. A missing
mount is a separate error. Custom roots outside this repository and consumers
bypassing the shared boundary retain their existing missing-file errors. Creating
new OUTPUT paths remains possible. No training entrypoints or queue commands are patched.

## Verification and exclusion

`mountpoint` and `findmnt` must confirm separate ext4 filesystems and the prescribed
atime options. A missing mount is never interpreted as an empty mirror. All data,
directory and manifest reads use `O_NOATIME`; a failed open is not retried normally.
The synthetic `/proc` root is enumerated normally because it belongs to root and
does not hold dataset access history.

Before mirroring or certifying a unit, inspect open descriptors, process cwd and
mappings by device/inode, including external hardlink aliases. An unreadable
process or queue directory fails the operation. Any `jobs/*.job` or `running/*.job`
blocks **all units**: arbitrary Python/Hydra/shell code can resolve inputs indirectly,
so parsing command text cannot prove that other units are unused. Stale running
records must be resolved through the queue workflow; this tool does not rewrite them.
Recently written units wait for the configured settling interval (default one hour).
Snapshots before/after hashing and publication detect concurrent changes. Published
metadata must match the hashed staging tree before its digests can enter a manifest.

Mirroring hashes all D: files, copies into a new E: staging tree, hashes all staged
E: files and compares the complete tree. A pre-existing unrecorded E: tree must
match D: before being adopted. Recorded mirrors can be replaced only if their E:
metadata is unchanged. Publication uses Linux `RENAME_NOREPLACE`/`RENAME_EXCHANGE`.
Manifests are atomic JSON under E:`<mirror-repo>/.storage-tier/manifests/`.
They contain both sides' SHA-256 and size/mtime/ctime/device/inode metadata.

Eviction requires every regular file's atime to be at least 14 days old. Empty
units remain. D: is rehashed against the manifest; E:'s full file/directory set and
recorded metadata must remain unchanged. The D: directory is quarantined under a
temporary sibling name, then checked again before removal. `ckpt/` is never evicted.
An expired unit with existing evidence is verified **before** the daily cycle can
refresh it, so a hash mismatch cannot be hidden by remirroring it first.
Hydration rejects existing D: destinations, hashes E:, copies and hashes D:, then
publishes the whole unit with fresh file atimes and updates the manifest.

## Journals and recovery

D:`<repo>/.storage-tier/audit.jsonl` records verification, hydration, failure, deletion
intent and completion with UTC time, unit, byte counts and verification method.
A single repo lock serializes mutating operations; dry-runs do not acquire/create it.
Durable deletion transactions live in `transactions/` beside the audit.

```bash
"${S[@]}" recover
```

Recovery restores quarantines with no overwrite and **never resumes deletion**.
An occupied original path or changed identity is an explicit failure. A partially
deleted tree is restored and marked as possibly partial; inspect it and use the
verified E: copy for recovery. If removal completed before a crash, recovery records
that fact without claiming a measured capacity gain. Any pending transaction blocks
further operations until recovery. An interrupted E: exchange keeps the old mirror
under its staging name if manifest publication failed; inspect both trees, use
explicit `verify` only when the complete D:/E: contents match, and retain the old
copy until that proof is recorded. Staging leftovers are reported, never silently swept.

## Worktrees and service

After migration and `git worktree add`, run:

```bash
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker --dry-run
"${S[@]}" prepare-worktree /absolute/main/.claude/worktrees/new-worker
```

This requires real D: data/output/checkpoint roots and validates the full plan
before creating D: outputs and shared input/venv links.
Existing output symlinks are rejected; no existing path is switched automatically.

The existing [336-hour worktree expiration](../worktree_cleanup/README.md) removes
the entire worktree, including real D: output directories, independently of storage
verification. **The two deletion policies cannot be enabled together with this
layout until their lifecycle integration is agreed and implemented.** This tool
does not weaken that separate human-approved policy or disable its timer.

After migration, process-inspection permission and lifecycle integration are resolved:

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

The user timer runs daily at 05:30 Asia/Tokyo with up to 15 minutes of jitter.
`OnFailure` calls the user bus `org.freedesktop.Notifications.Notify` through busctl;
successful or retained-only cycles send no notification. The notification service
must exist on the host; delivery failures are visible as failure of the helper unit.
Host validation must confirm notification delivery and full `/proc` inspection.
There is no ordering dependency on the system mount service; each operation checks
the mount itself. Installation only links units. Enablement is a separate deployment step.

## Migration and validation

1. Coordinate a pause in producers, queue admissions and consumers holding old paths.
   Resolve the conflicting worktree expiration policy and the user unit's inability
   to inspect some `/proc` entries before enabling automatic deletion. Do not change
   permissions, stop other sessions or install a privileged helper implicitly.
2. Inventory E: units and active/queued references, including indirect config defaults,
   old hardlink aliases, E:-absolute command paths and pre-existing D: copies. Decide
   the needed version/run units. Hydrate all `ckpt/` and those units only.
3. Bootstrap E:-only manifests with `index`, one explicit stable unit at a time.
   At 80–150 MiB/s, one sequential pass over 432 GiB takes about 49–92 minutes;
   small-file seeks can take longer. This is a planning assumption, not a measured
   full-disk benchmark. Indexing reads E: once. Hydration then reads E: for hashing
   and copying, and D: for verification. Do not infer HDD speed from warm tiny probes.
4. Keep original symlink entries under a migration record. Detach links by renaming
   the **link itself**, never its target. Create empty real D: `data/` and `outputs/`
   roots; leave `ckpt/` absent until whole-unit hydration publishes it. Detach each
   affected worktree's output link and create its real D: root. E: payloads stay intact.
   Hydrate the chosen units, verify contents and a representative hardlink operation,
   then activate input diagnostics. Existing D: copies need explicit comparison;
   hydrate never merges into them.
5. Update affected active/queued commands in their owning sessions. Run a read-only
   cycle, then a selected disposable-unit mirror/evict/hydrate round trip. Compare
   `statvfs` free space and audit evidence; external hardlinks may prevent reclamation.
   Confirm consumer loading and failure notification before enabling the timer.
   Roll back by restoring saved symlink entries only while consumers are paused;
   preserve new D: writes separately. Root VHDX relocation/compaction is outside scope.

Validation uses temporary real files and simulated mount/process inputs in
`tests/integration/agents_ops/test_storage_tier.py`; native FD and hardlink behavior
is also exercised. CLI/worktree and unit-install contracts are in
`tests/e2e/agents_ops/test_storage_tier_cli.py` and the existing ops-control tests.
Common input-boundary diagnostics extend the existing configuration path tests.
