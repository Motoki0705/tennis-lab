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
| Worktree `outputs/` | Link to D:`<main>/.worktree-outputs/<name>/` | `.worktree-outputs/<name>/...` | Task/purpose/experiment/run-id |
| Existing worktree alias of main `outputs/` | Link to main `outputs/` | Main `outputs/...` | No independent unit |
| Main `ckpt/` | Entire real directory, always hydrated | Verified mirror | Never evicted |
| Worktree `data/`, `ckpt/`, `.venv` | Links to main checkout | No independent copies | Not independently scanned |

[config.toml](config.toml) declares dataset **containers**, whose direct children
are versions, and exceptional **unit roots**. Overlapping declarations are rejected.
New task namespaces must be declared there. Raw assets and ambiguous legacy layouts
are reported as `unmanaged`, without recursive guesses. Output discovery uses the
standard four-component run hierarchy and known purposes. An experiment directory
containing files is treated as ambiguous legacy output, not as a collection of runs.
Persistent `.worktree-outputs/<name>/` remains discoverable after Git registration
and the worktree disappear. Worktree names reuse that persistent namespace; use a
new name for a separate namespace. `outputs` links themselves are references,
never independent units. Scans list them without reading their targets, since
even stat through an old symlink changes its atime. Use canonical main `outputs/`
or `.worktree-outputs/` paths in storage commands. Real output roots in the old
`.claude/worktrees/<name>/outputs/` layout are unmanaged migration sources, listed
with their new destination in `legacy_outputs`; relocate them before indexing.
Directory symlinks in managed root paths, nested directory mounts, special files and all
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

`--repo` and `--config` precede the subcommand. `--repo` selects the
same Git common directory as the launcher; the main checkout is derived from Git.
Explicit unit arguments may name a child inside exactly one discovered unit.
Without arguments, mirror/verify/evict/cycle use local units. Index/hydrate require
explicit units. Use explicit arguments for real-machine probes as well.
`scan` never hashes payloads. It also reports `unmanaged_data_sizes_E`: per
unclassified E:`data/<top-level>` regular-file logical bytes and allocated bytes
including directories/symlinks, deduplicated by inode within that top level, like
`du`. Symlink targets and nested mounts are not traversed; directory reads use
`O_NOATIME`. Failed sizing is an explicit error, not a zero-byte result. These
assets remain outside eviction. `index --dry-run` hashes only the selected E: units
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
The synthetic `/proc` root and kernel-generated metadata are read normally because
they do not hold dataset access history. Injected real test directories still use
`O_NOATIME`.

Before mirroring or certifying a unit, inspect open descriptors, process cwd and
mappings by device/inode, including external hardlink aliases. Only processes
whose effective UID in `/proc/<pid>/status` matches the execution user's effective
UID are inspected. Other UIDs, including root and CI, are explicitly excluded;
their counts by UID appear in `process_scope` in JSON output and audit records.
Denied same-UID processes are excluded only with independent nondumpability
evidence: all four UIDs match the non-root caller, fd/maps/cwd are root:root-owned,
all three reads fail with EACCES/EPERM, and PID starttime/UID/name/ownership remain
stable across the probes. An explicit `Dumpable: 1` contradicts exclusion;
`CoreDumping: 0` is not evidence. The ownership rule comes from
[proc_pid(5)](https://man7.org/linux/man-pages/man5/proc_pid.5.html).
This inference remains conservative for other ownership/namespace mappings.
Excluded count, readable comm (status Name), PID and evidence appear in
`process_scope.excluded_nondumpable`. Z/X states count as exited only with
`Threads: 1` and a task inventory containing just that PID: a zombie leader may
still have live threads ([proc_pid_fd(5)](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html)).
Missing metadata for a still-live process is not ignored.
Non-dumpable same-UID processes such as the Codex CLI core, sshd and gpg-agent
are outside inspection. Training/data-reading scripts must run dumpable.
Unknown UID, unexpected I/O failures, partial denials and unreadable dumpable
processes still fail closed. Capability-bearing systemd can be such an error:
ordinary ownership with denied fd targets/maps/cwd does not establish nondumpability.
Other-user consumers are therefore not protected: run storage and data consumers
as the same user. Training queue records are never read, and queued/planned jobs
may lose a 14-day-old input and need manual hydration. There is no mtime/ctime
settling interval; eligibility uses regular-file atime and open-process references.
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
Removal pins the verified root FD and checks its device/inode before deleting its
contents through that FD. A replaced quarantine is neither deleted nor moved back
as our unit; its identity mismatch remains a manual recovery condition. Mount IDs
also reject a nested directory bind mount even when its device ID is unchanged.
An expired unit with existing evidence is verified **before** the daily cycle can
refresh it, so a hash mismatch cannot be hidden by remirroring it first.
Hydration rejects existing D: destinations, hashes E:, copies and hashes D:, then
publishes the whole unit with fresh file atimes and updates the manifest.

## Journals and recovery

D:`<repo>/.storage-tier/audit.jsonl` records verification, hydration, failure, deletion
intent and completion with UTC time, unit, byte counts and verification method.
A single repo lock serializes mutating operations; dry-runs do not acquire/create it.
Process exclusion scope is included in audit events. Failures remain in this audit
and the service journal; no desktop, D-Bus, mail or other notification is sent.
Read-only `index --dry-run` reports process inspection errors but never certifies
a manifest or authorizes deletion; it can still inspect the selected tiny tree.
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

This requires real D: data/output/checkpoint roots. It creates
`<main>/.worktree-outputs/<name>/` and links the worktree's `outputs` to it, with
shared input/venv links. Both the persistent root and data must be on D:, so
hardlinks work. A correct existing persistent link or main-output alias is kept;
other existing outputs are rejected and never switched automatically.
For existing symlinks, initialization dry-run refuses target validation before
reading the link, because that would change its atime. Use actual prepare-worktree
to validate an already initialized worktree; it preserves correct link targets.

The existing [336-hour worktree expiration](../worktree_cleanup/README.md) removes
only the worktree and its symlink entries, never their targets. Persistent output
units remain managed solely by this verified storage eviction after the worktree
is gone. The expiration policy itself is unchanged.

After migration and successful own-process inspection:

```bash
spin ops install storage-tier --dry-run
spin ops install storage-tier
spin ops on storage-tier
spin ops status storage-tier
spin ops logs storage-tier --lines 100
spin ops off storage-tier
```

The user timer runs daily at 05:30 Asia/Tokyo with up to 15 minutes of jitter.
Failures are recorded only in the service journal and `audit.jsonl`; there is no
notification helper or `OnFailure`. Host validation must confirm inspection of
the execution user's processes. Other UIDs are outside the agreed safety scope.
There is no ordering dependency on the system mount service; each operation checks
the mount itself. Installation only links units. Enablement is a separate deployment step.

## Migration and validation

1. Coordinate a pause in producers, queue admissions and consumers holding old paths.
   Confirm the user unit can inspect its own UID's dumpable consumers before
   enabling deletion. Confirm nondumpable exclusions and their recorded evidence.
   Unreadable capability-bearing systemd or another unproven case remains an
   error; do not ignore it or change permissions/stop sessions implicitly.
2. Inventory E: units and active/queued references, including indirect config defaults,
   old hardlink aliases, E:-absolute command paths and pre-existing D: copies. Decide
   the needed version/run units. Hydrate all `ckpt/` and those units only.
3. Classify main-output aliases as references to main outputs. For independent
   old E:`.claude/worktrees/<name>/outputs`, rename the root to
   E:`.worktree-outputs/<name>` while consumers are paused. Refuse occupied
   destinations and compare any existing copies explicitly; do not merge trees.
   The scan's `legacy_outputs` gives the mapping. Re-index at the new canonical
   path: old manifest paths/ctime cannot be reused. This is a second-stage mutation.
   Bootstrap E:-only manifests with `index`, one explicit stable unit at a time.
   At 80–150 MiB/s, one sequential pass over 432 GiB takes about 49–92 minutes;
   small-file seeks can take longer. This is a planning assumption, not a measured
   full-disk benchmark. Indexing reads E: once. Hydration then reads E: for hashing
   and copying, and D: for verification. Do not infer HDD speed from warm tiny probes.
4. Keep original symlink entries under a migration record. Detach links by renaming
   the **link itself**, never its target. Create empty real D: `data/` and `outputs/`
   roots; leave `ckpt/` absent until whole-unit hydration publishes it. Detach each
   affected worktree's output link. Link independent outputs to the persistent D:
   root; link main-output aliases to D: main outputs. Hydrate into canonical paths.
   Hydrate the chosen units, verify contents and a representative hardlink operation,
   then activate input diagnostics. Existing D: copies need explicit comparison;
   hydrate never merges into them.
5. Update affected active/queued commands in their owning sessions. Run a read-only
   cycle, then a selected disposable-unit mirror/evict/hydrate round trip. Compare
   `statvfs` free space and audit evidence; external hardlinks may prevent reclamation.
   Confirm consumer loading, own-UID inspection and failure journal/audit records
   before enabling the timer. Queue state and other UIDs do not protect units.
   Roll back by restoring saved symlink entries only while consumers are paused;
   preserve new D: writes separately. Root VHDX relocation/compaction is outside scope.

Validation uses temporary real files and simulated mount/process inputs in
`tests/integration/agents_ops/test_storage_tier.py`; native FD and hardlink behavior
is also exercised. CLI/worktree and unit-install contracts are in
`tests/e2e/agents_ops/test_storage_tier_cli.py` and the existing ops-control tests.
Common input-boundary diagnostics extend the existing configuration path tests.
