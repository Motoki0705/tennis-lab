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
| Resident inputs | Real directories declared in `resident_units` | Same relative paths | Never evicted |
| Unmanaged E: references | Explicit `unmanaged_data_links` symlinks | Original files/directories | Never selected as units |
| Main `outputs/` | Real directory | Same relative paths | Task/purpose/experiment/run-id |
| Worktree `outputs/` | Link to D:`<main>/.worktree-outputs/<name>/` | `.worktree-outputs/<name>/...` | Task/purpose/experiment/run-id |
| Existing worktree alias of main `outputs/` | Link to main `outputs/` | Main `outputs/...` | No independent unit |
| Main `ckpt/` | Real container; hydrate each child and root files | Same child/file units | Never evicted |
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

## Resident inputs and unmanaged E: references

`resident_units` declares permanent D: inputs: ACCAD, samples, smplh, smplx and
both `data/tennis_multiview/{raw,processed}/meiji_4cam`. The processed directory
includes `projects.json`. They use the existing full SHA-256 index/mirror/verify/
hydrate protocol as `resident` units, but `evict` always keeps them and the daily
cycle never evicts them. Their payloads and ancestors must be real directories,
with the same no-symlink/mount checks as other units. These declarations do not
make the rest of tennis_multiview managed datasets; that is a separate task.

`unmanaged_data_links` explicitly lists allowed D: references, including the
other raw/processed tennis_multiview entries and the two legacy top-level trees.
The tennis_multiview/raw/processed containers are real D: directories: only their
listed nonresident children link to E:. The archive file `meiji_3cam.tar` is an
allowed file reference. The exact list lives in [config.toml](config.toml).
Declarations cannot duplicate or overlap dataset/resident/reference boundaries.
Omitting the two new keys keeps compatibility with older custom configurations.

Each present reference must be a symlink whose text is the **absolute, exact**
`Config.mirror / relative_path`; relative or alternate spellings are rejected.
The target must exist as a real regular file/directory on the configured E:
mount, without symlink ancestors or nested mounts. Unknown symlinks encountered
in data's declared containers, wrong targets, dangling links and real directories
at reference paths fail closed, including explicit unrelated-unit commands and
`activate`. Missing references are allowed while setting up the layout. E: mount
validation happens first; missing E: never means an empty external tree.

References remain `unmanaged`, are never traversed for unit discovery, and cannot
be explicitly mirrored/evicted/hydrated as units. The original E: unmanaged-size
metadata inventory is retained. Command output and write-operation audit records
include `unmanaged_links_D`. Reference failures include observations on stderr.

Linux `readlinkat(fd, "")` on an `O_PATH|O_NOFOLLOW` descriptor also updates an old
symlink's atime on this WSL kernel; there is no ordinary no-atime link-text read.
The validator pins the parent and link, reads each allowed link **once per command**,
and records `atime_before_ns`, `atime_after_ns`, `atime_changed` and the method.
Dry-runs still create no state/copies, but may have this recorded atime effect on
these unmanaged references. No link target or payload is rewritten, and managed
payload symlinks remain forbidden before reading their text.

Initial migration leaves all previous outputs on E: and starts D: output roots
empty. Copy/hash resident inputs, create the declared external references, and
pause/restart Clip Studio around the switch; verify both raw videos and processed
`projects.json` through their original data-relative paths on D:. Capacity acceptance
uses Linux filesystem free space; record Windows D: separately. VHDX sparse/compact
operations remain outside this migration.

## Checkpoint units and CI assets

Each `ckpt/<top-level-directory>` is an independent `checkpoint` unit. Direct
regular files form the separate `checkpoint-files` unit named `ckpt` (currently
`README.md`); a root-file argument selects that projection, not all child units.
Both kinds are mirror/verify/hydrate-only and are never evicted. The E: `ckpt/`
container inode is never exchanged or removed. Explicit whole-root checkpoint
operations from older code are rejected.

If a root-file manifest exists, hydrate/mirror/verify/index compare both its saved
metadata and SHA-256 against E: before any publication or evidence update. Explicit
index does not reset a mismatch. Legacy whole-ckpt migration also compares the
recorded direct-file subset before archiving its old manifest.

Root-file publication stages and hashes only direct regular files, then publishes
one file at a time. Replaced or removed source files are retained under the target
tier's `.storage-tier/checkpoint-file-history/<transaction>/`; no automatic cleanup
of that history is provided. Child directories are untouched. Publication is atomic
per file, not across the file set. The durable D: `checkpoint-file-operations/`
journal lists verified expected/copied entries, previous entries/manifest and publication
progress. An unfinished operation blocks the projection: pause writers and run
`recover` as described in [Journals and recovery](#journals-and-recovery). Keep the journal
and retained history; moving the journal aside and reindexing is not a recovery procedure.
A mismatch never authorizes deleting the archived inode. Root-directory timestamps
are excluded from this projection's metadata identity; root device/inode/mode and
all direct-file metadata/hashes are still checked. Container modes are preserved independently
on D: and E:, and are not compared as root-file payload.

When migrating a former whole-`ckpt` manifest, run explicit `index ckpt` first.
If indexing was interrupted, retry `index ckpt`; an existing legacy archive is
accepted only when its full contents equal the current old manifest. A conflict
fails with both copies retained.
It preserves the old manifest under E:`.storage-tier/legacy-manifests/` before
recording root files. Then index every `ckpt/<directory>` reported by scan. Do not
reuse the old whole-tree hashes for child units. Create a real D: `ckpt/` container
and hydrate root files and each child independently; E: `ckpt/` remains in place.

Mount-reference inspection applies only to destructive eviction, old-mirror
retirement (including its publication check), and verified stage discard. Read-only
index/verify/mirror/hydrate inspections retain own-UID fd/cwd/mmap checks and
before/after snapshots; they do not stat bind targets. The audit records
`mount_scope.inspection_enabled`. Exact/internal bind roots still fail closed on
deletion, regardless of read-only mount options. A directory `index --dry-run`
still skips quiet; the root-file projection performs quiet without mount checks.

The runner installer derives asset sources from the same `Config.load(...).mirror`
rule and configuration, never from repo `data/ckpt` symlink resolution. CI binds
E: `data/` and `ckpt/`; only already mirrored units are visible. A new D:-only unit
appears after successful mirroring (normally the daily cycle). The runtime requires the configured E: mountpoint and re-reads the source mount on every start (device IDs may change on reboot) and checks
FSROOT **and device** against the configured source's actual filesystem-relative
root, not its `/mnt/...` spelling; existing incorrect or writable mounts fail.

Finish all non-sudo migration and acceptance steps first. The current E: binds
remain valid through the D: cutover. The parent then notifies the human and
coordinates a CI/runner-queue maintenance window, waiting for jobs to finish
(stopping an active queue terminates its jobs). Run from the reviewed checkout;
the installer also supports linked worktrees and derives the main repo from Git:

```bash
runner_service="$(sudo cat /opt/actions-runner/.service)"
sudo systemctl stop "$runner_service" tennis-lab-training-queue.service
sudo bash scripts/github_actions/install_self_hosted_runner.sh
sudo /opt/tennis-lab-actions/bin/mount_assets.sh start
# Migration and non-sudo acceptance are already complete; resume normal service.
sudo systemctl start tennis-lab-training-queue.service "$runner_service"
```

The installer updates `/opt` helpers/config and system unit dependencies on
`tennis-data-mount.service`, reloads systemd, and starts/enables services. It also
runs its existing package/runtime provisioning, so allow download time. The current
binds already resolve to E:, so their FSROOT/device do not change and no unmount is
needed before cutover when validation passes. Reinstallation fixes future starts
after the repo paths become D:; it does not by itself replace an active bind.
This stage implements the procedure only: no sudo or unit operations are performed.

## CLI

Run from the checkout containing the desired implementation. The finite
[checkout launcher](../lib/README.md#既存cliと新規jobの接続) is mandatory, including dry-runs.

A linked checkout can supply the code/config while `--repo` selects its main
checkout, without updating that checkout:

```bash
export PYTHONPATH="$PWD/.agents/ops" PYTHONDONTWRITEBYTECODE=1
R="$(git rev-parse --path-format=absolute --git-common-dir)"
R="$(dirname "$R")"
S=(.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier --repo "$R")
"${S[@]}" scan
```

The user service still imports the main checkout. Before enabling it, its deployed
code/config must include the reviewed resident/reference declarations; a successful
linked-checkout CLI does not upgrade the service. Preserve others' dirty main files
and coordinate deployment with their owner.

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
are outside inspection. A separate `permitted-capabilities` exclusion assumes
the same user namespace as an operational precondition. It requires the target's
`CapPrm` to contain a bit absent from `/proc/self/status` `CapPrm`, EACCES/EPERM
for both maps and cwd, and stable PID/UID/capabilities. The fd inventory may remain
readable. Count, comm, both masks and the namespace assumption are recorded in
`process_scope.excluded_capabilities`; neither rule uses process names.
This follows the capability subset check in
[commoncap](https://github.com/torvalds/linux/blob/v6.6/security/commoncap.c).
Training/data-reading scripts must run dumpable and inspectable. Unknown UID,
unexpected I/O failures and denied processes without either proof still fail closed.
Namespace identity is always recorded as `unverified`; the handle is not opened
because `/proc/<pid>/ns/user` uses the same ptrace check
([kernel namespace implementation](https://github.com/torvalds/linux/blob/v6.6/fs/proc/namespaces.c)).
As an auxiliary check, read target/self `uid_map` before and after the probes.
Different readable mappings or malformed metadata fail closed. Unreadable mappings
are nonfatal and record their errors; matching mappings record `matched` without
claiming namespace identity. Both observations remain in the audit evidence.
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
Removal pins the root and consumes the full verified entry plan, including file
hashes and atimes. Each entry is atomically claimed into a private sibling retained
directory, then checked through that namespace before unlink/rmdir. A stale writer
recreating the original name cannot make that new entry the unlink target. Mismatched
claims are restored without overwrite or retained for recovery. Internal hardlink
ctime/link-count changes caused by our own claims are tracked; bytes are still hashed.
Every regular file must acquire a Linux `F_SETLEASE(F_WRLCK)` lease on its claimed
read-only FD before final identity/metadata/SHA-256 verification. Existing open FDs,
ownership/permission errors, unsupported leases, or disabled `/proc/sys/fs/leases-enable`
stop removal without an unlocked fallback. The lease covers the inode, including
external hardlink names. New opens request a break and cannot write while the lease
holds. `F_GETLEASE == F_WRLCK` is required immediately before unlink and checked again
after unlink and immediately before `F_UNLCK`, after all fsync/journal work.
A pending break before unlink restores/retains the entry;
a break observed after unlink is an incident: journal/audit record the phase and the
confirmed removal and the operation fails. SIGIO is blocked, directed to the calling
thread with Linux `F_SETOWN_EX`, and collected before restoring that thread's signal mask.
Any collected break notification also fails the operation, including a break between
the final `F_GETLEASE` and `F_UNLCK`; it is never silently drained as success.
The CLI reserves that thread's SIGIO while holding a lease.

Usage inspection matches FD device/inode identities, including existing `O_PATH` FDs.
On destructive paths, for inspectable same-UID processes it also reads `mountinfo`. The unit's path inside
its filesystem comes from the tool's own namespace: pin the snapshot root, match its
FD mount ID to our mount root/point, and combine that root with the relative path.
This works for quarantine, retained and stage names too. Only matching-device mounts
whose root equals or lies **inside** that filesystem unit path are statted through
`/proc/<pid>/root` in the consumer's view. Ancestor and outside/sibling roots are
excluded without stat; component boundaries distinguish `data/a` from `data/ab`.
`process_scope.mount_scope` audits the filesystem unit, checked roots, excluded
ancestor roots and excluded outside roots (observations per process, not unique mounts).
A
matching mountpoint inode is busy, including a file bind visible only in another
mount namespace. Metadata/read permission failures do not become an empty inventory;
the existing process-exclusion rules still apply. An unclassifiable root such as
`//deleted`, or an in-scope mountpoint that cannot be resolved/read, fails inspection. If the
visible inode does not belong to the verified unit, inspection also fails as unknown:
an overmount may hide a still-live lower file bind. Ancestor/outside roots remain excluded.
An inaccessible ancestor/outside bind, such as a runner's read-only data view, is not
an in-scope inode reference and does not fail inspection. A root equal to the unit
remains in scope even when read-only. Namespaces without an
inspectable member process cannot be inventoried and remain outside this coverage.

Qualification on WSL2 Linux 6.6 confirmed that both an `O_PATH` reference and a private
file bind allow a write lease at `nlink=1`. After unlink and lease release, reopening
through `/proc/self/fd/N` or the bind alias can write the same inode at `nlink=0`.
Existing references are therefore detected before removal, but a consumer must not
acquire **new O_PATH/proc-FD/bind references after usage inspection**, or enumerate or
manipulate private random token names. Inspection, lease checks and unlink are separate
operations; this race interval includes verification/fsync work and has no microsecond
bound. A newly retained reference can receive an already-unlinked inode after release;
its later writes may have no recoverable name. Detected breaks are incidents, not
successful eviction. The reproducible probes are
`test_opath_reference_is_busy_and_can_reopen_an_unlinked_inode` and
`test_external_file_bind_reference_is_inspected_and_survives_unlink` in the storage tests.
With `nlink > 1`, another hardlink normally retains the payload, unless that name is
also removed externally. Leases do not lock directory entries against same-UID changes;
identity/link-count checks reject detected replacements, not hostile namespace control.
This adds D:/E: reads and per-entry journal/directory fsyncs; full-dataset removal timing
remains a stage-two measurement. The implementation is in `removal.py`.
A replaced quarantine is neither deleted nor moved back as our unit.
Every component from the repository/E: mount anchor through the unit root is opened
with no-follow FDs and checked by mount ID, including same-device bind mounts.
Checkpoint directory device/inode identities independently reject aliases as dataset
or run units. Ordinary file hardlinks to checkpoints do not delete checkpoint names.
Mount IDs are runtime evidence and are not persisted as cross-boot identities.
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
Read-only `index --dry-run` reports global process inspection errors but never certifies
a manifest or authorizes deletion; for directory units it hashes the selected tiny tree without
unit `quiet` checks (the checkpoint root-file projection also inspects fd/cwd/mmap). It is therefore not evidence that E: mount-reference
inspection passed. Host qualification must exercise both sides through
`verify --dry-run` and record successful E: `quiet` calls, together with unchanged
metadata. Before placement migration, distinguish a read-only existing-pair
qualification mapping from the normal CLI, which still refuses the pending layout.
Durable deletion transactions live in `transactions/` beside the audit.
`removals/<transaction>.jsonl` durably records claim/removal intents, claimed
identities, verified digests and confirmed removed entries. A crash may leave an
intent without a completion record; this is uncertain, not a confirmed deletion.

```bash
"${S[@]}" recover
```

Recovery restores quarantines with no overwrite and **never resumes deletion**.
An occupied original path or changed identity is an explicit failure. A partially
deleted tree is restored and marked as possibly partial. Retained claims move,
without overwrite, into `recovered-assets/`; full plans remain in
`recovered-transactions/`. A partial/changed tree or retained payload creates a
`recovery-required/` marker that blocks mirror, eviction and cycle for that unit.
It cannot overwrite the complete E: copy during automatic mirroring.

1. Inspect the recovery result, retained assets, journal and full plan. Preserve all
   unmirrored changes; an identity mismatch or occupied destination needs manual inspection.
2. Keep the recovered partial tree outside managed roots, then `hydrate` the complete
   unit into its now-empty D: destination. Alternatively, repair the complete D: tree
   explicitly and run `verify` against E:.
3. Only a successful full SHA-256 hydrate/verify clears the recovery marker. Keep
   retained/unmirrored data until explicitly reconciled; recovery never resumes deletion.

If removal completed before a crash, recovery records
that fact without claiming a measured capacity gain. Any pending transaction blocks
further operations until recovery. For interrupted mirror/hydrate publication, pause
writers and run `recover`; do not remove journals, force index or discard stages.
Directory units persist a `copying` receipt in D:`publications/` **before creating
the stage** and change it to `prepared` only after verifying the copy. Root-file operations
persist verified copies and per-file progress in D:`checkpoint-file-operations/`.
Recovery compares fresh, stable source/destination SHA-256 with that durable intent.
It resumes only verified remaining publication steps, never deletion. For a `prepared`
publication, a changed source, unplanned destination or unverified stage causes failure.
Before issuing new evidence it preserves the exact previous manifest, journal and
observed manifest in D:`publication-evidence/`. Old directory trees move, without
deletion, to the target tier's `.storage-tier/recovered-publications/`; root-file
inodes remain in `checkpoint-file-history/`. Both D:/E: are hashed again before the
new manifest is recorded and the receipt is marked complete. Completed receipts and
archives remain available for inspection. Then run `verify`; ordinary mirror/hydrate
and index still reject mismatches with saved evidence. An operation interrupted before
a verified root-file copy exists can only be aborted if the old destination contents
are intact; it does not issue a new manifest. Unsupported older incomplete journals
remain blocked for manual inspection. Staging leftovers are never silently swept.

An interrupted `copying` operation archives its stage, including incomplete or
unverified bytes, and aborts without issuing a new manifest. Recovery first checks
that the old destination is intact. Retry mirror/hydrate after recovery. A `complete`
receipt that still has its old stage also requires recovery: it writes an `archiving`
intent, preserves that exact old directory in `recovered-publications/`, then completes
the receipt. This works after the manifest was committed but before retirement began,
including receipts written by the earlier implementation. It does not refresh the manifest.

If interruption occurs between creating a retained directory and saving its identity,
recovery pins and records the observed identity **for archival custody only** before
moving it. The record flags this observation; it cannot authorize payload deletion.
An already-recorded identity mismatch remains an error. Atomic JSON replacement may
leave a `.<record>.<uuid>.tmp` file: committed-record enumeration preserves these
regular files and reads only the canonical `.json` records. Other unexpected names,
special files and corrupt committed records remain errors. Every command validates
committed control JSON and reports preserved temporaries in `uncommitted_json`, with
path, byte count and `committed=false`. Only canonical, complete records supply evidence.
A partial manifest, transaction or receipt temporary leaves its previous committed
record intact. Preserve the reported temporaries for inspection.

An interrupted append can leave a torn final line in D:`audit.jsonl` or
`removals/<transaction>.jsonl`. Ordinary commands fail with an explicit journal error.
Run `recover` while writers are paused. Recovery first persists a repair receipt in
D:`.storage-tier/journal-repairs/`, including the original SHA-256 and complete-prefix
length. It saves the entire original journal, including the torn tail, under
`journal-history/`, then atomically publishes only newline-terminated JSON object
lines as the live prefix. It completes the receipt and records `journal_repaired` in
the audit and command result, including the archive path and tail byte count. Repeated
`recover` validates the same archived bytes and resumes an interrupted repair.
A valid JSON object missing its terminal newline also belongs to the preserved tail.
A corrupt interior line or corrupt canonical JSON requires inspection and restoration
of the matching durable record from preserved evidence before retrying `recover`.
Preserve all conflicting records. Publication and checkpoint operation journals use
atomic JSON replacement and the same committed-record validation.

The persistence sequences below are exercised at every recorded filesystem mutation
in the [crash-boundary tests](../../../tests/integration/agents_ops/test_storage_tier_crash_boundaries.py).
The [boundary inventory](../../../tests/support/agents_ops/fixtures/storage-crash-boundaries.json)
includes JSON temporary creation/write/flush/fsync/replacement, copy chunks and
metadata, writable `os.open` flags, `os.write`/`pwrite`/`writev`, stream write/flush,
truncate/ftruncate, rename/claim, removal, journal append and recovery writes.
Consecutive JSON encoder fragments share a representative stream-write boundary;
binary copies retain their first and next chunk boundaries. The
[partial-write tests](../../../tests/integration/agents_ops/test_storage_tier_writes.py)
exercise copies larger than 8 MiB, partially written large JSON, torn appends and
changed repair evidence. An injected existing-file write must fail exact trace coverage. Directory fsync calls share
the preceding mutation's observable process-crash state. These are process-exit tests.
Every interrupted state is mapped to a [recovery sequence](../../../tests/support/agents_ops/fixtures/storage-recovery-boundaries.json).
The [recovery-boundary tests](../../../tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py)
interrupt each distinct sequence again and require canonical payload equality and a
successful unselected daily cycle after the documented repair.

| Operation | Persistence sequence |
|---|---|
| New/update mirror | Copy intent → stage copy → verified copy intent → publish rename → manifest → complete receipt → old-stage retirement |
| Hydrate | Copy intent → stage copy → verified copy intent → publish rename → manifest → complete receipt → recovery-marker clear |
| Evict | Prepared transaction → quarantine rename → moved identity → retained mkdir → verified plan → entry claim/removal journal → removal → audit → complete transaction → transaction unlink |
| Retire/discard stage | Parent publication receipt → prepared removal transaction → retained mkdir → verified plan → entry claim/removal → audit → complete transaction → unlink |
| Checkpoint root files | Prepared journal → history/copy → publishing proof → each exchange/progress → each old-file archival/progress → manifest → complete journal → audit |
| JSONL repair | Durable receipt with original digest/prefix → preserve full journal → atomically publish complete prefix → complete receipt → audit; interrupted repair resumes against the preserved original |
| Legacy checkpoint index | Verify recorded direct files → archive old manifest → publish root-file manifest → audit; retry index accepts the matching prior archive |
| Recover | Restore or preserve tree → establish archival custody → archive tree/entries → save prior evidence/plan → mark partial recovery or verify publication → complete receipt/clear active transaction |

After any interrupted recovery, run `recover` again. Preserve the reported archives.
For a partial D: tree follow the full hydrate/verify procedure above, then run the
unselected daily `cycle` after activation. Recovery never resumes entry deletion.

After `RENAME_EXCHANGE`, the old E: tree is removed only against its pre-exchange
hashed snapshot through the same claimed-entry, lease and durable journal path.
Changed/added old payload is retained and mirroring fails. Recovery archives old trees
and claims under E:`<mirror-repo>/.storage-tier/recovered-trees/` and `recovered-assets/`;
plans/journals remain on D:. It never replaces the newly published mirror. Interrupted
retirement creates the unit's recovery marker; preserve/reconcile the archived E-only
changes and run full `verify`/`hydrate` before continuing. Archives are never auto-deleted.
Unpublished mirror/hydrate stages are cleaned only with a complete verified snapshot
through that same removal path. Incomplete/unverified copies initially remain at their
staging name with a `stage_preserved` audit record; receipt-backed recovery archives them.

An older orphan may have no receipt or transaction identifying its unit. `recover`
rescans managed roots and returns failure with `unclaimed storage artifacts` while
such a stage remains. Preserve it explicitly before retrying:

1. Pause its producers and select the exact stage path reported by scan/recover.
2. Create the same tier's `<tier-repo>/.storage-tier/manual-recovery/` with the normal
   mount/no-follow checks. Use `filesystem.rename(stage, archive)` to atomically move
   it to that directory under its original stage name. This uses `RENAME_NOREPLACE`;
   an occupied destination is an error. Record both paths and the directory device/inode.
3. Keep every archived byte for inspection. Run `recover`, retry the original
   mirror/hydrate/index or perform the full verify/partial-tree repair above, then
   check the unselected daily cycle. The orphan-archival test fixes this procedure.

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
   enabling deletion. Confirm nondumpable/capability exclusions, their evidence,
   and the same-user-namespace operational precondition. Different readable uid_map
   values or another unproven core case remain errors. Do not change permissions
   or stop sessions implicitly.
2. Inventory E: units and active/queued references, including indirect config defaults,
   old hardlink aliases, E:-absolute command paths and pre-existing D: copies. Decide
   the needed version/run units. Hydrate all checkpoint children plus the `ckpt` root-file projection and those units only.
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
   roots plus a real `ckpt/` container; hydrate its children and root files separately. Detach each
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
