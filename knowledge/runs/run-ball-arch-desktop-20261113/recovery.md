# Resume a stopped desktop attempt

First inspect the unit/job in `monitoring-handoff.json`, queue state, GPU processes,
`production-FAILED.json`, stage `FAILED.json` and the terminal event. Diagnose the
cause and confirm the old writer is stopped. A second writer is forbidden.
The original tail 5,500 is the migration input only; once desktop checkpoints
exist, recover the newest independently verified desktop generation. Completed
stages are skipped only with their validation and persistence evidence.

Use the exact pushed runtime checkout recorded in `monitoring-handoff.json`.
Its private interpreter is reused; shared `.venv` stays unchanged. On the same
host, create a new output root with hardlinked, SHA-verified immutable inputs.
The commands below run from that checkout (`$SOURCE`) with the private rclone
configuration. Choose a fresh root suffix, never delete or reuse a failed root.

```bash
cd "$SOURCE"
export RCLONE_CONFIG=/home/kamimura/.config/tennis-lab/rclone-drive.conf
OLD_ROOT=/home/kamimura/projects/tennis-lab/.claude/worktrees/ball-arch-desktop/outputs/ball_detection/train/arch_efficiency/desktop-resume-20261113
NEW_ROOT=/home/kamimura/projects/tennis-lab/.claude/worktrees/ball-arch-desktop/outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261113-a2
ADAPTER="$SOURCE/knowledge/runs/run-ball-arch-desktop-20261113"
PY="$OLD_ROOT/env-original/bin/python"
"$PY" "$ADAPTER/bootstrap_recovery.py" --from-root "$OLD_ROOT" --root "$NEW_ROOT"
"$PY" "$ADAPTER/recover.py" --root "$NEW_ROOT" --old-writer-stopped
ATTEMPT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_queue_attempt"])' "$NEW_ROOT/recovery-bootstrap.json")
"$PY" "$ADAPTER/launch.py" --root "$NEW_ROOT" --kind production \
  --session "$CODEX_SESSION_ID" --attempt "$ATTEMPT"
```

`bootstrap_recovery.py` rejects a live old unit, preserves the original storage
baseline, and sets `next_queue_attempt` to the largest previous production
registration attempt plus one. For a later recovery, `OLD_ROOT` is the latest
attempt root. If a same-day queue/unit key is already registered elsewhere,
inspect and preserve it, then choose an integer larger than every registered
`production-aN` attempt; never repurpose an existing key.

`recover.py` downloads and verifies the desktop pointers and original desktop
preflight, then restores each pinned immutable generation. `migrate_resume`
changes only audited store/source paths and derived manifest hashes, compares
model/AdamW/RNG/schedule/sampler recursively, and preserves committed logs.
A zero-update initialization pointer contains no scientific state; the pinned
initializer reproduces it. It never silently falls back from a later desktop
state to the historical 5,500 checkpoint. Scope, shapes, precision and compile
policy remain fixed. Recovery refuses any unexplained state/config difference.

On a new host without these local immutable inputs/private environment, first
repeat verified input acquisition, fixed-snapshot reconstruction and original
runtime/native qualification. `recover.py` then performs the same desktop-state
restoration. Stop on failed full-state reproduction, nonfinite numbers, an
unfit allocator policy, old remote writes, or projected storage growth above
20 GB. No weights-only restart, smaller batch, alternative scope or eager
fallback is automatic. The parent owns monitoring after worker handoff.
