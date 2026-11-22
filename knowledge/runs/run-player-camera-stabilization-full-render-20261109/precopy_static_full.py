"""Bounded verified Drive pre-copy of immutable, strictly validated full outputs."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison")
OPS = BASE / "operations/attempt2"
REPAIR = OPS / "resume1"
FULL = BASE / "attempt2/full"


def read(path):
    return json.loads(path.read_text())


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    operation = read(OPS / "gpu-operation.json")
    assert operation["deadline_utc"] == "2026-11-09T15:22:50.337862Z"
    deadline = dt.datetime.fromisoformat(operation["command_deadline_utc"].replace("Z", "+00:00"))
    remaining = int((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds())
    assert remaining > 180
    verified = read(REPAIR / "persist/validated-full.json")
    assert verified["source_pngs"] == 1920 and verified["archives"] == 8 and verified["mp4s"] == 14
    expected = {r["path"]: r for r in verified["files"]}
    actual = {p.relative_to(FULL).as_posix() for p in FULL.rglob("*") if p.is_file()}
    assert actual == expected.keys(), "Artifact set changed after validation"
    for name, row in expected.items():
        path = FULL / name
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], name
    session = read(Path("/content/.tennis-colab/player-view-stabilization-20261109-a2/session.json"))
    environment = dict(os.environ, RCLONE_CONFIG=session["rclone_config"])
    remote = "gdrive:tennis_lab/" + str(FULL)
    # Explicit cached preflight: the last bounded listing succeeded, and no writer
    # remains. Avoid a redundant quota-consuming lookup; upload still verifies.
    observation = read(REPAIR / "listing-retry.json")
    assert observation["returncode"] == 0
    observed = dt.datetime.fromisoformat(observation["observed_at"])
    assert (dt.datetime.now(dt.timezone.utc) - observed).total_seconds() < 600
    cached = REPAIR / "listing-retry.stdout.json"
    before = read(cached)
    (REPAIR / "precopy-drive-before.json").write_text(json.dumps(before, indent=2) + "\n")
    (REPAIR / "precopy-preflight-cache.json").write_text(json.dumps({"source": str(cached), "sha256": digest(cached), "observed_at": observation["observed_at"], "reason": "Original renderer and publisher stopped; no Drive writes occurred after successful listing. Use explicit immutable conflict check and fresh upload --verify."}, indent=2) + "\n")
    for row in before:
        name = row["Path"]
        if row.get("IsDir") or name not in expected or name == "checkpoint.json":
            continue
        assert row["Size"] == expected[name]["bytes"] and row.get("Hashes", {}).get("md5") == expected[name]["md5"], f"Existing immutable Drive artifact differs: {name}"
    remaining = int((deadline - dt.datetime.now(dt.timezone.utc)).total_seconds())
    overall_timeout = min(1800, remaining - 40)
    assert overall_timeout > 60
    command = [".venv/bin/python", ".agents/skills/tennis-drive/scripts/drive.py", "--timeout-seconds", str(min(1200, overall_timeout - 20)), "upload", str(FULL), str(FULL), "--overwrite", "--verify"]
    record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": command, "overall_timeout_seconds": overall_timeout, "source_files": len(expected), "source_bytes": sum(r["bytes"] for r in expected.values()), "command_deadline_utc": operation["command_deadline_utc"], "hard_deadline_utc": operation["deadline_utc"], "mutable_progress_record_allowlist": ["checkpoint.json"], "immutable_drive_conflicts": 0}
    (REPAIR / "static-precopy-command.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    with (REPAIR / "static-precopy.json").open("w") as output, (REPAIR / "static-precopy.stderr.log").open("w") as error:
        process = subprocess.run(command, env=environment, stdout=output, stderr=error, timeout=overall_timeout)
    result = {"returncode": process.returncode, "elapsed_seconds": time.monotonic() - started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    (REPAIR / "static-precopy-result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert process.returncode == 0, result
    receipt = read(REPAIR / "static-precopy.json")
    assert receipt["status"] == "completed" and receipt["verified"] is True
    print("STATIC_PRECOPY_VERIFIED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
