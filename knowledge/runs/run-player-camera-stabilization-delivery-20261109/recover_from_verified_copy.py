"""Recover small immutable files using the completed full-tree Drive verification.

No new Drive inventory: the exact source file hashes are pinned by strict
validation, a successful upload --verify receipt and the saved failed job.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

BASE = Path("outputs/tennis_scene/visualize/player_view/stabilization-comparison")
OPS = BASE / "operations/attempt2"
FULL = BASE / "attempt2/full"
REPAIR = OPS / "resume1"
REF = "d133993421c206af28340f4ffe082c65b70fe7a7"
sys.path.insert(0, str(Path(".agents/skills/tennis-colab/scripts").resolve()))
from colab_core import Session  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path, kind: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, kind).hexdigest()


def main() -> None:
    receipt_path = REPAIR / "static-precopy.json"
    receipt = read(receipt_path)
    assert receipt["status"] == "completed" and receipt["verified"] is True
    assert receipt["action"] == "upload" and receipt["dry_run"] is False
    assert receipt["source"] == "/content/tennis-lab/" + str(FULL)
    assert receipt["destination"] == "gdrive:tennis_lab/" + str(FULL)
    strict_path = REPAIR / "persist/validated-full.json"
    strict = read(strict_path)
    assert strict["source_ref"] == REF and strict["source_pngs"] == 1920
    assert strict["rendered_or_modified_images"] == 0
    expected = {r["path"]: r for r in strict["files"]}
    assert len(expected) == 68
    assert sum(r["bytes"] for r in expected.values()) == 1065261659
    records = read(OPS / "full-recovery-vm-file-records.json")
    assert len(records) == 39 and len({r["path"] for r in records}) == 39
    assert len([r for r in records if r["path"].endswith(".mp4")]) == 14
    for row in records:
        assert row == expected[row["path"]], row["path"]
        assert row["bytes"] <= 16 * 1024 * 1024
    session = Session("player-view-stabilization-20261109-a2")
    native = session.rpc("status", job_id="full-a2")
    job = next(j for j in native["jobs"] if j["job_id"] == "full-a2")
    assert job["status"] == "failed" and job["returncode"] == -15
    assert job["failure_stage"] == "periodic_save" and job["drive_saved_at"]
    (OPS / "full-recovery-terminal-status.json").write_text(
        json.dumps(native, indent=2) + "\n"
    )
    recovered = []
    for row in records:
        local = FULL / row["path"]
        if not local.exists():
            transfer = session.download("/content/tennis-lab/" + str(local), local)
            assert transfer["sha256"] == row["sha256"]
        assert local.stat().st_size == row["bytes"]
        assert digest(local) == row["sha256"] and digest(local, "md5") == row["md5"]
        recovered.append(
            {
                **row,
                "drive_id": None,
                "method": "ssh_sha256_after_exact_full_tree_drive_verify",
            }
        )
    summary = read(FULL / "manifest.json")
    assert (
        digest(FULL / "manifest.json")
        == read(REPAIR / "recovery-handoff.json")["full_manifest_sha256"]
    )
    assert summary["git_ref"] == REF and summary["phase"] == "full"
    assert summary["source_frames"] == list(range(240))
    assert summary["archived_source_pngs"] == 1920
    archives = [r for r in strict["files"] if r["path"].endswith(".zip")]
    assert len(archives) == 8
    assert not list(FULL.glob("**/*.zip")) and not list(FULL.glob("**/*.png"))
    result = {
        "schema_version": 1,
        "recovered_at": dt.datetime.now(dt.UTC).isoformat(),
        "colab_job_id": "full-a2",
        "drive_saved_at": job["drive_saved_at"],
        "source_ref": REF,
        "mp4s_recovered": 14,
        "render_audits_recovered": 8,
        "archives_retained_on_drive": archives,
        "archived_source_pngs": 1920,
        "raw_pngs_or_zips_recovered": 0,
        "original_execution_status": "failed",
        "original_execution_returncode": -15,
        "files": recovered,
        "drive_verification": {
            "method": "completed_exact_full_tree_upload_verify",
            "receipt": str(receipt_path),
            "receipt_sha256": digest(receipt_path),
            "source_hash_manifest": str(strict_path),
            "source_hash_manifest_sha256": digest(strict_path),
            "files": 68,
            "bytes": 1065261659,
        },
        "redundant_inventory_attempt": "failed before any download; original log retained",
        "human_acceptance": "pending",
    }
    (OPS / "full-recovery-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": "full_recovery_hash_verified",
                "mp4s": 14,
                "audits": 8,
                "archive_count_on_drive": 8,
                "fresh_drive_inventory": False,
            }
        )
    )


if __name__ == "__main__":
    main()
