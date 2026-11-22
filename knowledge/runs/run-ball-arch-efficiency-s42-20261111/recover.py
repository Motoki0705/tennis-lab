"""Recover verified run metadata/native build before serial checkpoint replay.

Run only after fetching/verifying this VM's fixed inputs. The previous VM must
be gone; a second live VM must never resume the same logical campaign.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from protocol import CONFIG, DRIVE, REMOTE, ROOT

from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256


def copy_immutable(source, destination):
    if destination.exists():
        if dual_sha256(source) != dual_sha256(destination):
            raise RuntimeError(
                "Local/remote recovery metadata differs: " + str(destination)
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if dual_sha256(source) != dual_sha256(destination):
        raise RuntimeError("Recovery readback differs")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        CONFIG.write_text(
            "[mounted_drive]\ntype = alias\nremote = " + str(DRIVE) + "\n"
        )
        CONFIG.chmod(0o600)
    for name in (
        "plan.json",
        "execution.json",
        "level-decision.json",
        "preparation-receipt.json",
        "smoke/COMPLETED.json",
        "benchmark-receipt.json",
        "FAILED.json",
    ):
        source = DRIVE / REMOTE / name
        if source.exists():
            copy_immutable(source, ROOT / name)
    receipt = json.loads((ROOT / "preparation-receipt.json").read_text())
    with tempfile.TemporaryDirectory(
        prefix="preparation-recovery-", dir=ROOT
    ) as temporary:
        restored = Path(temporary) / "verified"
        store = RcloneSnapshotStore(
            restored,
            remote="mounted_drive",
            remote_root=REMOTE + "/preparation",
            config_path=CONFIG,
        )
        store.restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
        for name in (
            "msda-build/lib/MultiScaleDeformableAttention.so",
            "msda-build/build.json",
            "msda-build-manifest.json",
        ):
            copy_immutable(restored / name, ROOT / name)
    print(
        "Metadata and native binary verified. Verify the native op, then enqueue train; stage restoration is automatic."
    )


if __name__ == "__main__":
    main()
