"""Twenty real updates per condition, with independent checkpoint readback/resume."""

import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from protocol import ROOT, read
from serial import CASES, prepare, publish_index, state_path


def main():
    prepare()
    logs = ROOT / "smoke/logs"
    logs.mkdir(parents=True, exist_ok=True)
    proof = {}
    for case in CASES:
        for stop in (10, 20):
            output = ROOT / "smoke" / case
            if (output / "latest.json").exists() and read(output / "latest.json")[
                "update"
            ] >= stop:
                continue
            with (logs / (case + f"-to{stop}.log")).open("ab", buffering=0) as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("serial.py")),
                        "smoke",
                        "--case",
                        case,
                        "--stop",
                        str(stop),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        output = ROOT / "smoke" / case
        rows = [
            __import__("json").loads(x)
            for x in (output / "train.jsonl").read_text().splitlines()
        ]
        base = 10000 if case.startswith("T") else 0
        if (
            len(rows) != 20
            or [r["update"] for r in rows] != list(range(1, 21))
            or [r["sample_position"] for r in rows] != list(range(base + 1, base + 21))
        ):
            raise RuntimeError("Noncontinuous smoke update/sample sequence")
        if not all(
            math.isfinite(r[k])
            for r in rows
            for k in ("loss", "gradient_norm", "learning_rate", "encoder_learning_rate")
        ):
            raise RuntimeError("Nonfinite smoke state")
        saved = torch.load(
            output / "checkpoint-update-00000020.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not saved["optimizer"]["state"] or saved["updates_completed"] != 20:
            raise RuntimeError("Incomplete resumed full checkpoint")
        status = read(state_path(Path("smoke") / case))
        if status["readback"]["status"] != "verified" or status["update"] != 20:
            raise RuntimeError("Smoke is not durably verified")
        proof[case] = dict(
            updates=20,
            resumed_at=10,
            finite=True,
            peak_allocated=max(r["peak_vram_bytes"] for r in rows),
            peak_reserved=max(r["peak_reserved_vram_bytes"] for r in rows),
            draws=[(r["pack_indices"], r["augmentation_sha256"]) for r in rows],
            receipt=status["receipt"],
        )
        publish_index(ROOT / "smoke/progress.json", proof)
    for group in (("T1", "T2"), ("C1", "C2", "C3")):
        if any(proof[c]["draws"] != proof[group[0]]["draws"] for c in group[1:]):
            raise RuntimeError("Architecture changed data/augmentation draws")
    publish_index(
        ROOT / "smoke/COMPLETED.json",
        dict(
            complete=True,
            conditions=proof,
            qualification_only=True,
            main_updates=0,
            controls="CNN unfrozen, global snapshot verified, lazy per-shard rechecks; main recipes unchanged",
        ),
    )


if __name__ == "__main__":
    main()
