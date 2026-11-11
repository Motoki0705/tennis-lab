"""Compare the actual historical seed42 construction to the new first-T64 path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from control import REFERENCE, ROOT, now, read, sha, write  # noqa: E402


def historical(output):
    from src.tasks.ball_detection.checkpoints.temporal import initialize_finetuning
    from src.tasks.ball_detection.models.mdd import MDDTemporalQueryConfig

    run = read(REFERENCE / "query-T32/run.json")
    for name, digest in run["code"]["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(
                "Historical source differs from the actual baseline: " + name
            )
    values = dict(run["recipe"]["model"])
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        values[key] = tuple(values[key])
    path = REFERENCE / "common-dpt-T32/checkpoint-update-00010000.pt"
    torch.manual_seed(42)
    model, receipt = initialize_finetuning(
        path,
        expected_sha256=run["recipe"]["initialization"]["artifact_sha256"],
        parent_update=10000,
        config=MDDTemporalQueryConfig(**values),
    )
    torch.save(
        dict(state=model.state_dict(), rng=torch.get_rng_state(), receipt=receipt),
        output,
    )


def verify(reference_cwd):
    from src.tasks.ball_detection.checkpoints.temporal import initialize_finetuning
    from src.tasks.ball_detection.models.mdd import MDDTemporalQueryConfig

    with tempfile.TemporaryDirectory(prefix="query-initialization-") as temporary:
        reference = Path(temporary) / "historical.pt"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--historical",
                "--output",
                str(reference),
            ],
            cwd=reference_cwd,
            check=True,
        )
        expected = torch.load(reference, map_location="cpu", weights_only=True)
        path = REFERENCE / "common-dpt-T32/checkpoint-update-00010000.pt"
        config = MDDTemporalQueryConfig.load(
            Path("src/tasks/ball_detection/configs/model/mdd_temporal_query.yaml")
        )
        from dataclasses import replace

        audits = []
        for frames in (32, 64):
            torch.manual_seed(42)
            model, receipt = initialize_finetuning(
                path,
                expected_sha256=sha(path),
                parent_update=10000,
                config=replace(config, frames=frames),
            )
            actual = model.state_dict()
            if actual.keys() != expected["state"].keys():
                raise ValueError("Historical/new initial state keys differ")
            for name, tensor in actual.items():
                if not torch.equal(tensor, expected["state"][name]):
                    raise ValueError("Historical/new initial state differs: " + name)
            if not torch.equal(torch.get_rng_state(), expected["rng"]):
                raise ValueError(
                    "The initial Torch RNG differs from the historical construction"
                )
            audits.append(
                dict(
                    frames=frames,
                    exact_tensors=len(actual),
                    new_query_keys=len(receipt["new_keys"]),
                    rng_exact=True,
                    all_weights_exact=True,
                )
            )
        evidence = dict(
            schema="ball_query_initialization_audit.v1",
            verified_at=now(),
            complete=True,
            baseline_code=read(REFERENCE / "query-T32/run.json")["code"],
            parent_sha256=sha(path),
            results=audits,
        )
        write(ROOT / "initialization-audit.json", evidence)
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-cwd",
        type=Path,
        default=Path("/content/tennis-lab/.claude/worktrees/ball-reference-init"),
    )
    args = parser.parse_args()
    if args.historical:
        if args.output is None:
            parser.error("--output required")
        historical(args.output)
    else:
        print(json.dumps(verify(args.reference_cwd)), flush=True)


if __name__ == "__main__":
    main()
