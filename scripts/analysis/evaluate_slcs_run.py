"""Evaluate a training run's validation-selected checkpoint under paired inputs.

Example (from repository root):
    .venv/bin/python -m scripts.analysis.evaluate_slcs_run \
        --output-root /absolute/outputs --training-run slcs/train/experiment/run-id \
        --output slcs/evaluate/experiment/run-id --domain-prefix video_=meiji \
        --default-domain broadcast

Training and output fragments are OUTPUT-relative. Test is opt-in via
--splits val test and is never used for checkpoint selection. CUDA execution
requires the shared training queue environment. Existing outputs are rejected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.slcs.evaluation.run_evaluation import evaluate_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run",
        type=Path,
        required=True,
        help="OUTPUT-relative training run containing config.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--last-checkpoint",
        help="Training-run-relative last.ckpt selecting the retained validation callback (required when multiple exist)",
    )
    parser.add_argument(
        "--output", required=True, help="slcs/evaluate/<experiment>/<run-id>"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=["val"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gap-no-rgb",
        action="store_true",
        help="Add detector_gap_no_rgb and a paired RGB comparison within detector gaps",
    )
    parser.add_argument(
        "--ball-train-mean",
        action="store_true",
        help="Compare ball errors to a confidence-weighted train-only constant (CPU label view)",
    )
    parser.add_argument(
        "--domain-prefix",
        action="append",
        required=True,
        help="PREFIX=DOMAIN, e.g. video_=meiji",
    )
    parser.add_argument("--default-domain", required=True)
    args = parser.parse_args()
    prefixes = []
    for rule in args.domain_prefix:
        prefix, separator, domain = rule.partition("=")
        if not prefix or not separator or not domain:
            parser.error("--domain-prefix must be a nonempty PREFIX=DOMAIN")
        prefixes.append((prefix, domain))
    print(
        evaluate_training_run(
            training_run=args.training_run,
            output_root=args.output_root,
            output=args.output,
            splits=args.splits,
            device=args.device,
            batch_size=args.batch_size,
            domain_prefixes=prefixes,
            default_domain=args.default_domain,
            ball_train_mean=args.ball_train_mean,
            gap_no_rgb=args.gap_no_rgb,
            last_checkpoint=args.last_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
