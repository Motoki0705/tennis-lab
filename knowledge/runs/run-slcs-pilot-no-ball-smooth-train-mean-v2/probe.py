"""CPU postprocessing of a saved four-condition evaluation; no model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tasks.slcs.configuration import SLCSEvaluationConfig
from src.tasks.slcs.evaluation.ball_baseline import TrainBallMean
from src.tasks.slcs.evaluation.comparison import CONDITIONS
from src.utils.checksum import dual_sha256
from src.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation_root.resolve(strict=True)
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    save_json({"status": "running"}, output / "status.json")
    try:
        runtime = SLCSEvaluationConfig.from_config(
            OmegaConf.load(root / "evaluation_config.yaml")
        )
        baseline = TrainBallMean.fit(runtime.data)
        expected = baseline.expected_labels("val")
        domains = json.loads(
            (root / "val/comparison/comparison.json").read_text()
        )["domain_mapping"]
        results = {}
        inputs = [root / "evaluation_config.yaml", root / "selection.json"]
        for mode in CONDITIONS:
            directory = root / "val" / mode
            metrics = json.loads((directory / "metrics.json").read_text())
            with np.load(directory / "eval_arrays.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            results[mode] = baseline.compare(
                arrays,
                expected=expected,
                split="val",
                domains=domains,
                headline_error_m=metrics.get("ball_position_error_m"),
            )
            inputs.extend([directory / "metrics.json", directory / "eval_arrays.npz"])
        save_json(baseline.fit_report, output / "fit.json")
        save_json(results, output / "comparisons.json")
        save_json(
            {
                "status": "done",
                "evaluation_root": str(root),
                "input_sha256": {str(path): dual_sha256(path) for path in inputs},
            },
            output / "status.json",
        )
        print(json.dumps({mode: result["rows"][0] for mode, result in results.items()}))
    except Exception as error:
        save_json(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            output / "status.json",
        )
        raise


if __name__ == "__main__":
    main()
