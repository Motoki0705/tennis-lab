"""Execute the committed comparison list sequentially using the foundation runner."""

import argparse
import json
import platform
import resource
import subprocess
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
from common import (
    BUNDLE_SHA,
    HERE,
    REFERENCE_SHA,
    ROOT,
    code_fingerprint,
    git_head,
    load_script,
    prior_run,
    write_json,
)
from methods import make_method

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import verify_baseline
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.evaluation.ball3d.scoring import annotation_oracle
from src.utils.checksum import dual_sha256


def differences(actual, expected, path="result"):
    """Exact comparison of every result field, not runtimes/code provenance."""
    result = []
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected)
        ):
            value = {
                "field": path,
                "actual_shape": actual.shape,
                "prior_shape": expected.shape,
                "actual_dtype": str(actual.dtype),
                "prior_dtype": str(expected.dtype),
            }
            if actual.shape == expected.shape and np.issubdtype(
                actual.dtype, np.number
            ):
                value["max_absolute_difference"] = float(
                    np.max(
                        np.abs(actual.astype(float) - expected.astype(float)), initial=0
                    )
                )
            result.append(value)
    elif is_dataclass(actual) and is_dataclass(expected):
        for field in fields(actual):
            result.extend(
                differences(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    path + "." + field.name,
                )
            )
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            result.append(
                {
                    "field": path,
                    "actual_keys": sorted(actual),
                    "prior_keys": sorted(expected),
                }
            )
        for key in actual.keys() & expected.keys():
            result.extend(differences(actual[key], expected[key], path + "." + key))
    elif isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            result.append({"field": path, "different_lengths": True})
        else:
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                result.extend(differences(a, b, f"{path}[{i}]"))
    elif actual != expected:
        result.append({"field": path, "actual": actual, "prior": expected})
    return result


def check_prior(row, values, plan, root, output):
    if "prior" not in row:
        return {"name": row["name"], "status": "new_combination_no_saved_prediction"}
    source = row["prior"]
    report = {"name": row["name"], "source": source, "folds": []}
    for run in values:
        prior = prior_run(root, source["directory"], f"{source['prefix']}/{run.key}")
        if prior.bundle_sha256 != BUNDLE_SHA:
            raise ValueError("Prior result input digest differs")
        delta = differences(run.result, prior.result)
        report["folds"].append(
            {
                "fold": run.key,
                "bitwise_equal_all_result_fields": not delta,
                "differences": delta,
                "prior_code_commit": prior.code_commit,
                "prior_manifest_sha256": dual_sha256(
                    root
                    / source["directory"]
                    / source["prefix"]
                    / run.key
                    / "result.json"
                ),
            }
        )
    report["status"] = (
        "exact_all_result_fields"
        if all(v["bitwise_equal_all_result_fields"] for v in report["folds"])
        else "mismatch_stop_for_investigation"
    )
    write_json(output / "prior_checks" / (row["name"] + ".json"), report)
    if report["status"] != "exact_all_result_fields":
        raise ValueError(
            f"Integrated replay changed {row['name']}; see prior_checks before continuing"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "execution-plan.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="New reproduction directory; default is the frozen destination",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checksum-verified completed rows only, no source replacement",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = args.output or Path(plan["output"])
    root = Path(plan["inputs"])
    if (output.exists() and not args.resume) or (not output.exists() and args.resume):
        raise ValueError(
            "Use a new directory, or explicit --resume for an interrupted comparison"
        )
    if output.resolve() == root.resolve() or any(
        output.resolve().is_relative_to((root / p).resolve())
        for p in (
            "fixed_input_bundle_v1",
            "human_annotations",
            *plan["source_manifest_sha256"],
        )
    ):
        raise ValueError(
            "Output would overwrite immutable input or previous experiment"
        )
    if code_fingerprint() != plan["code_fingerprint"]:
        raise ValueError("Executable bytes differ from the committed execution list")
    # A separate plan-only commit is intentional; executable bytes are pinned above.
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(args.plan.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", str(HERE.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ):
        raise ValueError(
            "Commit all executable code and the execution list before replay"
        )
    output.mkdir(parents=True, exist_ok=args.resume)
    plan_sha = dual_sha256(args.plan)
    if args.resume and dual_sha256(output / "execution-plan.json") != plan_sha:
        raise ValueError("Cannot resume another execution list")
    if not args.resume:
        (output / "execution-plan.json").write_bytes(args.plan.read_bytes())
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    bundle = load_bundle(root / "fixed_input_bundle_v1", expected_sha256=BUNDLE_SHA)
    if (
        bundle.inputs.camera_ids != ("cam0", "cam1", "cam2")
        or bundle.inputs.num_frames != 1010
    ):
        raise ValueError("Only the agreed clip/cameras/timeline may run")
    identity = {
        "bundle_sha256": BUNDLE_SHA,
        "code_commit": plan["code_commit"],
        "code_fingerprint": plan["code_fingerprint"],
    }
    runs, checks, completions = {}, [], []
    for row in plan["methods"]:
        name = row["name"]
        receipt = output / "completed" / (name + ".json")
        print(f"Starting {name}: all views + 3 LOVO, one method at a time", flush=True)
        if receipt.exists() and args.resume:
            saved = json.loads(receipt.read_text())
            values = tuple(
                load_run(output / "results" / name / key, expected_sha256=digest)
                for key, digest in saved["manifests"].items()
            )
            for run in values:
                if (
                    run.config != row["config"]
                    or run.code_commit != plan["code_commit"]
                    or run.code_fingerprint != plan["code_fingerprint"]
                ):
                    raise ValueError(
                        "Resume record does not match the frozen executable/configuration"
                    )
        else:
            if (output / "results" / name).exists():
                raise ValueError(
                    "Partial unpublished row requires investigation; never overwrite it"
                )
            method = make_method(row, bundle)
            values = run_method(method, bundle.inputs, **identity)
            manifests = {
                r.key: save_run(output / "results" / name / r.key, r) for r in values
            }
            write_json(
                receipt,
                {
                    "name": name,
                    "manifests": manifests,
                    "code_commit": plan["code_commit"],
                },
            )
            del method
        checks.append(check_prior(row, values, plan, root, output))
        runs[name] = values
        completions.append(
            {
                "name": name,
                "valid_frames": {r.key: int(r.result.valid.sum()) for r in values},
                "seconds": sum(r.elapsed_seconds for r in values),
            }
        )
        write_json(output / "execution-progress.json", completions)
        print(json.dumps(completions[-1]), flush=True)
    # Labels are first opened after all blind method/composition outputs are saved.
    annotations = load_annotations(
        root / "human_annotations/ground_truth_v1/ground_truth.json",
        bundle,
        expected_sha256=REFERENCE_SHA,
    )
    diagnostics = load_script(
        "run-ball3d-error-decomposition-20261103",
        "diagnostics.py",
        "integrated_diagnostics",
    )
    oracle = annotation_oracle(
        bundle,
        annotations,
        code_commit=plan["code_commit"],
        code_fingerprint=plan["code_fingerprint"],
    )
    offsets, clock = diagnostics.fit_offsets(bundle.inputs, annotations)
    references = {
        "annotation_oracle": oracle,
        "annotation_time_oracle": diagnostics.timing_oracle(
            bundle.inputs, annotations, offsets, oracle
        ),
        **{
            f"candidate{k}_oracle": diagnostics.candidate_oracle(
                bundle.inputs, annotations, oracle, k=k
            )
            for k in (1, 8)
        },
    }
    for row in plan["diagnostics"]:
        values = references[row["name"]]
        path = output / "diagnostics" / row["name"]
        receipt = output / "completed" / (row["name"] + ".json")
        if path.exists():
            if not args.resume or not receipt.exists():
                raise ValueError("Partial diagnostic publication; never overwrite")
            saved = json.loads(receipt.read_text())
            for run in values:
                old = load_run(
                    path / run.key, expected_sha256=saved["manifests"][run.key]
                )
                if differences(run.result, old.result):
                    raise ValueError("Recomputed diagnostic differs on resume")
        else:
            manifests = {r.key: save_run(path / r.key, r) for r in values}
            write_json(receipt, {"name": row["name"], "manifests": manifests})
        checks.append(check_prior(row, values, plan, root, output))
    write_json(output / "clock-diagnostic.json", clock)
    write_json(output / "prior-comparison.json", checks)
    write_json(
        output / "execution-verification.json",
        {
            "plan_sha256": plan_sha,
            "code_commit": plan["code_commit"],
            "checkout_commit": git_head(),
            "code_fingerprint": plan["code_fingerprint"],
            "executable_bytes_equal_frozen_plan": True,
            "baseline": verify_baseline(bundle),
            "reference_opened_after_all_predictions_saved": True,
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "prediction_rows": len(runs),
            "prediction_runs": 4 * len(runs),
            "diagnostic_runs": 4 * len(references),
            "seconds_this_invocation": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "gpu_used": False,
            "detector_inference_rerun": False,
        },
    )
    print(
        f"Finished execution. All {len(checks)} prior/composition comparisons recorded at {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
