"""Apply the explicit fold-table schema correction to the frozen report driver.

The original report.py and all prediction/scoring code remain byte-for-byte
unchanged. This presentation adapter has its own commit/SHA provenance.
"""

import importlib.util
import json
import sys
from pathlib import Path

from common import HERE, git_head, write_json

from src.utils.checksum import dual_sha256

SPEC = importlib.util.spec_from_file_location(
    "ball3d_frozen_report_v1", HERE / "report.py"
)
assert SPEC is not None and SPEC.loader is not None
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)
FROZEN_METRIC_ROWS = ORIGINAL.metric_rows


def metric_rows(scores, plan):
    # Bind the existing driver's table input to the foundation's coverage-bearing
    # summaries. The scorer and its returned report are not modified.
    display = {
        "methods": {
            name: {
                **method,
                "folds": {
                    key: {
                        segment_name: {
                            **segment,
                            "reprojection_px": segment[
                                "reprojection_by_reference_kind"
                            ]["all"],
                        }
                        for segment_name, segment in fold.items()
                    }
                    for key, fold in method["folds"].items()
                },
            }
            for name, method in scores["methods"].items()
        }
    }
    return FROZEN_METRIC_ROWS(display, plan)


def main():
    ORIGINAL.metric_rows = metric_rows
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    stage = {
        "version": 2,
        "code_commit": git_head(),
        "script_sha256": dual_sha256(Path(__file__)),
        "frozen_driver_sha256": dual_sha256(HERE / "report.py"),
        "reason": "Fold reprojection_px contains only a distribution; use the foundation's reference-kind all summary for count/eligible/coverage.",
        "prediction_code_changed": False,
        "scorer_changed": False,
        "previous_attempt": "attempts/report-v1"
        if (output / "attempts/report-v1").is_dir()
        else None,
    }
    write_json(output / "report-stage.json", stage)
    frozen_page = ORIGINAL.write_page

    def page_with_stage(
        directory, plan, rows, cohorts, verification, images, events, shortlist
    ):
        verification["report_stage"] = stage
        write_json(directory / "verification.json", verification)
        return frozen_page(
            directory, plan, rows, cohorts, verification, images, events, shortlist
        )

    ORIGINAL.write_page = page_with_stage
    ORIGINAL.main()
    old_scores = output / "attempts/report-v1/scores.json"
    if old_scores.exists() and json.loads(old_scores.read_text()) != json.loads(
        (output / "scores.json").read_text()
    ):
        raise ValueError("Presentation correction changed foundation scoring values")


if __name__ == "__main__":
    main()
