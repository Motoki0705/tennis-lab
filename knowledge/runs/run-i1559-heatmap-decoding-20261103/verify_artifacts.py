"""Read-only audit of completed, checksummed foundation results and media."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.visualization.ball3d.report import broken_html_references
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.comparison
    manifest = json.loads((root / "artifacts.json").read_text())
    digest = dual_sha256(root / "artifacts.json")
    assert digest == (root / "artifacts.sha256").read_text().strip()
    expected = set(manifest["files"]) | {"artifacts.json", "artifacts.sha256"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert actual == expected
    for relative, record in manifest["files"].items():
        path = root / relative
        assert path.stat().st_size == record["size_bytes"], relative
        assert dual_sha256(path) == record["sha256"], relative
    verification = json.loads((root / "verification.json").read_text())
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    assert fingerprint == verification["code_fingerprint"]
    for path in (root / "results").glob("*/*/result.json"):
        run = load_run(path.parent)
        assert run.code_commit == verification["code_commit"]
        assert run.code_fingerprint == fingerprint
        assert run.bundle_sha256 == verification["bundle_sha256"]
    baseline = load_run(root / "results/pipeline_baseline/all_views")
    reference = load_run(args.baseline / "results/pipeline_baseline/all_views")
    assert baseline.bundle_sha256 == reference.bundle_sha256
    for field in ("valid", "used_views"):
        np.testing.assert_array_equal(getattr(baseline.result, field), getattr(reference.result, field))
    np.testing.assert_allclose(baseline.result.positions, reference.result.positions, rtol=0, atol=2e-6)
    broken = broken_html_references(root / "index.html")
    assert not broken
    media = verification["comparison_video"]
    assert media["decoded_frames"] == 1010
    assert media["sha256"] == dual_sha256(root / "comparison.mp4")
    receipt = {
        "comparison": str(root.resolve()), "manifest_sha256": digest,
        "verified_files": len(manifest["files"]),
        "typed_results": len(list((root / "results").glob("*/*/result.json"))),
        "current_source_matches_executed_fingerprint": True,
        "baseline_reference": str(args.baseline.resolve()),
        "baseline_masks_exact": True,
        "baseline_positions_bitwise_equal": bool(np.array_equal(baseline.result.positions, reference.result.positions)),
        "baseline_positions_max_difference_m": float(np.abs(baseline.result.positions - reference.result.positions).max()),
        "broken_html_references": broken, "full_video_decode": media,
        "important_files": {p: dual_sha256(root / p) for p in ("index.html", "comparison.mp4", "scores.json", "selection.json", "paired-scores.json")},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
