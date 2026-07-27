"""
Fit stable metric court instances to a published fit-only ground-line map.

Usage:
    python -m src.synthetic_data_generation.scripts.fit_ground_courts
    python -m src.synthetic_data_generation.scripts.fit_ground_courts fit.seed=20260725

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/fit_ground_courts.yaml`.
    - Evidence-guided multi-start clustering rejects unstable and ambiguous fits.
    - This stage never reads holdout images and publishes a fit candidate, not
      an accepted alignment.
"""

from __future__ import annotations

import hashlib
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.court_template_fit import (
    COURT_GEOMETRY_SCHEMA,
    publish_court_geometry_artifact,
)
from src.synthetic_data_generation.alignment.evidence_guided_court_fit import (
    CourtMultiStartFitSettings,
    fit_unknown_number_of_courts,
)
from src.synthetic_data_generation.alignment.ground_line_map import (
    load_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.synthetic_data_generation.provider.bundle import sha256_file
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="fit_ground_courts",
)
def main(cfg: DictConfig) -> int:
    """Fit and publish distinct court candidates from fit-only evidence."""
    repo_root = Path(to_absolute_path(".")).resolve()
    ground_line_path = _path(cfg.ground_line_artifact)
    output_dir = _path(cfg.output_dir)
    manifest, arrays = load_ground_line_map_artifact(ground_line_path)
    if manifest["split"]["holdout_inference_status"] != "not_run":
        raise ValueError("Ground-line holdout images must remain uninferred.")
    fit_config = OmegaConf.to_container(cfg.fit, resolve=True)
    if not isinstance(fit_config, dict):
        raise ValueError("fit config must be a mapping.")
    settings = CourtMultiStartFitSettings(**cast(dict[str, Any], fit_config))
    plane_payload = manifest["ground_plane"]["estimate"]
    if not isinstance(plane_payload, dict):
        raise ValueError("Ground-line artifact has no ground-plane estimate.")
    plane = GroundPlaneEstimate(**plane_payload)
    bounds = tuple(float(value) for value in manifest["projection"]["bounds_uv"])
    if len(bounds) != 4:
        raise ValueError("Ground-line projection bounds must have four values.")
    grid_spacing = float(manifest["projection"]["settings"]["grid_spacing"])
    result = fit_unknown_number_of_courts(
        arrays["evidence_sum"],
        bounds=bounds,
        grid_spacing=grid_spacing,
        plane=plane,
        settings=settings,
    )
    candidates = result.accepted_candidates
    if not candidates:
        raise ValueError(
            "Evidence-guided fitting produced no reliable court candidate; "
            f"stop_status={result.stop_status}."
        )
    accepted_clusters = [
        cluster
        for clusters in result.clusters_by_iteration
        for cluster in clusters
        if cluster.candidate is not None
        and cluster.candidate.candidate_id
        in {candidate.candidate_id for candidate in candidates}
        and not cluster.rejection_reasons
    ]
    cluster_by_candidate_id = {
        cluster.candidate.candidate_id: cluster
        for cluster in accepted_clusters
        if cluster.candidate is not None
    }
    candidate_payloads = []
    for candidate in candidates:
        candidate_payload = candidate.to_dict()
        cluster = cluster_by_candidate_id[candidate.candidate_id]
        scene_from_court = np.asarray(candidate.scene_from_court).reshape(4, 4)
        court_from_scene = np.linalg.inv(scene_from_court)
        candidate_payload.update(
            {
                "court_from_scene": [
                    float(value) for value in court_from_scene.ravel()
                ],
                "scale_metres_per_scene_unit": (1.0 / candidate.scale_scene_per_metre),
                "linear_determinant": float(np.linalg.det(scene_from_court[:3, :3])),
                "confidence": cluster.confidence,
                "support_rate": cluster.support_rate,
                "bootstrap_survival_rate": cluster.bootstrap_survival_rate,
                "component_scores": cluster.component_scores,
                "parameter_dispersion": cluster.parameter_dispersion,
            }
        )
        candidate_payloads.append(candidate_payload)
    selected = candidates[0]
    ground_line_reference, ground_line_path_scope = _provenance_path(
        ground_line_path,
        repo_root=repo_root,
    )
    config_path = (
        repo_root / "src/synthetic_data_generation/configs/fit_ground_courts.yaml"
    )
    script_path = (
        repo_root / "src/synthetic_data_generation/scripts/fit_ground_courts.py"
    )
    payload = {
        "schema": COURT_GEOMETRY_SCHEMA,
        "artifact_id": str(cfg.artifact_id),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ground_line_artifact": {
            "path": ground_line_reference,
            "path_scope": ground_line_path_scope,
            "manifest_sha256": sha256_file(ground_line_path / "manifest.json"),
            "artifact_fingerprint": manifest["artifact_fingerprint"],
            "fit_camera_count": len(manifest["split"]["fit_camera_ids"]),
            "holdout_camera_count": len(manifest["split"]["holdout_camera_ids"]),
            "holdout_inference_status": "not_run",
        },
        "fit_settings": asdict(settings),
        "candidates": candidate_payloads,
        "selection": {
            "selected_candidate_id": selected.candidate_id,
            "rule": (
                "sequential residual selection by evidence-guided multi-start "
                "support, bootstrap stability, line coverage, contrast, and "
                "explained evidence with explicit 90-degree ambiguity rejection"
            ),
            "selected_template_score": selected.template_score,
            "selected_confidence": cluster_by_candidate_id[
                selected.candidate_id
            ].confidence,
            "family_metrics": _family_metrics(candidates),
            "multistart_diagnostics": result.diagnostics_dict(),
            "coordinate_conventions": {
                "court": (
                    "right_handed_metres:+x_right_sideline,+y_far_baseline,+z_up"
                ),
                "scene": "provider scene coordinates, right handed",
                "ground_uv": (
                    "u=basis_u, v=basis_v, basis_u_cross_basis_v=plane_normal"
                ),
            },
        },
        "acceptance_status": "fit_candidate_holdout_not_run",
        "provenance": {
            "command": shlex.join(
                [str(repo_root / ".venv/bin/python"), "-m", __spec__.name]
            ),
            "config": {
                "path": str(config_path.relative_to(repo_root)),
                "sha256": sha256_file(config_path),
                "resolved": OmegaConf.to_container(cfg, resolve=True),
            },
            "code": {
                "git_revision": _git_revision(repo_root),
                "git_diff_sha256": _git_diff_sha256(repo_root),
                "script": {
                    "path": str(script_path.relative_to(repo_root)),
                    "sha256": sha256_file(script_path),
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    }
    published = publish_court_geometry_artifact(payload, output_dir=output_dir)
    print(published)
    return 0


def _family_metrics(candidates: tuple[Any, ...]) -> dict[str, float | None]:
    if len(candidates) < 2:
        return {
            "center_separation_scene": None,
            "center_separation_metres": None,
            "orientation_difference_degrees": None,
            "relative_scale_difference": None,
        }
    first, second = candidates[:2]
    center_distance = float(
        np.linalg.norm(np.asarray(first.center_uv) - np.asarray(second.center_uv))
    )
    mean_scale = 0.5 * (first.scale_scene_per_metre + second.scale_scene_per_metre)
    angle_difference = abs(first.orientation_radians - second.orientation_radians)
    angle_difference = min(angle_difference, np.pi - angle_difference)
    return {
        "center_separation_scene": center_distance,
        "center_separation_metres": center_distance / mean_scale,
        "orientation_difference_degrees": float(np.degrees(angle_difference)),
        "relative_scale_difference": (
            abs(first.scale_scene_per_metre - second.scale_scene_per_metre) / mean_scale
        ),
    }


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _provenance_path(path: Path, *, repo_root: Path) -> tuple[str, str]:
    try:
        return str(path.relative_to(repo_root)), "repository_relative"
    except ValueError:
        return str(path), "external_absolute"


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_diff_sha256(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "--", "src/synthetic_data_generation"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
