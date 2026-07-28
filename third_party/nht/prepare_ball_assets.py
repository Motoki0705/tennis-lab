#!/usr/bin/env python3
"""Run capture-verified NHT fitting and publish a multi-ball BLCS registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.asset_ingestion import (  # noqa: E402
    BALL_ASSET_INGESTION_SCHEMA,
    FROZEN_TARGET_NHT_FIT,
    INDEPENDENT_NHT_SOURCE,
    VANILLA_3DGS_SOURCE,
    BallAssetIngestionSpec,
    publish_ball_asset_registry_from_sources,
)
from src.synthetic_data_generation.blcs.assets import (  # noqa: E402
    load_ball_asset_registry,
    verify_local_ball_artifact,
    verify_local_gaussian_asset,
)
from src.synthetic_data_generation.blcs.calibration import (  # noqa: E402
    load_ball_calibration_import,
)
from src.synthetic_data_generation.composition.contracts import (  # noqa: E402
    load_gaussian_scene_manifest,
)
from src.synthetic_data_generation.scene_contract import (  # noqa: E402
    ArtifactRef,
    SimilarityTransform,
)

ASSET_PREPARATION_ENTRY_SCHEMA = "tennis_ball_asset_preparation_entry_v1"
ASSET_PREPARATION_REQUEST_SCHEMA = "tennis_ball_asset_preparation_request_v1"
ASSET_PREPARATION_RUN_SCHEMA = "tennis_ball_asset_preparation_run_v1"
ASSET_PREPARATION_FAILURE_SCHEMA = "tennis_ball_asset_preparation_failure_v1"
_SOURCE_FORMATS = {INDEPENDENT_NHT_SOURCE, VANILLA_3DGS_SOURCE}
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PreparationEntry:
    """One immutable source request loaded before any CUDA work."""

    spec_path: Path
    variant_id: str
    asset_id: str
    nominal_diameter_m: float
    source_format: str
    source: ArtifactRef
    asset_from_prepared: SimilarityTransform
    source_is_user_asset: bool

    def to_dict(self) -> dict[str, object]:
        """Return normalized request semantics plus the exact source bytes."""
        return {
            "variant_id": self.variant_id,
            "asset_id": self.asset_id,
            "nominal_diameter_m": self.nominal_diameter_m,
            "source_format": self.source_format,
            "source": self.source.to_dict(),
            "asset_from_prepared": self.asset_from_prepared.to_dict(),
            "source_is_user_asset": self.source_is_user_asset,
            "spec": _absolute_ref(self.spec_path, f"{self.asset_id}-preparation-spec"),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-spec",
        action="append",
        type=Path,
        required=True,
        help="Repeat for every user ball variant to publish in one registry.",
    )
    parser.add_argument("--calibration-import", type=Path, required=True)
    parser.add_argument("--background-composition", type=Path, required=True)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimization-steps", type=int, default=600)
    parser.add_argument("--feature-lr", type=float, default=0.015)
    parser.add_argument("--final-lr-fraction", type=float, default=0.1)
    parser.add_argument("--min-validation-psnr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_entry(path: Path) -> PreparationEntry:
    spec_path = path.resolve()
    with spec_path.open(encoding="utf-8") as handle:
        raw = _strict_mapping(
            json.load(handle),
            name="ball asset preparation entry",
            keys={
                "schema",
                "variant_id",
                "asset_id",
                "nominal_diameter_m",
                "source_format",
                "source",
                "asset_from_prepared",
                "source_is_user_asset",
            },
        )
    if raw["schema"] != ASSET_PREPARATION_ENTRY_SCHEMA:
        raise ValueError(f"Unsupported asset preparation schema: {raw['schema']!r}.")
    variant_id = _identifier(raw["variant_id"], name="variant_id")
    asset_id = _identifier(raw["asset_id"], name="asset_id")
    source_format = _string(raw["source_format"], name="source_format")
    if source_format not in _SOURCE_FORMATS:
        raise ValueError(f"Unsupported source_format: {source_format!r}.")
    diameter = _finite_float(raw["nominal_diameter_m"], name="nominal_diameter_m")
    if not 0.05 <= diameter <= 0.09:
        raise ValueError("nominal_diameter_m must lie in [0.05, 0.09].")
    source = ArtifactRef.from_dict(raw["source"])
    verify_local_ball_artifact(source)
    source_is_user_asset = raw["source_is_user_asset"]
    if not isinstance(source_is_user_asset, bool):
        raise TypeError("source_is_user_asset must be a boolean.")
    return PreparationEntry(
        spec_path=spec_path,
        variant_id=variant_id,
        asset_id=asset_id,
        nominal_diameter_m=diameter,
        source_format=source_format,
        source=source,
        asset_from_prepared=SimilarityTransform.from_dict(raw["asset_from_prepared"]),
        source_is_user_asset=source_is_user_asset,
    )


def _load_entries(paths: list[Path]) -> tuple[PreparationEntry, ...]:
    entries = tuple(
        sorted((_load_entry(path) for path in paths), key=lambda x: x.variant_id)
    )
    if len({entry.variant_id for entry in entries}) != len(entries):
        raise ValueError("asset-spec variant_id values must be unique.")
    if len({entry.asset_id for entry in entries}) != len(entries):
        raise ValueError("asset-spec asset_id values must be unique.")
    return entries


def _validate_options(args: argparse.Namespace) -> None:
    _identifier(args.registry_id, name="registry_id")
    if args.optimization_steps <= 0:
        raise ValueError("optimization-steps must be positive.")
    if not math.isfinite(args.feature_lr) or args.feature_lr <= 0.0:
        raise ValueError("feature-lr must be finite and positive.")
    if (
        not math.isfinite(args.final_lr_fraction)
        or not 0.0 < args.final_lr_fraction <= 1.0
    ):
        raise ValueError("final-lr-fraction must lie in (0,1].")
    if (
        not math.isfinite(args.min_validation_psnr_db)
        or args.min_validation_psnr_db < 20.0
    ):
        raise ValueError("min-validation-psnr-db cannot be lower than 20.")
    if isinstance(args.seed, bool) or args.seed < 0:
        raise ValueError("seed must be non-negative.")
    if not isinstance(args.device, str) or not args.device.startswith("cuda:"):
        raise ValueError("device must explicitly select a CUDA device.")


def _feature_fit_command(
    *,
    entry: PreparationEntry,
    calibration_bundle: Path,
    target_appearance: Path,
    appearance_space_sha256: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "ball_feature_fit.py"),
        "--source",
        str(verify_local_ball_artifact(entry.source)),
        "--source-format",
        entry.source_format,
        "--calibration-bundle",
        str(calibration_bundle),
        "--target-appearance",
        str(target_appearance),
        "--target-appearance-space-sha256",
        appearance_space_sha256,
        "--output-dir",
        str(output_dir),
        "--optimization-steps",
        str(args.optimization_steps),
        "--feature-lr",
        str(args.feature_lr),
        "--final-lr-fraction",
        str(args.final_lr_fraction),
        "--min-validation-psnr-db",
        str(args.min_validation_psnr_db),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _write_process_log(
    path: Path,
    *,
    command: list[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    path.write_text(
        json.dumps({"command": command, "returncode": result.returncode}, indent=2)
        + "\n--- stdout ---\n"
        + result.stdout
        + "\n--- stderr ---\n"
        + result.stderr,
        encoding="utf-8",
    )


def _write_failure(
    output_dir: Path,
    *,
    error: BaseException,
    completed_asset_ids: list[str],
) -> None:
    failure_path = output_dir / "failure.json"
    if failure_path.exists():
        return
    _write_json_exclusive(
        failure_path,
        {
            "schema": ASSET_PREPARATION_FAILURE_SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_asset_ids": completed_asset_ids,
            "registry_published": (output_dir / "registry" / "registry.json").is_file(),
        },
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    _validate_options(args)
    entries = _load_entries(args.asset_spec)
    calibration = load_ball_calibration_import(args.calibration_import)
    composition_path = args.background_composition.resolve()
    composition = load_gaussian_scene_manifest(composition_path)
    verify_local_gaussian_asset(composition.background)
    target_appearance = verify_local_ball_artifact(
        composition.background.appearance_payload
    )
    feature_worker = Path(__file__).resolve().parent / "ball_feature_fit.py"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    request_unsigned: dict[str, object] = {
        "schema": ASSET_PREPARATION_REQUEST_SCHEMA,
        "registry_id": args.registry_id,
        "background_composition": _absolute_ref(
            composition_path,
            "target-background-composition",
        ),
        "calibration_import": _absolute_ref(
            calibration.root / "capture-import.json",
            "ball-calibration-import",
        ),
        "calibration_bundle_fingerprint": calibration.bundle.manifest[
            "content_fingerprint"
        ],
        "entries": [entry.to_dict() for entry in entries],
        "optimization": {
            "steps": args.optimization_steps,
            "feature_lr": args.feature_lr,
            "final_lr_fraction": args.final_lr_fraction,
            "minimum_validation_psnr_db": args.min_validation_psnr_db,
            "seed": args.seed,
            "device": args.device,
        },
        "launcher_sha256": _sha256_file(Path(__file__).resolve()),
        "feature_worker_sha256": _sha256_file(feature_worker),
    }
    request = {
        **request_unsigned,
        "content_fingerprint": _canonical_sha256(request_unsigned),
    }
    _write_json_exclusive(output_dir / "request.json", request)

    completed_asset_ids: list[str] = []
    ingestion_specs: list[BallAssetIngestionSpec] = []
    fit_records: list[dict[str, object]] = []
    try:
        for entry_index, entry in enumerate(entries):
            fit_dir = output_dir / "fit" / entry.asset_id
            fit_dir.parent.mkdir(exist_ok=True)
            command = _feature_fit_command(
                entry=entry,
                calibration_bundle=calibration.bundle.root,
                target_appearance=target_appearance,
                appearance_space_sha256=(
                    composition.background.appearance_space_sha256
                ),
                output_dir=fit_dir,
                args=args,
            )
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            log_path = output_dir / "logs" / f"{entry.asset_id}.log"
            _write_process_log(log_path, command=command, result=result)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Feature fit failed for {entry.asset_id} with exit code "
                    f"{result.returncode}; see {log_path}."
                )
            fit_manifest = fit_dir / "manifest.json"
            prepared_tensors = fit_dir / "prepared-nht-tensors.pt"
            conversion_report = fit_dir / "conversion-report.json"
            for required in (fit_manifest, prepared_tensors, conversion_report):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"Feature fit omitted required output: {required}"
                    )
            source_artifacts = (
                entry.source,
                _artifact(
                    entry.spec_path,
                    f"{entry.asset_id}-preparation-spec",
                ),
                _artifact(
                    calibration.root / "capture-import.json",
                    f"{entry.asset_id}-calibration-import",
                ),
                _artifact(
                    fit_manifest,
                    f"{entry.asset_id}-feature-fit-manifest",
                ),
            )
            ingestion_specs.append(
                BallAssetIngestionSpec(
                    schema=BALL_ASSET_INGESTION_SCHEMA,
                    variant_id=entry.variant_id,
                    asset_id=entry.asset_id,
                    nominal_diameter_m=entry.nominal_diameter_m,
                    source_format=entry.source_format,
                    source_artifacts=source_artifacts,
                    prepared_tensors=_artifact(
                        prepared_tensors,
                        f"{entry.asset_id}-prepared-nht-tensors",
                    ),
                    prepared_appearance_space_sha256=(
                        composition.background.appearance_space_sha256
                    ),
                    prepared_appearance_payload=(
                        composition.background.appearance_payload
                    ),
                    conversion_method=FROZEN_TARGET_NHT_FIT,
                    conversion_report=_artifact(
                        conversion_report,
                        f"{entry.asset_id}-conversion-report",
                    ),
                    asset_from_prepared=entry.asset_from_prepared,
                )
            )
            completed_asset_ids.append(entry.asset_id)
            fit_records.append(
                {
                    "entry_index": entry_index,
                    "variant_id": entry.variant_id,
                    "asset_id": entry.asset_id,
                    "manifest": _relative_ref(output_dir, fit_manifest),
                    "conversion_report": _relative_ref(
                        output_dir,
                        conversion_report,
                    ),
                    "log": _relative_ref(output_dir, log_path),
                }
            )
        registry_path = publish_ball_asset_registry_from_sources(
            output_dir / "registry",
            registry_id=args.registry_id,
            target_background=composition.background,
            sources=ingestion_specs,
        )
        registry = load_ball_asset_registry(
            registry_path,
            verify_local_artifacts=True,
        )
        manifest_unsigned: dict[str, object] = {
            "schema": ASSET_PREPARATION_RUN_SCHEMA,
            "status": "passed",
            "request": _relative_ref(output_dir, output_dir / "request.json"),
            "feature_fits": fit_records,
            "registry": _relative_ref(output_dir, registry_path),
            "registry_fingerprint": registry.registry_fingerprint,
            "variant_count": len(registry.entries),
            "source_is_user_asset": all(
                entry.source_is_user_asset for entry in entries
            ),
            "rgb_overlay_used": False,
        }
        manifest = {
            **manifest_unsigned,
            "content_fingerprint": _canonical_sha256(manifest_unsigned),
        }
        _write_json_exclusive(output_dir / "manifest.json", manifest)
    except BaseException as error:
        _write_failure(
            output_dir,
            error=error,
            completed_asset_ids=completed_asset_ids,
        )
        raise
    print(json.dumps(manifest, sort_keys=True))


def _absolute_ref(path: Path, artifact_id: str) -> dict[str, object]:
    return _artifact(path, artifact_id).to_dict()


def _artifact(path: Path, artifact_id: str) -> ArtifactRef:
    resolved = path.resolve()
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=resolved.as_uri(),
        sha256=_sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _relative_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    if set(value) != keys:
        raise ValueError(
            f"{name} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}."
        )
    return value


def _identifier(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    if _ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} contains unsupported characters.")
    return text


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


if __name__ == "__main__":
    main()
