"""CPU-only cross-model court-alignment benchmark entry point.

The command plans a deterministic manifest, runs each requested model once per
manifest sample on CPU, evaluates the shared metrics, and writes the report
artifacts.  Predictions are cached per sample so an interrupted run resumes
without repeating work, and any cached artifact that disagrees with the
requested run stops the command.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.adapters import (
    CourtCheckpointAdapter,
    KeypointModelAdapter,
    TennisCourtDetectorAdapter,
    checkpoint_training_config,
    require_cpu,
)
from src.tasks.court_detection.evaluation.contracts import (
    DOMAIN_NAMES,
    MODEL_NAMES,
    DomainName,
    LoadedSample,
    ModelName,
    ModelPrediction,
    SampleRef,
)
from src.tasks.court_detection.evaluation.datasets import (
    DomainRecords,
    build_domain_records,
    load_sample,
    sample_ref,
    select_manifest_refs,
)
from src.tasks.court_detection.evaluation.metrics import (
    SampleEvaluation,
    aggregate_evaluations,
    evaluate_sample,
)
from src.tasks.court_detection.evaluation.reporting import write_report
from src.tasks.court_detection.evaluation.settings import BenchmarkSettings
from src.tasks.court_detection.evaluation.storage import (
    ManifestDocument,
    PredictionStore,
    environment_provenance,
    fingerprint,
    write_json_atomic,
    write_manifest_once,
)
from src.tasks.court_detection.evaluation.visualization import (
    MODEL_ORDER,
    render_domain_montage,
    render_error_cdf,
    render_sample_row,
    render_summary_bars,
    select_review_samples,
)
from src.utils.paths import PROJECT_ROOT

DEFAULT_CONFIG = (
    PROJECT_ROOT / "src/tasks/court_detection/configs/benchmark_alignment.yaml"
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Resolved command-line contract for one benchmark invocation."""

    settings: BenchmarkSettings
    output_root: Path
    repo_root: Path
    models: tuple[ModelName, ...]
    domains: tuple[DomainName, ...]
    ours_checkpoint: Path | None
    tcd_repo: Path | None
    tcd_checkpoint: Path | None
    force: bool
    command: str

    def prediction_store(
        self,
        domain: DomainName,
        *,
        manifest: ManifestDocument,
        model_fingerprints: dict[str, str],
        model: str,
    ) -> PredictionStore:
        return PredictionStore(
            root=self.output_root,
            domain=domain,
            model=model,
            manifest_fingerprint=manifest.settings_fingerprint,
            model_fingerprint=model_fingerprints[model],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    arguments = parser.parse_args(raw)
    command = " ".join(["benchmark_alignment", *raw])
    _run(resolve_config(arguments, command=command))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI.  Lightning '=' filenames pass through verbatim."""
    parser = argparse.ArgumentParser(
        prog="benchmark_alignment",
        description=(
            "Compare Court detection checkpoints on one shared CPU manifest. "
            "The real domain is the held-out validation split, not an official "
            "test set."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Benchmark settings YAML."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=PROJECT_ROOT,
        help=(
            "Base directory for the config's relative data/cache/external roots "
            "and for the 'ours' checkpoint's backbone paths. Defaults to the "
            "checkout running this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory for the manifest, cache, metrics, and figures.",
    )
    parser.add_argument(
        "--models",
        choices=[*MODEL_NAMES, "all"],
        default="all",
        help="Which models to run (default: all).",
    )
    parser.add_argument(
        "--datasets",
        choices=[*DOMAIN_NAMES, "all"],
        default="all",
        help="Which domains to run (default: all).",
    )
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=None,
        help="Deterministic cap per domain; overrides the config when set.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Selection seed override."
    )
    parser.add_argument(
        "--ours-checkpoint",
        type=Path,
        default=None,
        help="Court checkpoint for 'ours' (Lightning '=' in the name is accepted).",
    )
    parser.add_argument(
        "--tcd-repo",
        type=Path,
        default=None,
        help="External yastrebksv/TennisCourtDetector checkout (never copied).",
    )
    parser.add_argument(
        "--tcd-checkpoint",
        type=Path,
        default=None,
        help="External baseline weights (never copied or committed).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Only 'cpu' is accepted; GPU execution is refused.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete this run's cached predictions before starting.",
    )
    return parser


def resolve_config(arguments: argparse.Namespace, *, command: str) -> RunConfig:
    """Validate CLI intent and freeze it into an immutable run contract."""
    require_cpu(cast("str", arguments.device))
    repo_root = Path(cast("Path", arguments.repo_root))
    if not repo_root.is_dir():
        raise ValueError(f"--repo-root is not a directory: {repo_root}")
    repo_root = repo_root.resolve()
    settings = BenchmarkSettings.load(
        cast("Path", arguments.config), project_root=repo_root
    )
    selection = settings.selection
    if arguments.seed is not None:
        selection = replace(selection, seed=int(arguments.seed))
    if arguments.max_samples_per_domain is not None:
        maximum = int(arguments.max_samples_per_domain)
        if maximum <= 0:
            raise ValueError("--max-samples-per-domain must be positive.")
        selection = replace(selection, max_samples_per_domain=maximum)
    settings = replace(settings, selection=selection)
    models = cast(
        "tuple[ModelName, ...]",
        MODEL_NAMES if arguments.models == "all" else (arguments.models,),
    )
    domains = cast(
        "tuple[DomainName, ...]",
        DOMAIN_NAMES if arguments.datasets == "all" else (arguments.datasets,),
    )
    if "ours" in models and arguments.ours_checkpoint is None:
        raise ValueError("--ours-checkpoint is required when running the 'ours' model.")
    if "tcd" in models and (
        arguments.tcd_repo is None or arguments.tcd_checkpoint is None
    ):
        raise ValueError(
            "--tcd-repo and --tcd-checkpoint are required when running 'tcd'."
        )
    output_root = Path(cast("Path", arguments.output_dir))
    if not output_root.is_absolute():
        raise ValueError("--output-dir must be an absolute path.")
    return RunConfig(
        settings=settings,
        output_root=output_root,
        repo_root=repo_root,
        models=models,
        domains=domains,
        ours_checkpoint=cast("Path | None", arguments.ours_checkpoint),
        tcd_repo=cast("Path | None", arguments.tcd_repo),
        tcd_checkpoint=cast("Path | None", arguments.tcd_checkpoint),
        force=bool(arguments.force),
        command=command,
    )


def _run(config: RunConfig) -> None:
    settings = config.settings
    started = time.time()
    if config.force:
        clear_cache(config.output_root)
    domains = {
        domain: build_domain_records(
            settings.domains[domain], derived_target_root=settings.derived_target_root
        )
        for domain in config.domains
    }
    refs = _manifest_refs(domains, settings=settings)
    manifest = write_manifest_once(
        config.output_root / "manifest.json",
        settings_fingerprint=_settings_fingerprint(config, refs),
        quality={
            "pck_fractions": list(settings.quality.pck_fractions),
            "ransac_threshold_fraction": settings.quality.ransac_threshold_fraction,
            "line_samples_per_segment": settings.quality.line_samples_per_segment,
            "strata_axes": list(settings.quality.strata_axes),
        },
        selection={
            "seed": settings.selection.seed,
            "max_samples_per_domain": settings.selection.max_samples_per_domain,
        },
        domains={
            domain: {
                "display_name": settings.domains[domain].display_name,
                "split": settings.domains[domain].split,
                "source_root": str(settings.domains[domain].source_root),
            }
            for domain in config.domains
        },
        samples=refs,
    )
    adapters, adapter_provenance = _build_adapters(config)
    model_fingerprints = {
        name: fingerprint(provenance) for name, provenance in adapter_provenance.items()
    }
    evaluations = _predict_and_evaluate(
        config,
        domains=domains,
        manifest=manifest,
        adapters=adapters,
        model_fingerprints=model_fingerprints,
    )
    display_names: dict[str, str] = {
        str(domain): settings.domains[domain].display_name for domain in config.domains
    }
    aggregates = _aggregate(config, evaluations)
    report_paths = write_report(
        config.output_root,
        aggregates=aggregates,
        display_names=display_names,
        manifest_fingerprint=manifest.settings_fingerprint,
        command=config.command,
    )
    figures = _render_figures(
        config,
        domains=domains,
        manifest=manifest,
        adapters=adapters,
        model_fingerprints=model_fingerprints,
        evaluations=evaluations,
        aggregates=aggregates,
        display_names=display_names,
    )
    elapsed = time.time() - started
    write_json_atomic(
        config.output_root / "provenance.json",
        {
            "command": config.command,
            "settings_fingerprint": manifest.settings_fingerprint,
            "models": list(config.models),
            "domains": dict(display_names),
            "device": "cpu",
            "environment": environment_provenance(
                root=config.repo_root, code_root=PROJECT_ROOT
            ),
            "adapter_provenance": adapter_provenance,
            "model_fingerprints": model_fingerprints,
            "sample_counts": {
                f"{domain}:{model}": sum(
                    1
                    for item in evaluations
                    if item.ref.domain == domain and item.model == model
                )
                for domain in config.domains
                for model in config.models
            },
            "elapsed_seconds": elapsed,
            "artifacts": {
                "manifest": str(config.output_root / "manifest.json"),
                **{f"metrics_{key}": str(value) for key, value in report_paths.items()},
                **{f"figure_{key}": str(value) for key, value in figures.items()},
            },
        },
    )
    print(
        f"benchmark complete: {len(refs)} samples, {len(config.models)} model(s), "
        f"{len(config.domains)} domain(s) in {elapsed:.1f}s -> {config.output_root}"
    )


def clear_cache(output_root: Path) -> None:
    """Remove only this run's prediction cache, never its report artifacts."""
    predictions = output_root / "predictions"
    if not predictions.exists():
        return
    for path in sorted(predictions.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    predictions.rmdir()


def _primary_review_model(config: RunConfig) -> str:
    """Pick the model whose error ranks the qualitative review samples.

    ``ours`` is the primary model when it is part of the run; a baseline-only
    run ranks its own samples so single-model invocations still produce figures.
    """
    for model in MODEL_ORDER:
        if model in config.models:
            return model
    raise ValueError("A run must select at least one model.")


def _settings_fingerprint(config: RunConfig, refs: Sequence[SampleRef]) -> str:
    settings = config.settings
    return fingerprint(
        {
            "quality": {
                "pck_fractions": list(settings.quality.pck_fractions),
                "ransac_threshold_fraction": settings.quality.ransac_threshold_fraction,
                "line_samples_per_segment": settings.quality.line_samples_per_segment,
                "strata_axes": list(settings.quality.strata_axes),
            },
            "selection": {
                "seed": settings.selection.seed,
                "max_samples_per_domain": settings.selection.max_samples_per_domain,
            },
            "samples": [ref.to_json() for ref in refs],
        }
    )


def _manifest_refs(
    domains: dict[DomainName, DomainRecords], *, settings: BenchmarkSettings
) -> tuple[SampleRef, ...]:
    collected: list[SampleRef] = []
    for domain in DOMAIN_NAMES:
        records = domains.get(domain)
        if records is None:
            continue
        candidates = [
            sample_ref(records, record) for record in records.records.values()
        ]
        collected.extend(select_manifest_refs(candidates, selection=settings.selection))
    return tuple(sorted(collected, key=lambda item: (item.domain, item.sample_id)))


def _build_adapters(
    config: RunConfig,
) -> tuple[dict[str, KeypointModelAdapter], dict[str, object]]:
    adapters: dict[str, KeypointModelAdapter] = {}
    provenance: dict[str, object] = {}
    if "ours" in config.models:
        checkpoint = cast("Path", config.ours_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Court checkpoint is missing: {checkpoint}")
        # Validate the serialized config before allocating the model, so an
        # unreplayable checkpoint fails before any inference work.
        checkpoint_training_config(checkpoint, project_root=config.repo_root)
        ours = CourtCheckpointAdapter(
            checkpoint, project_root=config.repo_root, device="cpu"
        )
        adapters[ours.name] = ours
        provenance[ours.name] = dict(ours.provenance())
    if "tcd" in config.models:
        tcd = TennisCourtDetectorAdapter(
            cast("Path", config.tcd_repo),
            cast("Path", config.tcd_checkpoint),
            device="cpu",
        )
        adapters[tcd.name] = tcd
        provenance[tcd.name] = dict(tcd.provenance())
    return adapters, provenance


def _predict_and_evaluate(
    config: RunConfig,
    *,
    domains: dict[DomainName, DomainRecords],
    manifest: ManifestDocument,
    adapters: dict[str, KeypointModelAdapter],
    model_fingerprints: dict[str, str],
) -> list[SampleEvaluation]:
    evaluations: list[SampleEvaluation] = []
    for domain in config.domains:
        refs = manifest.samples_for(domain)
        if not refs:
            raise RuntimeError(f"Domain {domain!r} has no manifest samples.")
        stores = {
            model: config.prediction_store(
                domain,
                manifest=manifest,
                model_fingerprints=model_fingerprints,
                model=model,
            )
            for model in config.models
        }
        loaded: dict[str, LoadedSample] = {
            ref.sample_id: load_sample(domains[domain], ref) for ref in refs
        }
        for model in config.models:
            store = stores[model]
            store.ensure_header()
            completed = store.completed()
            for ref in refs:
                if ref.sample_id in completed:
                    continue
                prediction = adapters[model].predict(loaded[ref.sample_id].image_rgb)
                store.record(ref.sample_id, prediction)
        for ref in refs:
            predictions: dict[str, ModelPrediction] = {
                model: stores[model].load(ref.sample_id) for model in config.models
            }
            for model in config.models:
                evaluations.append(
                    evaluate_sample(
                        loaded[ref.sample_id],
                        predictions[model],
                        quality=config.settings.quality,
                    )
                )
    return evaluations


def _aggregate(
    config: RunConfig, evaluations: Sequence[SampleEvaluation]
) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    for domain in config.domains:
        for model in config.models:
            subset = [
                item
                for item in evaluations
                if item.ref.domain == domain and item.model == model
            ]
            if not subset:
                raise RuntimeError(
                    f"No evaluations for domain={domain!r} model={model!r}."
                )
            aggregates.append(
                aggregate_evaluations(subset, quality=config.settings.quality)
            )
    return aggregates


def _render_figures(
    config: RunConfig,
    *,
    domains: dict[DomainName, DomainRecords],
    manifest: ManifestDocument,
    adapters: dict[str, KeypointModelAdapter],
    model_fingerprints: dict[str, str],
    evaluations: Sequence[SampleEvaluation],
    aggregates: Sequence[dict[str, object]],
    display_names: dict[str, str],
) -> dict[str, Path]:
    import cv2

    figures_dir = config.output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    per_model_errors: dict[str, dict[str, NDArray[np.float64]]] = {
        model: {} for model in config.models
    }
    for domain in config.domains:
        for model in config.models:
            per_model_errors[model][domain] = np.asarray(
                [
                    error / item.ref.diagonal_px
                    for item in evaluations
                    if item.ref.domain == domain and item.model == model
                    for error in item.pair_errors_px
                ],
                dtype=np.float64,
            )
    figures: dict[str, Path] = {}
    cdf = figures_dir / "error_cdf.png"
    render_error_cdf(
        cdf, per_model_errors=per_model_errors, display_names=display_names
    )
    figures["error_cdf"] = cdf
    bars = figures_dir / "summary_bars.png"
    render_summary_bars(bars, aggregates=aggregates, display_names=display_names)
    figures["summary_bars"] = bars

    for domain in config.domains:
        refs = manifest.samples_for(domain)
        selected = select_review_samples(
            [item for item in evaluations if item.ref.domain == domain],
            primary_model=_primary_review_model(config),
            count=config.settings.visualization.samples_per_domain,
        )
        by_id = {ref.sample_id: ref for ref in refs}
        rows = []
        for entry in selected:
            ref = by_id[entry.sample_id]
            sample = load_sample(domains[domain], ref)
            predictions: dict[str, ModelPrediction] = {
                model: config.prediction_store(
                    domain,
                    manifest=manifest,
                    model_fingerprints=model_fingerprints,
                    model=model,
                ).load(ref.sample_id)
                for model in config.models
            }
            sample_evaluations: dict[str, SampleEvaluation] = {
                item.model: item
                for item in evaluations
                if item.ref.domain == domain and item.ref.sample_id == ref.sample_id
            }
            rows.append(
                (
                    entry,
                    render_sample_row(
                        sample,
                        predictions,
                        sample_evaluations,
                        quality=config.settings.quality,
                    ),
                )
            )
        montage = render_domain_montage(
            rows,
            title=(
                f"{display_names[domain]} ({domain}) - deterministic review selection"
            ),
        )
        path = figures_dir / f"montage_{domain}.png"
        if not cv2.imwrite(str(path), montage):
            raise RuntimeError(f"Failed to write montage image: {path}")
        figures[f"montage_{domain}"] = path
    return figures


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
