"""Analyze which PLCS loss term is dominant for a trained checkpoint.

Loads a trained checkpoint together with the ``hparams.yaml`` from its run
directory, evaluates it on a split, and reports for every registered
:class:`PLCSLoss` term its raw loss, configured weight, and weighted
contribution to the total. This shows which term dominates the objective and
how large the pose-naturalness terms (``joint_angle``, ``torsion_angle``,
``torso_twist``, ``bone_length``) are relative to the supervised ones.

Usage:
    python -m src.tasks.plcs.scripts.analysis.analyze_loss_dominance
    python -m src.tasks.plcs.scripts.analysis.analyze_loss_dominance \
        run.checkpoint=path/to/last.ckpt run.hparams=path/to/hparams.yaml \
        analysis.split=test analysis.max_batches=10

Notes:
    - Analysis settings are loaded from
      `src/tasks/plcs/configs/analyze_loss_dominance.yaml` via Hydra.
    - Model weights come from `run.checkpoint`; all loss/metric/data settings
      come from the run's `run.hparams` (`hparams.yaml`), so the report matches
      the trained configuration. Temporal losses are intentionally excluded.
    - The JSON report is written to `run.output_dir/analysis.output_filename`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


def _load_hparams_config(hparams_path: Path) -> DictConfig:
    """Load the ``config`` block stored in a Lightning ``hparams.yaml``."""
    raw = OmegaConf.load(hparams_path)
    config = raw.get("config", raw)
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)
    return config


def _running_mean(state: dict[str, float], key: str, value: float, count: int) -> None:
    prev = state.get(key, 0.0)
    state[key] = prev + (value - prev) / count


def analyze(
    checkpoint: Path,
    hparams: Path,
    split: str,
    device: torch.device,
    max_batches: int | None,
    loss_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the dominance analysis and return a result dict."""
    config = _load_hparams_config(hparams)

    # Model weights from the checkpoint, all settings from hparams.yaml.
    module = PLCSLightningModule.load_from_checkpoint(
        str(checkpoint),
        config=config,
        map_location=device,
        weights_only=False,
    )
    module.eval().to(device)
    model = module.model

    # PLCSLoss is the single source of truth for which terms exist and their
    # weights, so the analysis automatically tracks every registered term
    # (including the pose-naturalness losses) without hard-coded names. Loss
    # weights default to the checkpoint's hparams, optionally overridden to
    # preview a candidate weighting.
    loss_weights = dict(config.get("loss", {}))
    if loss_overrides:
        loss_weights.update(loss_overrides)
    loss_fn = PLCSLoss(config=PLCSLossConfig.from_dict(loss_weights))
    term_names = tuple(loss_fn.loss_terms)

    metrics_cfg = config.get("metrics", {})
    metrics = PLCSMetrics(
        position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
        angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
    )

    torch.manual_seed(int(config.get("run", {}).get("seed", 42)))
    datamodule = PLCSDataModule(config)
    datamodule.setup("test")
    if split != "test":
        datamodule.test_dataset = datamodule._build_dataset(
            scene_dir=datamodule.scene_dir,
            split_file=f"{split}.txt",
            augment=False,
        )
    loader = datamodule.test_dataloader()

    raw_loss: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(
                human_kp=batch["human_kp"],
                court_kp=batch["court_kp"],
                human_vis=batch.get("human_vis"),
                human_mask=batch.get("human_mask"),
                court_vis=batch.get("court_vis"),
            )
            losses = loss_fn(
                pred_position=outputs["position"],
                pred_rotation=outputs["rotation"],
                target_position=batch["position"],
                target_rotation=batch["rotation"],
                pred_canonical_pose=outputs.get("canonical_pose"),
                target_human_kp_3d=batch.get("human_kp_3d"),
                human_mask=batch.get("human_mask"),
            )
            metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
                human_mask=batch.get("human_mask"),
            )

            count += 1
            for name in term_names:
                _running_mean(raw_loss, name, float(losses[name].item()), count)

    weights = {name: loss_fn.weight_for(name) for name in term_names}
    weighted = {name: raw_loss[name] * weights[name] for name in term_names}
    total_weighted = sum(weighted.values())
    share = {
        name: (weighted[name] / total_weighted if total_weighted > 0 else 0.0)
        for name in term_names
    }

    return {
        "checkpoint": str(checkpoint),
        "hparams": str(hparams),
        "split": split,
        "num_batches": count,
        "terms": list(term_names),
        "weights": weights,
        "raw_loss": dict(raw_loss),
        "weighted_loss": weighted,
        "weighted_share": share,
        "total_weighted_loss": total_weighted,
        "metrics": metrics.compute(),
    }


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 72)
    print("PLCS loss dominance analysis")
    print("=" * 72)
    print(f"checkpoint : {result['checkpoint']}")
    print(f"hparams    : {result['hparams']}")
    print(f"split      : {result['split']}  ({result['num_batches']} batches)")
    print("-" * 72)
    print(f"{'term':<16}{'raw':>12}{'weight':>10}{'weighted':>14}{'share':>10}")
    print("-" * 72)
    for term in result["terms"]:
        print(
            f"{term:<16}"
            f"{result['raw_loss'][term]:>12.5f}"
            f"{result['weights'][term]:>10.3f}"
            f"{result['weighted_loss'][term]:>14.5f}"
            f"{result['weighted_share'][term] * 100:>9.1f}%"
        )
    print("-" * 72)
    print(
        f"{'total':<16}{'':>12}{'':>10}"
        f"{result['total_weighted_loss']:>14.5f}{'100.0%':>10}"
    )
    print("-" * 72)
    dominant = max(result["terms"], key=lambda t: result["weighted_loss"][t])
    print(f"Dominant (weighted) loss term: {dominant}")
    print("-" * 72)
    m = result["metrics"]
    print(
        "position_error_m={:.4f}  angular_error_deg={:.2f}  "
        "pos_acc@0.5m={:.3f}  ang_acc@15deg={:.3f}".format(
            m.get("position_error_m", 0.0),
            m.get("angular_error_deg", 0.0),
            m.get("position_accuracy_0.5m", 0.0),
            m.get("angle_accuracy_15deg", 0.0),
        )
    )
    print("=" * 72)


def _resolve_device(device_cfg: Any) -> torch.device:
    if device_cfg is not None:
        return torch.device(str(device_cfg))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra_main(
    config_path="../../configs",
    config_name="analyze_loss_dominance",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    max_batches = cfg.analysis.max_batches
    loss_overrides: dict[str, Any] | None = None
    if cfg.analysis.loss_config is not None:
        loaded = OmegaConf.load(to_absolute_path(str(cfg.analysis.loss_config)))
        loss_overrides = cast(dict[str, Any], OmegaConf.to_container(loaded, resolve=True))
    result = analyze(
        checkpoint=Path(to_absolute_path(str(cfg.run.checkpoint))),
        hparams=Path(to_absolute_path(str(cfg.run.hparams))),
        split=str(cfg.analysis.split),
        device=_resolve_device(cfg.analysis.device),
        max_batches=int(max_batches) if max_batches is not None else None,
        loss_overrides=loss_overrides,
    )
    _print_report(result)

    out_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(cfg.analysis.output_filename)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved JSON report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cast(Callable[[], int], main)())
