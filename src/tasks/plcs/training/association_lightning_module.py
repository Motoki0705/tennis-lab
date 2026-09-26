"""PLCS lifecycle for independently trained track Re-ID or camera-side models."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.plcs.model_io.factory import compose_plcs_person_model_io
from src.tasks.plcs.model_io.person_association import (
    MODEL_CONTRACTS,
    REID_MODEL,
    validate_person_checkpoint,
)
from src.tasks.plcs.model_io.track_matching import match_track_embeddings
from src.tasks.plcs.training.reid_losses import court_side_loss, reid_loss, reid_pairs
from src.utils.configuration import PathRole


class PLCSAssociationLightningModule(BaseLightningModule):
    """One model, one optimizer, one objective and one checkpoint contract."""
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model_io = compose_plcs_person_model_io(config)
        self.model = self.model_io.model
        self.is_reid = str(config.model.name) == REID_MODEL
        initial = float(config.metrics.cosine_threshold) if self.is_reid else 0.
        self.register_buffer("matching_threshold", torch.tensor(initial))
        self.register_buffer("calibration_candidates", torch.linspace(-.2, .95, 47), persistent=False)
        self._statistics: dict[str, dict[str, Tensor]] = {}
        self._test_pred_arrays: dict[str, list[Any]] = defaultdict(list)
        self._test_pred_scene_ids: list[str] = []

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self.model_io.run(batch)
        if self.is_reid:
            values = reid_loss(output, batch["track_person_id"], temperature=float(self.config.loss.temperature),
                margin=float(self.config.loss.margin))
            with torch.no_grad():
                scores, labels, mask = reid_pairs(output, batch["track_person_id"])
                thresholds = self.calibration_candidates if stage == "val" else self.matching_threshold[None]
                pred = scores[..., None] > thresholds
                selected, same = mask[..., None], labels[..., None]
                axes = (0, 1, 2)
                values.update(pair_tp=(pred & same & selected).sum(axes), pair_fp=(pred & ~same & selected).sum(axes),
                    pair_fn=(~pred & same & selected).sum(axes), pair_tn=(~pred & ~same & selected).sum(axes))
            if stage == "test":
                ids = torch.stack([match_track_embeddings(z, valid, threshold=float(self.matching_threshold.cpu()))
                    for z, valid in zip(output["track_embedding"], output["track_valid"], strict=True)]).to(labels.device)
                flat = ids.flatten(1)
                matched = flat[:, :, None].eq(flat[:, None, :]) & flat[:, :, None].ge(0) & flat[:, None, :].ge(0)
                target_present = output["track_valid"] & batch["track_person_id"].ge(0)
                complete = (ids.ge(0) | ~target_present).flatten(1).all(-1)
                values.update(match_tp=(matched & labels & mask).sum(), match_fp=(matched & ~labels & mask).sum(), match_fn=(~matched & labels & mask).sum(),
                    group_correct=((matched.eq(labels) | ~mask).flatten(1).all(-1) & complete & target_present.flatten(1).any(-1)).sum(),
                    group_count=target_present.flatten(1).any(-1).sum())
                self._save_arrays({"track_embedding": output["track_embedding"], "track_valid": output["track_valid"],
                    "slot_global_ids": ids,
                    "track_person_id": batch["track_person_id"], "sample_index": batch["sample_index"],
                    "side_target": batch["side_target"], "reference_view_index": batch["reference_view_index"],
                    "track_observation_count": batch["human_vis"].any(-1).sum(2)})
        else:
            values = court_side_loss(output, batch["side_target"], batch["padding_mask"], batch["reference_view_index"])
            values["loss"] = values["loss"] * float(self.config.loss.weight)
            if stage == "test":
                self._save_arrays({"side_logits": output["side_logits"], "side_target": batch["side_target"],
                    "reference_view_index": batch["reference_view_index"], "sample_index": batch["sample_index"]})
        if not bool(torch.isfinite(values["loss"])):
            raise FloatingPointError(f"{stage}: non-finite person loss; sample_indices={batch['sample_index'].tolist()}")
        state = self._statistics.setdefault(stage, {})
        for key, value in values.items():
            if key != "loss":
                state[key] = state.get(key, torch.zeros_like(value)) + value.detach()
        if stage == "train":
            self.log("train/step_loss", values["loss"], on_step=True, on_epoch=False, batch_size=len(batch["human_kp"]))
        return values["loss"]

    def _save_arrays(self, arrays: dict[str, Tensor]) -> None:
        for key, value in arrays.items():
            self._test_pred_arrays[key].append(value.detach().cpu().numpy())
        self._test_pred_scene_ids.extend(f"test-index:{i}" for i in arrays["sample_index"].tolist())

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def _finish_epoch(self, stage: str) -> dict[str, float]:
        state = self._statistics.pop(stage, {})
        if not state:
            return {}
        def ratio(a: str, b: str) -> Tensor:
            return state[a] / state[b].clamp_min(1)
        if self.is_reid:
            tp, fp, fn, tn = (state[f"pair_{key}"].float() for key in ("tp", "fp", "fn", "tn"))
            f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
            chosen = 0
            if stage == "val" and not self.trainer.sanity_checking:
                # Among identical F1 scores prefer the most conservative threshold.
                chosen = len(f1) - 1 - int(f1.flip(0).argmax())
                self.matching_threshold.copy_(self.calibration_candidates[chosen])
            terms = (state["positive_count"] > 0).float() + (state["negative_count"] > 0).float()
            loss = (ratio("positive_loss_sum", "positive_count") + ratio("negative_loss_sum", "negative_count")) / terms.clamp_min(1)
            metrics = {"loss": loss, "pair_precision": tp[chosen] / (tp[chosen] + fp[chosen]).clamp_min(1),
                "pair_recall": tp[chosen] / (tp[chosen] + fn[chosen]).clamp_min(1), "pair_f1": f1[chosen],
                "pair_balanced_accuracy": .5 * (tp[chosen] / (tp[chosen] + fn[chosen]).clamp_min(1) + tn[chosen] / (tn[chosen] + fp[chosen]).clamp_min(1)),
                "cosine_threshold": self.matching_threshold}
            if stage == "test":
                a, b, c = (state[f"match_{key}"].float() for key in ("tp", "fp", "fn"))
                metrics.update(matching_precision=a / (a + b).clamp_min(1), matching_recall=a / (a + c).clamp_min(1),
                    matching_f1=2 * a / (2 * a + b + c).clamp_min(1), group_accuracy=ratio("group_correct", "group_count"))
        else:
            metrics = {"loss": ratio("side_loss_sum", "side_count") * float(self.config.loss.weight),
                "side_balanced_accuracy": .5 * (ratio("same_correct", "same_count") + ratio("opposite_correct", "opposite_count")),
                "side_accuracy": (state["same_correct"] + state["opposite_correct"]) / state["side_count"].clamp_min(1)}
        self.log_dict({f"{stage}/{key}": value for key, value in metrics.items()}, on_step=False, on_epoch=True)
        record = {key: float(value.detach().cpu()) for key, value in metrics.items()}
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            path = self.path_resolver.resolve(PathRole.OUTPUT, str(self.config.run.output_dir), "association_metrics.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"stage": stage, "epoch": self.current_epoch, "step": self.global_step, **record}, allow_nan=False) + "\n")
        return record

    def on_train_epoch_end(self) -> None:
        self._finish_epoch("train")

    def on_validation_epoch_end(self) -> None:
        self._finish_epoch("val")

    def on_test_epoch_start(self) -> None:
        self._test_pred_arrays.clear()
        self._test_pred_scene_ids.clear()

    def on_test_epoch_end(self) -> None:
        metrics = self._finish_epoch("test")
        self.save_test_predictions(metrics=metrics)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        name = str(self.config.model.name)
        checkpoint["person_association_model"] = name
        checkpoint["person_association_contract"] = MODEL_CONTRACTS[name]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_person_checkpoint(checkpoint, model_name=str(self.config.model.name))
        if checkpoint.get("weights_only_export") and self.config.run.resume is not None:
            raise ValueError("Exported Re-ID weights support inference or run.init_weights, not optimizer resume")
