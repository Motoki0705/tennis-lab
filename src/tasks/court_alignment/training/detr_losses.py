"""Hungarian matching and losses for DINO oriented court boxes.

Predictions follow the DINO contract: sigmoid class logits ``[B,Q,C]``,
normalized axis-aligned ``cxcywh`` boxes ``[B,Q,4]``, and raw court-head
values ``[B,Q,3]``.  Raw court values are decoded as
``(sigmoid(long_logit), normalize(axis_raw))``.  Targets are the variable-
length mappings emitted by :func:`build_detr_court_targets`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

DetrOutputs = Mapping[str, object]
DetrTarget = Mapping[str, Tensor]
MatchIndices = list[tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class CourtDetrLossWeights:
    """Weights shared by Hungarian cost and the final training objective."""

    classification: float = 2.0
    bbox: float = 5.0
    giou: float = 2.0
    scale: float = 2.0
    axis: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.classification,
            self.bbox,
            self.giou,
            self.scale,
            self.axis,
        )
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values):
            raise ValueError("DETR loss weights must be finite and non-negative.")
        if not any(value > 0.0 for value in values):
            raise ValueError("At least one DETR loss weight must be positive.")


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized ``cxcywh`` boxes to ``xyxy`` without clamping."""

    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have final dimension four.")
    return _box_cxcywh_to_xyxy(boxes)


def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert already-validated boxes inside the loss computation graph."""

    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def generalized_box_iou(first_xyxy: Tensor, second_xyxy: Tensor) -> Tensor:
    """Return pairwise generalized IoU for valid ``xyxy`` box matrices."""

    if first_xyxy.ndim != 2 or first_xyxy.shape[-1] != 4:
        raise ValueError("first_xyxy must have shape [N,4].")
    if second_xyxy.ndim != 2 or second_xyxy.shape[-1] != 4:
        raise ValueError("second_xyxy must have shape [M,4].")
    if first_xyxy.device != second_xyxy.device:
        raise ValueError("Box matrices must share a device.")
    if first_xyxy.dtype != second_xyxy.dtype:
        raise ValueError("Box matrices must share a dtype.")
    if first_xyxy.shape[0] == 0 or second_xyxy.shape[0] == 0:
        return first_xyxy.new_empty((first_xyxy.shape[0], second_xyxy.shape[0]))

    first_min, first_max = first_xyxy[:, :2], first_xyxy[:, 2:]
    second_min, second_max = second_xyxy[:, :2], second_xyxy[:, 2:]
    if bool((first_max < first_min).any()) or bool((second_max < second_min).any()):
        raise ValueError("xyxy boxes must have non-negative width and height.")
    return _generalized_box_iou(first_xyxy, second_xyxy)


def _generalized_box_iou(first_xyxy: Tensor, second_xyxy: Tensor) -> Tensor:
    """Compute pairwise GIoU after the explicit boundary has validated boxes."""

    first_min, first_max = first_xyxy[:, :2], first_xyxy[:, 2:]
    second_min, second_max = second_xyxy[:, :2], second_xyxy[:, 2:]
    intersection_min = torch.maximum(first_min[:, None], second_min[None])
    intersection_max = torch.minimum(first_max[:, None], second_max[None])
    intersection = (intersection_max - intersection_min).clamp_min(0.0).prod(dim=-1)
    first_area = (first_max - first_min).prod(dim=-1)
    second_area = (second_max - second_min).prod(dim=-1)
    union = first_area[:, None] + second_area[None] - intersection
    eps = torch.finfo(first_xyxy.dtype).eps
    iou = intersection / union.clamp_min(eps)

    enclosing_min = torch.minimum(first_min[:, None], second_min[None])
    enclosing_max = torch.maximum(first_max[:, None], second_max[None])
    enclosing_area = (enclosing_max - enclosing_min).clamp_min(0.0).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(eps)


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    normalizer: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """DINO-style sigmoid focal loss normalized by target count."""

    if logits.shape != targets.shape:
        raise ValueError("Focal logits and targets must have identical shapes.")
    if logits.ndim != 3:
        raise ValueError("Focal logits and targets must have shape [B,Q,C].")
    if not math.isfinite(float(normalizer)) or normalizer <= 0.0:
        raise ValueError("normalizer must be finite and positive.")
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("Focal alpha/gamma values are invalid.")
    return _sigmoid_focal_loss(
        logits,
        targets,
        normalizer=normalizer,
        alpha=alpha,
        gamma=gamma,
    )


def _sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    normalizer: float,
    alpha: float,
    gamma: float,
) -> Tensor:
    """Compute focal loss for tensors validated at the lifecycle boundary."""

    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_target = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = cross_entropy * (1.0 - probability_target).pow(gamma)
    alpha_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    # This is algebraically DINO's mean-over-query then *Q formulation.
    return (alpha_target * loss).sum() / normalizer


def _required_tensor(
    values: Mapping[str, object],
    name: str,
    *,
    final_dimension: int | None,
    rank: int = 3,
) -> Tensor:
    value = values.get(name)
    if (
        not isinstance(value, Tensor)
        or value.ndim != rank
        or (final_dimension is not None and value.shape[-1] != final_dimension)
    ):
        ending = "C" if final_dimension is None else str(final_dimension)
        raise ValueError(f"{name} must have shape [B,Q,{ending}].")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _prediction_tensors(outputs: Mapping[str, object]) -> tuple[Tensor, Tensor, Tensor]:
    logits, boxes = _detection_prediction_tensors(outputs)
    court_boxes = _required_tensor(outputs, "pred_court_boxes", final_dimension=3)
    if logits.shape[:2] != court_boxes.shape[:2]:
        raise ValueError("All DETR predictions must share batch and query dimensions.")
    if logits.device != court_boxes.device:
        raise ValueError("All DETR predictions must share a device.")
    return logits, boxes, court_boxes


def _detection_prediction_tensors(
    outputs: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    """Validate the class/AABB subset emitted by encoder and DN branches."""

    logits = _required_tensor(outputs, "pred_logits", final_dimension=None)
    boxes = _required_tensor(outputs, "pred_boxes", final_dimension=4)
    if logits.shape[-1] <= 0:
        raise ValueError("pred_logits must contain at least one class.")
    if logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("DETR class and box predictions must share batch/query axes.")
    if logits.device != boxes.device:
        raise ValueError("DETR class and box predictions must share a device.")
    if bool(((boxes < 0.0) | (boxes > 1.0)).any()):
        raise ValueError("pred_boxes must be normalized to [0,1].")
    return logits, boxes


def _detection_prediction_values(
    outputs: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    """Extract class/AABB tensors already validated before ``forward``."""

    return cast(Tensor, outputs["pred_logits"]), cast(Tensor, outputs["pred_boxes"])


def _prediction_values(
    outputs: Mapping[str, object],
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract all task-head tensors already validated before ``forward``."""

    logits, boxes = _detection_prediction_values(outputs)
    return logits, boxes, cast(Tensor, outputs["pred_court_boxes"])


def _decode_court_values(raw_court_boxes: Tensor) -> Tensor:
    """Decode raw scale/axis values after validation at the outer boundary."""

    long_side = raw_court_boxes[..., :1].sigmoid()
    axis = F.normalize(raw_court_boxes[..., 1:], dim=-1, eps=1.0e-8)
    return torch.cat((long_side, axis), dim=-1)


def _validate_targets(
    targets: Sequence[DetrTarget],
    *,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> None:
    if len(targets) != batch_size:
        raise ValueError("targets length must equal prediction batch size.")
    for target in targets:
        labels = target.get("labels")
        boxes = target.get("boxes")
        court_boxes = target.get("court_boxes")
        if (
            not isinstance(labels, Tensor)
            or labels.ndim != 1
            or labels.dtype != torch.long
        ):
            raise TypeError("Target labels must be an int64 tensor [N].")
        count = labels.shape[0]
        if not isinstance(boxes, Tensor) or boxes.shape != (count, 4):
            raise ValueError("Target boxes must have shape [N,4].")
        if not isinstance(court_boxes, Tensor) or court_boxes.shape != (count, 5):
            raise ValueError("Target court_boxes must have shape [N,5].")
        if (
            labels.device != device
            or boxes.device != device
            or court_boxes.device != device
        ):
            raise ValueError("Targets and predictions must share a device.")
        if not boxes.is_floating_point() or not court_boxes.is_floating_point():
            raise TypeError("Target boxes must be floating point.")
        if not bool(torch.isfinite(boxes).all()) or not bool(
            torch.isfinite(court_boxes).all()
        ):
            raise ValueError("Target boxes must contain only finite values.")
        if count > 0:
            if bool(((labels < 0) | (labels >= num_classes)).any()):
                raise ValueError("Target class index is outside pred_logits classes.")
            if bool(((boxes < 0.0) | (boxes > 1.0)).any()):
                raise ValueError("Target boxes must be normalized to [0,1].")
            if bool((court_boxes[:, 2] <= 0.0).any()):
                raise ValueError("Target normalized long sides must be positive.")
            axis_norm = torch.linalg.vector_norm(court_boxes[:, 3:], dim=-1)
            if not bool(
                torch.allclose(
                    axis_norm, torch.ones_like(axis_norm), atol=1.0e-4, rtol=1.0e-4
                )
            ):
                raise ValueError("Target axial vectors must have unit norm.")


class CourtDetrHungarianMatcher(nn.Module):
    """One-to-one DINO matching with court scale and axial orientation costs."""

    def __init__(
        self,
        *,
        weights: CourtDetrLossWeights | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        resolved_weights = CourtDetrLossWeights() if weights is None else weights
        if not isinstance(resolved_weights, CourtDetrLossWeights):
            raise TypeError("weights must be CourtDetrLossWeights.")
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must lie in [0,1].")
        if not math.isfinite(focal_gamma) or focal_gamma < 0.0:
            raise ValueError("focal_gamma must be finite and non-negative.")
        self.weights = resolved_weights
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def forward(
        self, outputs: DetrOutputs, targets: Sequence[DetrTarget]
    ) -> MatchIndices:
        """Match decoder outputs using class, AABB, scale, and court axis."""

        with torch.no_grad():
            return self._match(outputs, targets, include_court=True)

    def validate_inputs(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
        *,
        include_court: bool = True,
    ) -> None:
        """Validate matcher inputs before entering computation-only ``forward``."""

        if include_court:
            logits, _, _ = _prediction_tensors(outputs)
        else:
            logits, _ = _detection_prediction_tensors(outputs)
        _validate_targets(
            targets,
            batch_size=logits.shape[0],
            num_classes=logits.shape[-1],
            device=logits.device,
        )

    def match_detection_outputs(
        self, outputs: DetrOutputs, targets: Sequence[DetrTarget]
    ) -> MatchIndices:
        """Match class/AABB-only outputs from encoder or DN branches."""

        with torch.no_grad():
            return self._match(outputs, targets, include_court=False)

    def _match(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
        *,
        include_court: bool,
    ) -> MatchIndices:
        if include_court:
            logits, boxes, raw_court_boxes = _prediction_values(outputs)
            decoded_court_boxes = _decode_court_values(raw_court_boxes)
        else:
            logits, boxes = _detection_prediction_values(outputs)
            decoded_court_boxes = None
        assignments: MatchIndices = []
        for batch_index, target in enumerate(targets):
            target_labels = target["labels"]
            target_boxes = target["boxes"]
            target_court_boxes = target["court_boxes"]
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                assignments.append((empty, empty))
                continue

            probability = logits[batch_index].sigmoid()
            negative = (1.0 - self.focal_alpha) * probability.pow(self.focal_gamma)
            negative = negative * (-(1.0 - probability + 1.0e-8).log())
            positive = self.focal_alpha * (1.0 - probability).pow(self.focal_gamma)
            positive = positive * (-(probability + 1.0e-8).log())
            classification_cost = (
                positive[:, target_labels] - negative[:, target_labels]
            )
            bbox_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            giou_cost = -_generalized_box_iou(
                _box_cxcywh_to_xyxy(boxes[batch_index]),
                _box_cxcywh_to_xyxy(target_boxes),
            )
            cost = (
                self.weights.classification * classification_cost
                + self.weights.bbox * bbox_cost
                + self.weights.giou * giou_cost
            )
            if decoded_court_boxes is not None:
                predicted_long = decoded_court_boxes[batch_index, :, 0].clamp_min(
                    1.0e-8
                )
                target_long = target_court_boxes[:, 2].clamp_min(1.0e-8)
                scale_cost = torch.cdist(
                    predicted_long.log().unsqueeze(-1),
                    target_long.log().unsqueeze(-1),
                    p=1,
                )
                axis_cost = (
                    1.0
                    - decoded_court_boxes[batch_index, :, 1:]
                    @ target_court_boxes[:, 3:].T
                )
                cost = (
                    cost
                    + self.weights.scale * scale_cost
                    + self.weights.axis * axis_cost
                )
            prediction_indices, target_indices = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            assignments.append(
                (
                    torch.as_tensor(
                        prediction_indices, dtype=torch.long, device=logits.device
                    ),
                    torch.as_tensor(
                        target_indices, dtype=torch.long, device=logits.device
                    ),
                )
            )
        return assignments


def _matched_values(
    prediction: Tensor,
    targets: Sequence[DetrTarget],
    assignments: MatchIndices,
    target_name: str,
) -> tuple[Tensor, Tensor]:
    prediction_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for batch_index, (prediction_indices, target_indices) in enumerate(assignments):
        if prediction_indices.numel() == 0:
            continue
        prediction_parts.append(prediction[batch_index, prediction_indices])
        target_parts.append(targets[batch_index][target_name][target_indices])
    if not prediction_parts:
        return (
            prediction.reshape(-1, prediction.shape[-1])[:0],
            targets[0][target_name][:0],
        )
    return torch.cat(prediction_parts), torch.cat(target_parts)


class CourtDetrCriterion(nn.Module):
    """Compute DINO focal/AABB losses plus matched court scale/axis losses.

    ``loss_total`` includes every main and auxiliary decoder-layer component.
    Auxiliary entries are suffixed ``_aux_<index>`` and use a fresh Hungarian
    assignment, matching DINO's standard deep-supervision behavior.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1,
        weights: CourtDetrLossWeights | None = None,
        matcher: CourtDetrHungarianMatcher | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        auxiliary_loss: bool = True,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer.")
        resolved_weights = CourtDetrLossWeights() if weights is None else weights
        if not isinstance(resolved_weights, CourtDetrLossWeights):
            raise TypeError("weights must be CourtDetrLossWeights.")
        if matcher is not None and not isinstance(matcher, CourtDetrHungarianMatcher):
            raise TypeError("matcher must be CourtDetrHungarianMatcher or None.")
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must lie in [0,1].")
        if not math.isfinite(focal_gamma) or focal_gamma < 0.0:
            raise ValueError("focal_gamma must be finite and non-negative.")
        if not isinstance(auxiliary_loss, bool):
            raise TypeError("auxiliary_loss must be boolean.")
        self.num_classes = num_classes
        self.weights = resolved_weights
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.auxiliary_loss = auxiliary_loss
        self.matcher = matcher or CourtDetrHungarianMatcher(
            weights=resolved_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    def _validate_detection_layer(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
    ) -> tuple[Tensor, Tensor]:
        logits, boxes = _detection_prediction_tensors(outputs)
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                "pred_logits class count does not match criterion num_classes."
            )
        _validate_targets(
            targets,
            batch_size=logits.shape[0],
            num_classes=self.num_classes,
            device=logits.device,
        )
        return logits, boxes

    def _validate_court_layer(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, boxes, court_boxes = _prediction_tensors(outputs)
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                "pred_logits class count does not match criterion num_classes."
            )
        _validate_targets(
            targets,
            batch_size=logits.shape[0],
            num_classes=self.num_classes,
            device=logits.device,
        )
        return logits, boxes, court_boxes

    def _validate_denoising_inputs(
        self,
        dn_meta: object,
        targets: Sequence[DetrTarget],
    ) -> None:
        if dn_meta is None:
            return
        if not isinstance(dn_meta, Mapping):
            raise TypeError("dn_meta must be a mapping or None.")
        if "pad_size" not in dn_meta or "num_dn_group" not in dn_meta:
            raise ValueError("dn_meta requires pad_size and num_dn_group.")
        pad_size = dn_meta["pad_size"]
        num_dn_groups = dn_meta["num_dn_group"]
        if isinstance(pad_size, bool) or not isinstance(pad_size, int) or pad_size < 0:
            raise ValueError("dn_meta.pad_size must be a non-negative integer.")
        if (
            isinstance(num_dn_groups, bool)
            or not isinstance(num_dn_groups, int)
            or num_dn_groups <= 0
        ):
            raise ValueError("dn_meta.num_dn_group must be a positive integer.")
        if pad_size % num_dn_groups != 0:
            raise ValueError("DN pad_size must be divisible by num_dn_group.")
        if pad_size == 0:
            if "output_known_lbs_bboxes" in dn_meta:
                raise ValueError(
                    "Zero-size DN metadata must not contain known outputs."
                )
            return

        known_outputs = dn_meta.get("output_known_lbs_bboxes")
        if not isinstance(known_outputs, Mapping):
            raise ValueError("Active dn_meta requires output_known_lbs_bboxes.")
        logits, boxes = self._validate_detection_layer(known_outputs, targets)
        if logits.shape[1] != pad_size:
            raise ValueError("DN known-output query count must equal pad_size.")
        group_width = pad_size // num_dn_groups
        if group_width % 2 != 0:
            raise ValueError("Each DN group must have equal positive/negative halves.")
        positive_capacity = group_width // 2
        if any(int(target["labels"].shape[0]) > positive_capacity for target in targets):
            raise ValueError(
                "DN positive half is smaller than this sample's target count."
            )

        auxiliary = known_outputs.get("aux_outputs")
        if auxiliary is not None:
            if not isinstance(auxiliary, Sequence) or isinstance(
                auxiliary, (str, bytes)
            ):
                raise TypeError("DN known aux_outputs must be a sequence.")
            for layer_outputs in auxiliary:
                if not isinstance(layer_outputs, Mapping):
                    raise TypeError("Each DN aux output must be a mapping.")
                aux_logits, aux_boxes = self._validate_detection_layer(
                    layer_outputs, targets
                )
                if aux_logits.shape != logits.shape or aux_boxes.shape != boxes.shape:
                    raise ValueError(
                        "DN auxiliary outputs must match main known-output shapes."
                    )

    def validate_inputs(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
    ) -> None:
        """Validate all DINO loss branches before computation-only ``forward``."""

        self._validate_court_layer(outputs, targets)
        auxiliary = outputs.get("aux_outputs")
        if self.auxiliary_loss and auxiliary is not None:
            if not isinstance(auxiliary, Sequence) or isinstance(
                auxiliary, (str, bytes)
            ):
                raise TypeError(
                    "aux_outputs must be a sequence of prediction mappings."
                )
            for layer_outputs in auxiliary:
                if not isinstance(layer_outputs, Mapping):
                    raise TypeError(
                        "Each aux_outputs item must be a prediction mapping."
                    )
                self._validate_court_layer(layer_outputs, targets)

        self._validate_denoising_inputs(outputs.get("dn_meta"), targets)
        intermediate = outputs.get("interm_outputs")
        if intermediate is not None:
            if not isinstance(intermediate, Mapping):
                raise TypeError("interm_outputs must be a prediction mapping.")
            self._validate_detection_layer(intermediate, targets)

        encoder_outputs = outputs.get("enc_outputs")
        if encoder_outputs is not None:
            if isinstance(encoder_outputs, Mapping):
                encoder_layers: Sequence[object] = (encoder_outputs,)
            elif isinstance(encoder_outputs, Sequence) and not isinstance(
                encoder_outputs, (str, bytes)
            ):
                encoder_layers = encoder_outputs
            else:
                raise TypeError("enc_outputs must be a mapping or sequence.")
            for layer_outputs in encoder_layers:
                if not isinstance(layer_outputs, Mapping):
                    raise TypeError("Each enc_outputs item must be a mapping.")
                self._validate_detection_layer(layer_outputs, targets)

    def _detection_losses_with_assignments(
        self,
        logits: Tensor,
        boxes: Tensor,
        targets: Sequence[DetrTarget],
        assignments: MatchIndices,
        *,
        normalizer: float,
        suffix: str,
    ) -> dict[str, Tensor]:
        """Compute class/AABB losses for Hungarian or fixed DN assignments."""

        class_targets = torch.zeros_like(logits)
        for batch_index, (prediction_indices, target_indices) in enumerate(assignments):
            if prediction_indices.numel() > 0:
                labels = targets[batch_index]["labels"][target_indices]
                class_targets[batch_index, prediction_indices, labels] = 1.0
        classification = _sigmoid_focal_loss(
            logits,
            class_targets,
            normalizer=normalizer,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        matched_boxes, target_boxes = _matched_values(
            boxes, targets, assignments, "boxes"
        )
        bbox = F.l1_loss(matched_boxes, target_boxes, reduction="sum") / normalizer
        pairwise_giou = _generalized_box_iou(
            _box_cxcywh_to_xyxy(matched_boxes),
            _box_cxcywh_to_xyxy(target_boxes),
        )
        giou = (1.0 - pairwise_giou.diag()).sum() / normalizer
        return {
            f"loss_class{suffix}": classification,
            f"loss_bbox{suffix}": bbox,
            f"loss_giou{suffix}": giou,
        }

    def _detection_losses_for_layer(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
        *,
        suffix: str,
    ) -> dict[str, Tensor]:
        """Hungarian-match and supervise an encoder class/AABB output."""

        logits, boxes = _detection_prediction_values(outputs)
        assignments = self.matcher.match_detection_outputs(outputs, targets)
        target_count = sum(int(target["labels"].shape[0]) for target in targets)
        return self._detection_losses_with_assignments(
            logits,
            boxes,
            targets,
            assignments,
            normalizer=float(max(target_count, 1)),
            suffix=suffix,
        )

    def _losses_for_layer(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
        *,
        suffix: str,
    ) -> dict[str, Tensor]:
        logits, boxes, raw_court_boxes = _prediction_values(outputs)
        assignments = self.matcher(outputs, targets)
        target_count = sum(int(target["labels"].shape[0]) for target in targets)
        normalizer = float(max(target_count, 1))
        losses = self._detection_losses_with_assignments(
            logits,
            boxes,
            targets,
            assignments,
            normalizer=normalizer,
            suffix=suffix,
        )
        decoded_court_boxes = _decode_court_values(raw_court_boxes)
        matched_court, target_court = _matched_values(
            decoded_court_boxes, targets, assignments, "court_boxes"
        )
        scale = (
            F.l1_loss(
                matched_court[:, 0].clamp_min(1.0e-8).log(),
                target_court[:, 2].clamp_min(1.0e-8).log(),
                reduction="sum",
            )
            / normalizer
        )
        axis = (
            1.0 - (matched_court[:, 1:] * target_court[:, 3:]).sum(dim=-1)
        ).sum()
        axis = axis / normalizer
        losses[f"loss_scale{suffix}"] = scale
        losses[f"loss_axis{suffix}"] = axis
        return losses

    def _denoising_assignments(
        self,
        targets: Sequence[DetrTarget],
        *,
        pad_size: int,
        num_dn_groups: int,
        device: torch.device,
    ) -> MatchIndices:
        """Build official DINO fixed positive-slot target assignments."""

        group_width = pad_size // num_dn_groups
        assignments: MatchIndices = []
        for target in targets:
            target_count = int(target["labels"].shape[0])
            if target_count == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                assignments.append((empty, empty))
                continue
            target_indices = torch.arange(target_count, dtype=torch.long, device=device)
            group_offsets = (
                torch.arange(num_dn_groups, dtype=torch.long, device=device)
                * group_width
            )
            prediction_indices = (
                group_offsets[:, None] + target_indices[None, :]
            ).reshape(-1)
            assignments.append(
                (prediction_indices, target_indices.repeat(num_dn_groups))
            )
        return assignments

    def _denoising_losses(
        self,
        dn_meta: object,
        targets: Sequence[DetrTarget],
    ) -> dict[str, Tensor]:
        """Supervise official DINO known DN outputs without Hungarian matching."""

        if dn_meta is None:
            return {}
        typed_dn_meta = cast(Mapping[str, object], dn_meta)
        pad_size = cast(int, typed_dn_meta["pad_size"])
        num_dn_groups = cast(int, typed_dn_meta["num_dn_group"])
        if pad_size == 0:
            return {}

        known_outputs = cast(
            Mapping[str, object], typed_dn_meta["output_known_lbs_bboxes"]
        )
        logits, boxes = _detection_prediction_values(known_outputs)
        assignments = self._denoising_assignments(
            targets,
            pad_size=pad_size,
            num_dn_groups=num_dn_groups,
            device=logits.device,
        )
        positive_count = sum(int(target["labels"].shape[0]) for target in targets)
        normalizer = float(max(positive_count * num_dn_groups, 1))
        losses = self._detection_losses_with_assignments(
            logits,
            boxes,
            targets,
            assignments,
            normalizer=normalizer,
            suffix="_dn",
        )

        auxiliary = known_outputs.get("aux_outputs")
        if auxiliary is not None:
            for index, layer_outputs in enumerate(
                cast(Sequence[Mapping[str, object]], auxiliary)
            ):
                aux_logits, aux_boxes = _detection_prediction_values(layer_outputs)
                if self.auxiliary_loss:
                    losses.update(
                        self._detection_losses_with_assignments(
                            aux_logits,
                            aux_boxes,
                            targets,
                            assignments,
                            normalizer=normalizer,
                            suffix=f"_dn_aux_{index}",
                        )
                    )
        return losses

    def _weighted_sum(self, losses: Mapping[str, Tensor]) -> Tensor:
        weighted = losses["loss_class"] * 0.0
        weight_by_prefix = {
            "loss_class": self.weights.classification,
            "loss_bbox": self.weights.bbox,
            "loss_giou": self.weights.giou,
            "loss_scale": self.weights.scale,
            "loss_axis": self.weights.axis,
        }
        for name, value in losses.items():
            weight = next(
                (
                    item
                    for prefix, item in weight_by_prefix.items()
                    if name.startswith(prefix)
                ),
                None,
            )
            if weight is None:
                continue
            term = value * weight
            weighted = weighted + term
        return weighted

    def forward(
        self,
        outputs: DetrOutputs,
        targets: Sequence[DetrTarget],
    ) -> dict[str, Tensor]:
        """Return named components and their weighted ``loss_total``."""

        losses = self._losses_for_layer(outputs, targets, suffix="")
        auxiliary = outputs.get("aux_outputs")
        if self.auxiliary_loss and auxiliary is not None:
            for index, layer_outputs in enumerate(
                cast(Sequence[Mapping[str, object]], auxiliary)
            ):
                losses.update(
                    self._losses_for_layer(
                        layer_outputs, targets, suffix=f"_aux_{index}"
                    )
                )
        losses.update(self._denoising_losses(outputs.get("dn_meta"), targets))

        intermediate = outputs.get("interm_outputs")
        if intermediate is not None:
            losses.update(
                self._detection_losses_for_layer(
                    cast(Mapping[str, object], intermediate),
                    targets,
                    suffix="_interm",
                )
            )

        encoder_outputs = outputs.get("enc_outputs")
        if encoder_outputs is not None:
            typed_encoder_outputs = cast(
                Mapping[str, object] | Sequence[Mapping[str, object]],
                encoder_outputs,
            )
            encoder_layers: Sequence[Mapping[str, object]]
            if "pred_logits" in typed_encoder_outputs:
                encoder_layers = (
                    cast(Mapping[str, object], typed_encoder_outputs),
                )
            else:
                encoder_layers = cast(
                    Sequence[Mapping[str, object]], typed_encoder_outputs
                )
            for index, layer_outputs in enumerate(encoder_layers):
                losses.update(
                    self._detection_losses_for_layer(
                        layer_outputs,
                        targets,
                        suffix=f"_enc_{index}",
                    )
                )
        losses["loss_total"] = self._weighted_sum(losses)
        return losses


__all__ = [
    "CourtDetrCriterion",
    "CourtDetrHungarianMatcher",
    "CourtDetrLossWeights",
    "box_cxcywh_to_xyxy",
    "generalized_box_iou",
    "sigmoid_focal_loss",
]
