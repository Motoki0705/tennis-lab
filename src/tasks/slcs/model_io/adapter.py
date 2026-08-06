"""Sole-model SLCS adapter with strict pre-forward tensor validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    ModelOutputContractError,
    TensorSpec,
    require_tensor,
)
from src.tasks.slcs.model_io.contracts import (
    SLCSDecodedOutput,
    SLCSPhysicalOutput,
    SLCSRawOutput,
    SLCSTrainingTargets,
)
from src.tasks.slcs.models.slcs_model import SLCSFusionModel
from src.utils.models.transformer_utils import build_self_attn_mask
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.player import NUM_HUMAN_KP

_FLOAT32 = frozenset({torch.float32})
_BOOL = frozenset({torch.bool})
_INT64 = frozenset({torch.int64})
_UV_TOLERANCE = 0.25
_OUTPUT_KEYS = frozenset(SLCSRawOutput.__required_keys__)


@dataclass(frozen=True, slots=True)
class SLCSModelIOSpec:
    """Model/data dimensions needed to validate every external batch."""

    num_players: int
    num_court_kp: int
    max_seq_len: int
    dino_num_tokens: int
    dino_encoded_num_tokens: int
    dino_embed_dim: int
    log_b_min: float
    log_b_max: float


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ModelInputContractError(f"{name} contains non-finite values.")


def _unit_interval(name: str, value: Tensor) -> None:
    _finite(name, value)
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ModelInputContractError(f"{name} values must lie in [0, 1].")


def _visible_uv(name: str, uv: Tensor, visible: Tensor) -> None:
    values = uv[visible]
    if values.numel() == 0:
        return
    _finite(name, values)
    if bool(
        ((values < -_UV_TOLERANCE) | (values > 1.0 + _UV_TOLERANCE)).any()
    ):
        raise ModelInputContractError(
            f"{name} visible values must be normalized UV within "
            f"[{-_UV_TOLERANCE}, {1.0 + _UV_TOLERANCE}]."
        )


class SLCSModelIOAdapter:
    """Validate SLCS batches and decode the sole model's raw tensor mapping."""

    def __init__(self, spec: SLCSModelIOSpec) -> None:
        self.spec = spec

    @property
    def model_type(self) -> type[nn.Module]:
        return SLCSFusionModel

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Validate observations and create an immutable model invocation."""
        spec = self.spec
        player_kp = require_tensor(
            batch,
            "player_kp",
            spec=TensorSpec(
                shape=(None, spec.num_players, None, NUM_HUMAN_KP, 2),
                dtypes=_FLOAT32,
            ),
        )
        batch_size, _, seq_len = player_kp.shape[:3]
        if batch_size <= 0 or not 0 < seq_len <= spec.max_seq_len:
            raise ModelInputContractError(
                f"player_kp requires B>0 and 0<T<={spec.max_seq_len}, got "
                f"B={batch_size}, T={seq_len}."
            )

        player_kp_vis = require_tensor(
            batch,
            "player_kp_vis",
            spec=TensorSpec(
                shape=(batch_size, spec.num_players, seq_len, NUM_HUMAN_KP),
                dtypes=_FLOAT32,
            ),
        )
        player_valid = require_tensor(
            batch,
            "player_valid",
            spec=TensorSpec(
                shape=(batch_size, spec.num_players, seq_len), dtypes=_BOOL
            ),
        )
        ball_uv = require_tensor(
            batch,
            "ball_uv",
            spec=TensorSpec(shape=(batch_size, seq_len, 2), dtypes=_FLOAT32),
        )
        ball_vis = require_tensor(
            batch,
            "ball_vis",
            spec=TensorSpec(shape=(batch_size, seq_len), dtypes=_BOOL),
        )
        court_kp = require_tensor(
            batch,
            "court_kp",
            spec=TensorSpec(
                shape=(batch_size, seq_len, spec.num_court_kp, 2),
                dtypes=_FLOAT32,
            ),
        )
        court_vis = require_tensor(
            batch,
            "court_vis",
            spec=TensorSpec(
                shape=(batch_size, seq_len, spec.num_court_kp), dtypes=_FLOAT32
            ),
        )
        frame_mask = require_tensor(
            batch,
            "frame_mask",
            spec=TensorSpec(shape=(batch_size, seq_len), dtypes=_BOOL),
        )
        dino_tokens = require_tensor(
            batch,
            "dino_tokens",
            spec=TensorSpec(
                shape=(batch_size, None, spec.dino_num_tokens, spec.dino_embed_dim),
                dtypes=_FLOAT32,
            ),
        )
        dino_samples = dino_tokens.shape[1]
        if dino_samples <= 0:
            raise ModelInputContractError(
                "dino_tokens must retain at least one padded sample slot."
            )
        dino_frame_idx = require_tensor(
            batch,
            "dino_frame_idx",
            spec=TensorSpec(shape=(batch_size, dino_samples), dtypes=_INT64),
        )
        dino_valid = require_tensor(
            batch,
            "dino_valid",
            spec=TensorSpec(shape=(batch_size, dino_samples), dtypes=_BOOL),
        )

        if not bool(frame_mask.any(dim=1).all()):
            raise ModelInputContractError(
                "frame_mask must contain at least one real frame per sample."
            )
        if seq_len > 1 and bool((frame_mask[:, 1:] & ~frame_mask[:, :-1]).any()):
            raise ModelInputContractError(
                "frame_mask must be a contiguous real-frame prefix."
            )
        if bool((player_valid & ~frame_mask.unsqueeze(1)).any()):
            raise ModelInputContractError(
                "player_valid cannot mark a padded frame as observed."
            )
        if bool((ball_vis & ~frame_mask).any()):
            raise ModelInputContractError(
                "ball_vis cannot mark a padded frame as observed."
            )
        observed_players = (player_kp_vis > 0).any(dim=-1)
        if not torch.equal(player_valid, observed_players):
            raise ModelInputContractError(
                "player_valid must exactly match nonzero player_kp_vis observations."
            )

        _unit_interval("player_kp_vis", player_kp_vis)
        _unit_interval("court_vis", court_vis)
        _visible_uv("player_kp", player_kp, player_kp_vis > 0)
        _visible_uv("ball_uv", ball_uv, ball_vis)
        _visible_uv("court_kp", court_kp, court_vis > 0)
        _finite("dino_tokens", dino_tokens)

        valid_indices = dino_frame_idx[dino_valid]
        if valid_indices.numel() and bool(
            ((valid_indices < 0) | (valid_indices >= seq_len)).any()
        ):
            raise ModelInputContractError(
                f"valid dino_frame_idx values must lie in [0, {seq_len})."
            )
        safe_indices = dino_frame_idx.clamp(min=0, max=seq_len - 1)
        if bool((dino_valid & ~frame_mask.gather(1, safe_indices)).any()):
            raise ModelInputContractError(
                "dino_valid cannot reference a padded frame."
            )
        if dino_samples > 1 and bool((dino_valid[:, 1:] & ~dino_valid[:, :-1]).any()):
            raise ModelInputContractError(
                "dino_valid must be a contiguous valid-sample prefix."
            )
        for sample_idx in range(batch_size):
            indices = dino_frame_idx[sample_idx][dino_valid[sample_idx]]
            if indices.numel() > 1 and bool((indices[1:] <= indices[:-1]).any()):
                raise ModelInputContractError(
                    "valid dino_frame_idx values must be strictly increasing "
                    f"for sample {sample_idx}."
                )

        num_entities = spec.num_players + 1
        token_valid = frame_mask.unsqueeze(-1).expand(
            batch_size, seq_len, num_entities
        )
        entity_axis_valid = token_valid.reshape(
            batch_size * seq_len, num_entities
        )
        time_axis_valid = token_valid.permute(0, 2, 1).reshape(
            batch_size * num_entities, seq_len
        )
        entity_attn_mask, _ = build_self_attn_mask(entity_axis_valid)
        time_attn_mask, _ = build_self_attn_mask(time_axis_valid)

        encoded_tokens = spec.dino_encoded_num_tokens
        dino_key_valid = (
            dino_valid.unsqueeze(-1)
            .expand(batch_size, dino_samples, encoded_tokens)
            .reshape(batch_size, dino_samples * encoded_tokens)
        )
        dino_batch_has_evidence = dino_key_valid.any(dim=1)
        dino_key_valid = dino_key_valid.clone()
        dino_key_valid[:, 0] = dino_key_valid[:, 0] | ~dino_batch_has_evidence
        dino_attn_mask = dino_key_valid[:, None, :].expand(
            batch_size,
            seq_len * num_entities,
            dino_samples * encoded_tokens,
        )

        return ModelCall(
            kwargs={
                "player_kp": player_kp,
                "player_kp_vis": player_kp_vis,
                "player_valid": player_valid,
                "ball_uv": ball_uv,
                "ball_vis": ball_vis,
                "court_kp": court_kp,
                "court_vis": court_vis,
                "frame_mask": frame_mask,
                "entity_attn_mask": entity_attn_mask,
                "time_attn_mask": time_attn_mask,
                "dino_tokens": dino_tokens,
                "dino_frame_idx": dino_frame_idx,
                "dino_attn_mask": dino_attn_mask,
                "dino_batch_has_evidence": dino_batch_has_evidence,
            }
        )

    def build_training_targets(
        self, batch: Mapping[str, object]
    ) -> SLCSTrainingTargets:
        """Validate all training targets before the model is entered."""
        spec = self.spec
        player_kp = require_tensor(
            batch,
            "player_kp",
            spec=TensorSpec(
                shape=(None, spec.num_players, None, NUM_HUMAN_KP, 2),
                dtypes=_FLOAT32,
            ),
        )
        batch_size, players, seq_len = player_kp.shape[:3]
        if batch_size <= 0 or not 0 < seq_len <= spec.max_seq_len:
            raise ModelInputContractError(
                f"training targets require B>0 and 0<T<={spec.max_seq_len}, got "
                f"B={batch_size}, T={seq_len}."
            )
        frame_mask = require_tensor(
            batch,
            "frame_mask",
            spec=TensorSpec(shape=(batch_size, seq_len), dtypes=_BOOL),
        )
        target_player_position = require_tensor(
            batch,
            "target_player_position",
            spec=TensorSpec(
                shape=(batch_size, players, seq_len, 3), dtypes=_FLOAT32
            ),
        )
        target_player_rotation = require_tensor(
            batch,
            "target_player_rotation",
            spec=TensorSpec(
                shape=(batch_size, players, seq_len, 2), dtypes=_FLOAT32
            ),
        )
        target_player_valid = require_tensor(
            batch,
            "target_player_valid",
            spec=TensorSpec(shape=(batch_size, players, seq_len), dtypes=_BOOL),
        )
        target_player_weight = require_tensor(
            batch,
            "target_player_weight",
            spec=TensorSpec(shape=(batch_size, players, seq_len), dtypes=_FLOAT32),
        )
        target_ball_position = require_tensor(
            batch,
            "target_ball_position",
            spec=TensorSpec(shape=(batch_size, seq_len, 3), dtypes=_FLOAT32),
        )
        target_ball_valid = require_tensor(
            batch,
            "target_ball_valid",
            spec=TensorSpec(shape=(batch_size, seq_len), dtypes=_BOOL),
        )
        target_ball_weight = require_tensor(
            batch,
            "target_ball_weight",
            spec=TensorSpec(shape=(batch_size, seq_len), dtypes=_FLOAT32),
        )
        for name, value in (
            ("target_player_position", target_player_position),
            ("target_player_rotation", target_player_rotation),
            ("target_player_weight", target_player_weight),
            ("target_ball_position", target_ball_position),
            ("target_ball_weight", target_ball_weight),
        ):
            _finite(name, value)
        if bool(
            ((target_player_weight < 0.0) | (target_player_weight > 1.0)).any()
        ) or bool(((target_ball_weight < 0.0) | (target_ball_weight > 1.0)).any()):
            raise ModelInputContractError("SLCS target weights must lie in [0, 1].")
        if bool((target_player_valid & ~frame_mask.unsqueeze(1)).any()) or bool(
            (target_ball_valid & ~frame_mask).any()
        ):
            raise ModelInputContractError(
                "SLCS targets cannot mark a padded frame as label-valid."
            )
        if bool((target_player_weight[~target_player_valid] != 0.0).any()) or bool(
            (target_ball_weight[~target_ball_valid] != 0.0).any()
        ):
            raise ModelInputContractError(
                "SLCS target weights must be zero wherever the target is invalid."
            )
        valid_rotations = target_player_rotation[target_player_valid]
        if valid_rotations.numel() and bool(
            (torch.linalg.vector_norm(valid_rotations, dim=-1) <= 0.0).any()
        ):
            raise ModelInputContractError(
                "valid target_player_rotation vectors must have nonzero norm."
            )

        player_mask = target_player_valid & frame_mask.unsqueeze(1)
        ball_mask = target_ball_valid & frame_mask
        return SLCSTrainingTargets(
            target_player_position=target_player_position,
            target_player_rotation=target_player_rotation,
            target_ball_position=target_ball_position,
            player_mask=player_mask,
            player_weight=target_player_weight,
            ball_mask=ball_mask,
            ball_weight=target_ball_weight,
            frame_mask=frame_mask,
        )

    def decode_output(self, output: SLCSRawOutput) -> SLCSDecodedOutput:
        """Validate exact keys, shapes, dtypes, and finite model semantics."""
        if not isinstance(output, Mapping):
            raise ModelOutputContractError(
                f"SLCS model output must be a mapping, got {type(output).__name__}."
            )
        keys = frozenset(output)
        if keys != _OUTPUT_KEYS:
            raise ModelOutputContractError(
                "SLCS model output key mismatch: "
                f"missing={sorted(_OUTPUT_KEYS - keys)}, "
                f"unknown={sorted(keys - _OUTPUT_KEYS)}."
            )
        output_mapping = cast(Mapping[str, object], output)
        tensors: dict[str, Tensor] = {}
        for name in _OUTPUT_KEYS:
            value = output_mapping[name]
            if not isinstance(value, Tensor) or not torch.is_floating_point(value):
                raise ModelOutputContractError(
                    f"SLCS output {name!r} must be a floating tensor."
                )
            if not bool(torch.isfinite(value).all()):
                raise ModelOutputContractError(
                    f"SLCS output {name!r} contains non-finite values."
                )
            tensors[name] = value

        player_position = tensors["player_position"]
        if player_position.ndim != 4 or player_position.shape[1:] != (
            self.spec.num_players,
            player_position.shape[2],
            3,
        ):
            raise ModelOutputContractError(
                "player_position must have shape (B, P, T, 3)."
            )
        batch_size, players, seq_len = player_position.shape[:3]
        expected_shapes = {
            "player_rotation": (batch_size, players, seq_len, 2),
            "player_position_log_b": (batch_size, players, seq_len),
            "player_rotation_log_b": (batch_size, players, seq_len),
            "ball_position": (batch_size, seq_len, 3),
            "ball_position_log_b": (batch_size, seq_len),
        }
        for name, expected in expected_shapes.items():
            if tensors[name].shape != expected:
                raise ModelOutputContractError(
                    f"{name} must have shape {expected}, got "
                    f"{tuple(tensors[name].shape)}."
                )
        rotation_norm = torch.linalg.vector_norm(tensors["player_rotation"], dim=-1)
        if bool((rotation_norm <= 0.0).any()):
            raise ModelOutputContractError(
                "player_rotation vectors must have nonzero norm."
            )
        for name in (
            "player_position_log_b",
            "player_rotation_log_b",
            "ball_position_log_b",
        ):
            value = tensors[name]
            if bool(
                ((value < self.spec.log_b_min) | (value > self.spec.log_b_max)).any()
            ):
                raise ModelOutputContractError(
                    f"{name} must stay inside the configured log-b range."
                )

        return SLCSDecodedOutput(
            player_position=player_position,
            player_rotation=tensors["player_rotation"],
            player_position_log_b=tensors["player_position_log_b"],
            player_rotation_log_b=tensors["player_rotation_log_b"],
            ball_position=tensors["ball_position"],
            ball_position_log_b=tensors["ball_position_log_b"],
        )

    @staticmethod
    def to_physical(output: SLCSDecodedOutput) -> SLCSPhysicalOutput:
        """Decode normalized predictions into physical units."""
        scale = output.player_position.new_tensor(COURT_COORD_SCALE_XYZ)
        scale_mean = sum(COURT_COORD_SCALE_XYZ) / len(COURT_COORD_SCALE_XYZ)
        rotation = torch.nn.functional.normalize(output.player_rotation, dim=-1)
        return SLCSPhysicalOutput(
            player_position_meters=output.player_position * scale,
            player_yaw_radians=torch.atan2(rotation[..., 1], rotation[..., 0]),
            ball_position_meters=output.ball_position * scale,
            player_position_sigma_m=output.player_position_log_b.exp() * scale_mean,
            player_rotation_sigma_rad=output.player_rotation_log_b.exp(),
            ball_position_sigma_m=output.ball_position_log_b.exp() * scale_mean,
        )

    def validate_model(self, model: SLCSFusionModel) -> None:
        """Reject a same-class model with dimensions incompatible with this adapter."""
        expected = (
            self.spec.num_players,
            self.spec.num_court_kp,
            self.spec.max_seq_len,
            self.spec.dino_num_tokens,
            self.spec.dino_encoded_num_tokens,
            self.spec.dino_embed_dim,
        )
        actual = (
            model.num_players,
            model.num_court_kp,
            model.max_seq_len,
            model.dino_encoder.num_input_tokens,
            model.dino_encoder.num_tokens,
            model.dino_encoder.input_dim,
        )
        if actual != expected:
            from src.tasks.base.model_io import ModelAdapterMismatchError

            raise ModelAdapterMismatchError(
                f"SLCS adapter dimensions {expected} do not match model {actual}."
            )


def as_raw_output(output: Mapping[str, Tensor]) -> SLCSRawOutput:
    """Narrow a model mapping for callers that cannot preserve TypedDict types."""
    return cast(SLCSRawOutput, output)


__all__ = ["SLCSModelIOAdapter", "SLCSModelIOSpec", "as_raw_output"]
