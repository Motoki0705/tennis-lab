"""Every public Ball selector, including the transferred deep-query family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.tasks.ball_detection.checkpoints.coordinates import (
    restore_coordinate_model,
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.checkpoints.hybrid_training import (
    initialize_hybrid,
    restore_hybrid_model,
)
from src.tasks.ball_detection.checkpoints.initialization import initialize_full_model
from src.tasks.ball_detection.checkpoints.pretraining import pretraining_config
from src.tasks.ball_detection.checkpoints.provenance import data_identity
from src.tasks.ball_detection.model_io.mdd_heatmap import MDDHeatmapAdapter
from src.tasks.ball_detection.model_io.mdd_hybrid import MDDHybridAdapter
from src.tasks.ball_detection.model_io.mdd_temporal import (
    build_temporal_model,
    restore_temporal_model,
    temporal_forward,
)
from src.tasks.ball_detection.model_io.predict import (
    predict_coordinates,
    predict_dpt_heatmaps,
)
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDHeatmapHybridConfig,
    MDDHybridConfig,
    MDDPoseConfig,
    MDDPoseDetector,
    MDDPretrainConfig,
    MDDQueryDetector,
)
from src.tasks.ball_detection.training.coordinates.objective import coordinate_loss
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.tasks.ball_detection.training.hybrid.objective import hybrid_objective
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from src.tasks.ball_detection.training.temporal.configuration import load_model_config
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_detection.contract_inputs import (
    assert_window_batch,
    window_loader,
    write_model_config,
    write_windows,
)
from tests.integration.tasks.ball_detection.convnext_contracts import CONVNEXT
from tests.support.model_contracts.parameters import (
    OptimizerContract,
    ParameterContract,
    names_under,
)
from tests.support.model_contracts.registry import ModelFamily
from tests.support.tasks.ball_detection.temporal import prepare_fixture

CONFIG_DIR = Path("src/tasks/ball_detection/configs")
SELECTORS = {
    "conv_next_unet": "conv_next_unet",
    "mdd_pose": "mdd_pose",
    "mdd_query": "mdd_pose",
    "mdd_dpt_pretrain": "mdd_dpt_pretrain",
    "mdd_dpt_convnext_v2": "mdd_dpt_pretrain",
    "mdd_dpt_fasternet": "mdd_dpt_pretrain",
    "mdd_dpt_rgb_mdd": "mdd_dpt_pretrain",
    "mdd_hybrid": "mdd_hybrid",
    "mdd_dpt_heatmap": "mdd_dpt_heatmap",
    "mdd_heatmap_hybrid": "mdd_heatmap_hybrid",
    "mdd_temporal_query": "mdd_temporal_query",
}


@dataclass
class LoopInputs:
    config: DictConfig
    kind: str
    manifest: Path
    model_file: Path
    parent: Path | None = None
    initialization: dict[str, Any] | None = None


def _compose(
    root: Path, gan: bool, selector: str, overrides: tuple[str, ...] = ()
) -> DictConfig:
    assert not gan, "The MDD loops have no GAN selector"
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"
    ):
        config = compose(
            config_name="train",
            overrides=[f"model={selector}", f"paths.data_root={root}", *overrides],
        )
    config.model.stem_channels = [4, 4, 8, 8]
    config.model.mixed_channels = [8, 16]
    config.model.dim, config.model.heads, config.model.layers, config.model.dropout = (
        16,
        2,
        1,
        0.0,
    )
    if config.model.name != "mdd_pose":
        config.model.residual_blocks = [0, 1, 1, 1, 1, 1]
        config.model.decoder_channels, config.model.ffn_dim = 8, 64
    return config


def _query_parent(root: Path, config: Any, manifest: Path) -> Path:
    model = DeepMDDQueryDetector(MDDPretrainConfig(**asdict(config)))
    contract = model.mdd.input_contract()
    decoder = jpeg_decoder_contract("opencv")
    payload = dict(
        schema="mdd_query_posttraining.v1",
        stage="query_posttraining",
        model_config=asdict(config),
        state_dict=model.state_dict(),
        global_step=0,
        epoch=0,
        recipe=dict(
            model=asdict(config),
            input_contract=contract,
            image_decode=decoder,
            parent=dict(input_contract=contract, image_decode=decoder),
            manifest_sha256=dual_sha256(manifest),
        ),
    )
    path = root / "query-parent.pt"
    save_coordinate_checkpoint(payload, path)
    return path


def _build(config: DictConfig, *, kind: str) -> tuple[Any, LoopInputs]:
    root = Path(config.paths.data_root)
    model_file = write_model_config(config, root)
    if kind == "coordinate":
        pose_spec = MDDPoseConfig.load(model_file)
        model: Any = (
            MDDPoseDetector(pose_spec)
            if pose_spec.requires_pose
            else MDDQueryDetector(pose_spec)
        )
        manifest = write_windows(root, pose=pose_spec.requires_pose)
    elif kind == "pretrain":
        model = MDDDPTDetector(MDDPretrainConfig.load(model_file))
        manifest = write_windows(root, pose=False)
    elif kind == "deep_query":
        query_spec = MDDPretrainConfig.load(model_file)
        manifest = write_windows(root, pose=False)
        parent = _query_parent(root, query_spec, manifest)
        model, receipt = initialize_full_model(
            parent, expected_sha256=dual_sha256(parent), jpeg_decoder="opencv"
        )
        return model, LoopInputs(config, kind, manifest, model_file, parent, receipt)
    elif kind == "hybrid":
        hybrid_spec = MDDHybridConfig.load(model_file)
        manifest = write_windows(root, pose=False)
        parent = _query_parent(root, hybrid_spec, manifest)
        model, receipt = initialize_hybrid(
            parent,
            hybrid_spec,
            mode="query",
            data=data_identity(manifest),
            image_decode=jpeg_decoder_contract("opencv"),
        )
        return model, LoopInputs(config, kind, manifest, model_file, parent, receipt)
    elif kind == "temporal":
        architecture = {
            "mdd_dpt_heatmap": "dpt",
            "mdd_heatmap_hybrid": "hybrid",
            "mdd_temporal_query": "query",
        }[config.model.name]
        model = build_temporal_model(load_model_config(model_file, architecture))
        manifest = prepare_fixture(root / "temporal") / "train_T32.json"
    else:
        raise ValueError(kind)
    return model, LoopInputs(config, kind, manifest, model_file)


def _batch(config: DictConfig, kind: str) -> Any:
    root = Path(config.paths.data_root)
    manifest = (
        root / "temporal/prepared/train_T32.json"
        if kind == "temporal"
        else root / "windows.json"
    )
    return next(
        iter(
            window_loader(
                manifest,
                pose=config.model.name == "mdd_pose"
                and config.model.pose_pooling is not None,
            )
        )
    )


def _prediction(model: Any, batch: Any, kind: str) -> Any:
    if kind in {"coordinate", "deep_query"}:
        return predict_coordinates(model, batch, torch.device("cpu"))
    if kind == "pretrain":
        return predict_dpt_heatmaps(model, batch["rgb"])
    if kind == "hybrid":
        return model(batch["rgb"], batch["timestamps"])
    return temporal_forward(model, batch["rgb"], batch["timestamps"])


def _result(model: Any, batch: Any, kind: str) -> Any:
    output = _prediction(model, batch, kind)
    if kind in {"coordinate", "deep_query"} or isinstance(model, DeepMDDQueryDetector):
        loss = coordinate_loss(output, batch["uv"], batch["position_valid"])
    elif kind == "hybrid":
        loss, _ = hybrid_objective(
            output,
            batch,
            loss_config=dict(
                heatmap_weight=1.0,
                presence_weight=1.0,
                sigma_ratio=0.012,
                focal_gamma=2.0,
            ),
        )
    else:
        loss, _ = heatmap_objective(output, batch, sigma_ratio=0.012, gamma=2.0)
    return {"loss": loss, "outputs": output}


def parameter_contract(model: Any, *, kind: str, frozen_phase: bool = False) -> Any:
    all_names = frozenset(dict(model.named_parameters()))
    unused = (
        names_under(model, "decoder.fusion_blocks.3.skip_block")
        if kind in {"pretrain", "hybrid", "temporal"}
        and not isinstance(model, DeepMDDQueryDetector)
        else frozenset()
    )
    groups = {"encoder": names_under(model, "encoder")}
    if kind == "coordinate":
        groups.update(
            {
                "head": names_under(model, "head"),
                "attention": names_under(model, "blocks"),
            }
        )
        if model.config.requires_pose:
            groups["pose"] = names_under(model, "pose")
    elif isinstance(model, DeepMDDQueryDetector):
        groups.update(
            {
                key: names_under(model, key)
                for key in ("head", "projection", "spatial_position", "blocks")
            }
        )
    else:
        groups["decoder"] = names_under(model, "decoder") - unused
        head = (
            "heatmap_head"
            if kind == "hybrid" or isinstance(model.config, MDDHeatmapHybridConfig)
            else "head"
        )
        groups["head"] = names_under(model, head)
        if head == "heatmap_head":
            groups["query attention"] = names_under(
                model, "blocks", "projection", "spatial_position"
            )
        if kind == "hybrid":
            groups["presence"] = names_under(model, "presence_head")
    if "ball_query" in all_names:
        groups["ball query"] = frozenset({"ball_query"})
    frozen: frozenset[str] = frozenset()
    if frozen_phase:
        assert kind in {"deep_query", "hybrid"}
        frozen = names_under(model, "encoder")
        if kind == "hybrid":
            frozen |= names_under(
                model, "projection", "spatial_position", "blocks"
            ) | frozenset({"ball_query"})
    active = all_names - unused - frozen
    updates = {name: group & active for name, group in groups.items() if group & active}
    return ParameterContract(active, frozen, unused, optimizer_frozen=frozen), (
        OptimizerContract("generator", all_names, updates),
    )


def _assert_batch(batch: Any, kind: str) -> None:
    assert batch["rgb"].dtype == torch.uint8 and batch["rgb"].shape[:3] == (1, 32, 3)
    assert batch["position_valid"].dtype == batch["heatmap_valid"].dtype == torch.bool
    assert not (batch["position_valid"] & ~batch["heatmap_valid"]).any()
    if kind != "temporal":
        assert_window_batch(batch)


def _assert_prediction(output: Any) -> None:
    tensors = (
        (output.heatmap_logits, output.presence_logits)
        if hasattr(output, "heatmap_logits")
        else (output,)
    )
    for tensor in tensors:
        assert (
            tensor.shape[:2] == (1, 32)
            and tensor.dtype == torch.float32
            and torch.isfinite(tensor).all()
        )


def _restore(config: DictConfig, path: Path, kind: str) -> Any:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if kind in {"coordinate", "deep_query"}:
        return restore_coordinate_model(saved)[0]
    if kind == "hybrid":
        model, recipe = restore_hybrid_model(saved)
        model._contract_presence_threshold = recipe["presence_threshold"]
        return model
    if kind == "temporal":
        return restore_temporal_model(saved["model_artifact"])
    model = MDDDPTDetector(pretraining_config(saved))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model


def _fit(*args: Any, **kwargs: Any) -> Any:
    from tests.integration.tasks.ball_detection.contract_loops import fit_loop

    return fit_loop(*args, **kwargs)


def _save(source: Path, target: Path) -> None:
    target.write_bytes(source.read_bytes())


def _public_prediction(model: Any, batch: Any, kind: str) -> Any:
    raw = _prediction(model, batch, kind)
    if kind == "hybrid":
        threshold = model._contract_presence_threshold
        decoded = MDDHybridAdapter(
            model.config, presence_threshold=threshold
        ).decode_output(raw)
        assert decoded.present.dtype == torch.bool
        assert torch.isnan(decoded.coords[~decoded.present]).all()
        assert not decoded.candidates.valid[~decoded.present].any()
        return dict(
            raw=raw,
            decoded=decoded,
            presence_threshold=threshold,
            position_valid=batch["position_valid"],
            heatmap_valid=batch["heatmap_valid"],
        )
    if kind == "temporal" and not isinstance(model, DeepMDDQueryDetector):
        return dict(raw=raw, decoded=MDDHeatmapAdapter(model.config).decode_output(raw))
    return raw


def family(
    selector: str,
    kind: str,
    *,
    overrides: tuple[str, ...] = (),
    name: str | None = None,
) -> ModelFamily:
    return ModelFamily(
        name=name or selector,
        compose=lambda root, gan: _compose(root, gan, selector, overrides),
        build=lambda config: _build(config, kind=kind),
        batch=lambda config: _batch(config, kind),
        supervised_result=lambda model, batch: _result(model, batch, kind),
        predict=lambda model, batch: _public_prediction(model, batch, kind),
        save=_save,
        restore=lambda config, path: _restore(config, path, kind),
        parameters=lambda model: parameter_contract(model, kind=kind),
        assert_batch=lambda batch: _assert_batch(batch, kind),
        assert_prediction=_assert_prediction,
        fit=_fit,
    )


FAMILIES = {
    "conv_next_unet": CONVNEXT,
    **{
        selector: family(selector, kind)
        for selector, kind in (
            ("mdd_pose", "coordinate"),
            ("mdd_query", "coordinate"),
            ("mdd_dpt_pretrain", "pretrain"),
            ("mdd_dpt_convnext_v2", "pretrain"),
            ("mdd_dpt_fasternet", "pretrain"),
            ("mdd_dpt_rgb_mdd", "pretrain"),
            ("mdd_hybrid", "hybrid"),
            ("mdd_dpt_heatmap", "temporal"),
            ("mdd_heatmap_hybrid", "temporal"),
            ("mdd_temporal_query", "temporal"),
        )
    },
}
DEEP_QUERY = family(
    "mdd_dpt_rgb_mdd", "deep_query", name="deep_query/freeze_then_joint"
)
# One case per remaining pose pooling path; this is not the 4x4x2 ablation grid.
POSE_VARIANTS = tuple(
    family(
        "mdd_pose",
        "coordinate",
        name=f"pose/{pool}/{readout}",
        overrides=(
            f"model.pose_pooling={pool}",
            f"model.readout={readout}",
            f"model.compression={compression}",
        ),
    )
    for pool, readout, compression in (
        ("deepsets", "pose", "average"),
        ("hierarchical", "query", "unshuffle"),
        ("gnn", "pose", "haar"),
    )
)
