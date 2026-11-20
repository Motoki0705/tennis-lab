"""Model and accuracy adapters for the shared Phase 1 benchmark."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from benchmark import PROFILE_ADDRESS, accuracy, normalize, verify_inputs
from clipreid import BATCH_SHAPE, FEATURE_DIMENSION, VisualFeatures
from convert import linear_names, modelopt_ptq
from fixed_crops import SHAPES, verify
from inputs import CHECKPOINT, CHECKPOINT_SHA256, MODEL, sha256_file
from torch import nn

from src.submodules.smpl_assets import require_smpl_asset
from src.submodules.vendor.gvhmr.hmr2 import load_hmr2
from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
from src.submodules.vendor.gvhmr.vitpose.flip_utils import flip_heatmap_coco17
from src.submodules.vendor.gvhmr.vitpose.heatmap_head import ViTPoseHeadConfig
from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps

VITPOSE = "vitpose_h"
HMR2 = "hmr2"
MODEL_ASSETS = {
    VITPOSE: (
        Path("ckpt/vitpose/vitpose-h-multi-coco.pth"),
        "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
    ),
    HMR2: (
        Path("ckpt/hmr2/epoch=10-step=25000.ckpt"),
        "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e",
    ),
    MODEL: (CHECKPOINT, CHECKPOINT_SHA256),
}
MEAN_PARAMS = Path("ckpt/body_models/gvhmr/smpl_mean_params.npz")
MEAN_PARAMS_SHA256 = "6fd6dd687800da946d0a0492383f973b92ec20f166a0b829775882868c35fcdd"
PIPELINE = Path("src/tennis_scene/configs/pipeline.yaml")
BLOCKS = {
    MODEL: re.compile(r"^/visual/transformer/resblocks/resblocks\.(\d+)/"),
    VITPOSE: re.compile(r"^/backbone/blocks\.(\d+)/"),
    HMR2: re.compile(r"^/hmr/backbone/blocks\.(\d+)/"),
}
ATTENTION_OUT = {
    MODEL: "/attn/Gemm",
    VITPOSE: "/attn/proj/MatMul",
    HMR2: "/attn/proj/MatMul",
}
COMPUTE_OPS = {"MatMul", "Gemm", "Conv", "ConvTranspose"}
RECIPES = {
    "attention-out-max": ("attention-out", "max"),
    "linear-max": ("linear", "max"),
    "linear-entropy": ("linear", "entropy"),
    "matmul-max": ("matmul", "max"),
}


def ptq_operations(model, graph, scope):
    pattern = BLOCKS[model]
    linear = linear_names(graph)
    selected = {}
    for node in graph.node:
        block = pattern.match(node.name)
        if block is None or node.op_type not in {"MatMul", "Gemm"}:
            continue
        if scope != "matmul" and node.name not in linear:
            continue
        if scope == "attention-out" and not node.name.endswith(ATTENTION_OUT[model]):
            continue
        selected[node.name] = int(block[1])
    block_count = 12 if model == MODEL else 32
    per_block = {"attention-out": 1, "linear": 4, "matmul": 6}[scope]
    if Counter(selected.values()) != {block: per_block for block in range(block_count)}:
        raise ValueError(
            f"{model} {scope} PTQ selects {len(selected)} operations; "
            f"expected {per_block} operations in each of {block_count} blocks"
        )
    return set(selected)


def head_settings():
    settings = yaml.safe_load(PIPELINE.read_text())["people_models"]["runtime"][
        "vitpose"
    ]
    if settings["flip_test"] is not True:
        raise ValueError("ViTPose benchmark requires the standard flip-enabled model")
    return settings["head"]


class Hmr2Images(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        require_smpl_asset(MEAN_PARAMS, "hmr2_mean_params")
        self.hmr = load_hmr2(str(checkpoint), mean_params_path=MEAN_PARAMS)
        # The vendor SMPL head creates a constant FP32 zero token. Match the
        # feature context when the whole model is converted to FP16 for export.
        self.hmr.smpl_head.transformer.register_forward_pre_hook(
            self.align_token, with_kwargs=True
        )

    @staticmethod
    def align_token(_module, args, kwargs):
        return (args[0].to(kwargs["context"].dtype), *args[1:]), kwargs

    def forward(self, pixels):
        return self.hmr({"img": pixels})


class FlipRunner:
    def __init__(self, runner):
        self.runner = runner
        self.pixels = torch.empty_like(runner.pixels[:1])

    def run(self):
        torch.cat((self.pixels, self.pixels.flip(3)), dim=0, out=self.runner.pixels)
        heatmaps, flipped = self.runner.run().chunk(2)
        return (heatmaps + flip_heatmap_coco17(flipped)) * 0.5


def decode_pose(heatmaps, boxes):
    scale = boxes[:, 2:] * np.asarray((24 / 32, 1), dtype=np.float32) / 200
    xy, confidence = keypoints_from_heatmaps(
        heatmaps=heatmaps.copy(), center=boxes[:, :2], scale=scale, use_udp=True
    )
    if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
        raise ValueError("UDP returned nonfinite ViTPose outputs")
    valid = confidence[..., 0] > 0
    return xy, confidence[..., 0], valid


def distribution(values, *, tail=0.95):
    if not len(values):
        return None
    return {
        "median": float(np.median(values)),
        f"p{int(tail * 100):02d}": float(np.quantile(values, tail)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def pose_accuracy(actual, baseline, boxes):
    xy, score, mask = decode_pose(actual, boxes)
    ref_xy, ref_score, ref_mask = decode_pose(baseline, boxes)
    common = mask & ref_mask
    distance = np.linalg.norm(xy - ref_xy, axis=-1)
    score_diff = np.abs(score - ref_score)
    return {
        "crops": len(xy),
        "joints": int(mask.size),
        "invalid_fp32_joints": int(np.count_nonzero(~ref_mask)),
        "invalid_condition_joints": int(np.count_nonzero(~mask)),
        "mask_mismatches": int(np.count_nonzero(mask != ref_mask)),
        "common_valid_joints": int(common.sum()),
        "xy_l2_px": distribution(distance[common]),
        "confidence_abs": distribution(score_diff[common]),
    }, np.where(common, distance, -1).max(axis=1)


def feature_accuracy(actual, baseline):
    diff = actual.astype(np.float64) - baseline
    cosine = np.clip(np.sum(normalize(actual) * normalize(baseline), axis=-1), -1, 1)
    relative_l2 = np.linalg.norm(diff, axis=-1) / np.linalg.norm(
        baseline.astype(np.float64), axis=-1
    )
    abs_error = np.abs(diff)
    return {
        "crops": len(actual),
        "cosine": distribution(cosine, tail=0.05),
        "relative_l2": distribution(relative_l2),
        "abs_error": distribution(abs_error.ravel()),
        "max_abs_error_by_dimension": abs_error.max(axis=0).tolist(),
    }, relative_l2


def compare_crops(model, actual, baseline, addresses, inputs):
    if (
        actual.shape != baseline.shape
        or not len(actual) == len(addresses)
        or not np.isfinite(actual).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("Model outputs are not aligned to the fixed crop addresses")
    if model == VITPOSE:
        boxes = np.load(inputs / "eval-boxes.npy", allow_pickle=False)
        result, by_crop = pose_accuracy(actual, baseline, boxes)
    else:
        result, by_crop = feature_accuracy(actual, baseline)
        result["missing_fp32_crops"] = result["missing_condition_crops"] = 0
    grouped = {}
    for i, address in enumerate(addresses):
        grouped.setdefault("/".join(address[:2]), []).append(i)
    result[
        "crop_max_xy_l2_px_by_camera" if model == VITPOSE else "relative_l2_by_camera"
    ] = {
        key: distribution(
            by_crop[np.asarray(indices)][by_crop[np.asarray(indices)] >= 0]
        )
        for key, indices in grouped.items()
    }
    worst = np.argsort(-by_crop, kind="stable")[:5]
    result["largest_error_crop_addresses"] = [
        list(addresses[i]) for i in worst if by_crop[i] >= 0
    ]
    return result


def external_onnx(model, sample, path):
    """Export a graph first, then stream parameters into ONNX external data."""
    import onnx

    torch.onnx.export(
        model,
        (sample,),
        str(path),
        input_names=["pixels"],
        output_names=["features"],
        opset_version=19,
        dynamo=False,
        export_params=False,
        do_constant_folding=False,
    )
    proto = onnx.load(path)
    parameters = model.state_dict()
    inputs = list(proto.graph.input)
    weights = {info.name: parameters[info.name] for info in inputs[1:]}
    if inputs[0].name != "pixels" or not weights:
        raise ValueError(
            "Graph-only export did not preserve the input and named parameters"
        )
    # Fold only weight transposes, as required by Q/DQ per-output-channel scales.
    usage = {}
    for node in proto.graph.node:
        for name in node.input:
            usage[name] = usage.get(name, 0) + 1
    for node in list(proto.graph.node):
        if node.op_type != "Transpose" or node.input[0] not in weights:
            continue
        name = node.input[0]
        permutation = next(attr.ints for attr in node.attribute if attr.name == "perm")
        weights[node.output[0]] = weights[name].permute(*permutation)
        if usage[name] == 1:
            del weights[name]
        proto.graph.node.remove(node)

    del proto.graph.input[1:]
    location = path.name + ".data"
    with (path.parent / location).open("xb") as stream:
        for name, tensor in weights.items():
            value = tensor.detach().contiguous().numpy()
            offset = stream.tell()
            stream.write(memoryview(value).cast("B"))
            initializer = proto.graph.initializer.add()
            initializer.name = name
            initializer.data_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            initializer.dims.extend(value.shape)
            initializer.raw_data = b""
            onnx.external_data_helper.set_external_data(
                initializer, location=location, offset=offset, length=value.nbytes
            )
            initializer.ClearField("raw_data")
    onnx.save_model(proto, path)
    onnx.checker.check_model(str(path))


class VitAdapter:
    def __init__(self, model):
        self.model = model
        _, self.checkpoint_sha256 = MODEL_ASSETS[model]
        self.batch_shape = (1, *SHAPES[model]) if model != MODEL else BATCH_SHAPE
        self.engine_shape = (
            (2, *self.batch_shape[1:]) if model == VITPOSE else self.batch_shape
        )
        self.output_shape = {
            VITPOSE: (17, 64, 48),
            HMR2: (1024,),
            MODEL: (FEATURE_DIMENSION,),
        }[model]
        self.engine_output_shape = (self.engine_shape[0], *self.output_shape)
        self.autocast = model in {HMR2, VITPOSE}

    def identity(self):
        if self.model == HMR2:
            if sha256_file(MEAN_PARAMS) != MEAN_PARAMS_SHA256:
                raise ValueError("HMR2 mean parameters differ from the fixed asset")
            return {"mean_params_sha256": MEAN_PARAMS_SHA256}
        if self.model == VITPOSE:
            return {"vitpose_head": head_settings(), "flip_test": True}
        return {}

    def verify_inputs(self, directory):
        self.inputs = directory
        if self.model == MODEL:
            lock, addresses = verify_inputs(directory)
            return {**lock, "profile_address": list(PROFILE_ADDRESS)}, addresses
        return verify(self.model, directory)

    def accuracy(self, actual, baseline, addresses):
        if self.model == MODEL:
            return accuracy(actual, baseline, addresses)
        return compare_crops(self.model, actual, baseline, addresses, self.inputs)

    def torch_model(self, checkpoint):
        if self.model == HMR2:
            return Hmr2Images(checkpoint)
        if self.model == VITPOSE:
            config = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in head_settings().items()
            }
            return build_vitpose_huge(
                checkpoint, head_config=ViTPoseHeadConfig(**config)
            )
        return VisualFeatures(checkpoint)

    def wrap_runner(self, runner):
        return FlipRunner(runner) if self.model == VITPOSE else runner

    def measurement_settings(self, condition):
        settings = {
            "engine_batch_shape": self.engine_shape,
            "flip_test": self.model == VITPOSE,
        }
        if condition == "1" and self.autocast:
            settings["torch_autocast"] = {
                "dtype": "float16",
                "parameter_dtype": "float32",
            }
        return settings

    def export(self, checkpoint, path, *, fp16):
        import onnx

        torch.set_num_threads(4)
        dtype = torch.float16 if fp16 else torch.float32
        model = self.torch_model(checkpoint).to(dtype=dtype).eval()
        sample = torch.zeros(self.engine_shape, dtype=dtype)
        with torch.no_grad():
            if self.model != MODEL and not fp16:
                external_onnx(model, sample, path)
            else:
                torch.onnx.export(
                    model,
                    (sample,),
                    str(path),
                    input_names=["pixels"],
                    output_names=["features"],
                    opset_version=19,
                    dynamo=False,
                    external_data=False,
                )
                onnx.checker.check_model(str(path))

    def quantize(self, graph, target, data, mode, recipe, *, calibration_shapes=None):
        import onnx

        if recipe not in RECIPES:
            raise ValueError(f"Select a PTQ recipe explicitly: {sorted(RECIPES)}")
        scope, method = RECIPES[recipe]
        if mode == "fp8" and method != "max":
            raise ValueError("Use max calibration for FP8")
        model = onnx.load(graph, load_external_data=False)
        selected = ptq_operations(self.model, model.graph, scope)
        # Static ViTPose graph batch 2 has adjacent original and mirrored views.
        calibration = (
            data
            if self.model != VITPOSE
            else np.stack((data, data[..., ::-1]), axis=1).reshape(-1, *data.shape[1:])
        )
        report = modelopt_ptq(
            graph,
            target,
            calibration,
            mode,
            input_name="pixels",
            nodes=sorted(selected),
            method=method,
            calibration_shapes=calibration_shapes,
            restore_dynamic_shapes=self.model == MODEL
            and calibration_shapes is not None,
            external_data=self.model != MODEL,
            disable_mha_qdq=self.model != MODEL and scope != "matmul",
        )
        report.update(
            recipe=recipe,
            representative_crops=len(data),
        )
        batch_field = (
            "engine_batch" if calibration_shapes is None else "calibration_batch"
        )
        report[batch_field] = self.engine_shape[0]
        return report, selected

    def compute_requirements(self, graph, mode, selected):
        import onnx

        exported = onnx.load(graph, load_external_data=False).graph
        if [inp.name for inp in exported.input] != ["pixels"]:
            raise ValueError("Export must have one input named pixels")
        dynamic = {"pixels"}
        operations = {}
        constants = []
        for node in exported.node:
            depends_on_pixels = not dynamic.isdisjoint(node.input)
            if depends_on_pixels:
                dynamic.update(node.output)
            if node.op_type in COMPUTE_OPS:
                operations[node.name] = mode if node.name in selected else "fp16"
                if not depends_on_pixels:
                    constants.append(node.name)
        if not operations.keys() >= selected or len(operations) == len(constants):
            raise ValueError("Export did not preserve the declared compute operations")
        if set(constants) & selected:
            raise ValueError("PTQ selection contains an input-independent operation")
        return {"expected_operations": operations, "constant_operations": constants}


def adapter(model):
    return VitAdapter(model)
