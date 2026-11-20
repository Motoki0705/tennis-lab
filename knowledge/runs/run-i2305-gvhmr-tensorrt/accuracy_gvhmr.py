"""Compare postprocessed GVHMR SMPL-X and COCO-17 in identical coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from benchmark import write_json
from convert import CONDITIONS, require_queue
from inputs import sha256_file
from locked_inputs import HERE, checked_segments
from model import load_network, smpl_assets
from torch import nn

from src.submodules.models import SmplCoco17Reconstructor
from src.submodules.vendor.gvhmr.endecoder import EnDecoder
from src.submodules.vendor.gvhmr.pipeline import DEMO_ENDECODER_STATS, Pipeline
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

HEADS = ("pred_x", "pred_cam", "static_conf_logits")
SPACES = ("incam", "global")
PARAMETERS = ("body_pose", "global_orient", "betas", "transl", "joints")


class SavedHeads(nn.Module):
    def forward(self, **_inputs):
        return self.values


def rotation_degrees(actual, reference):
    def matrices(samples):
        vectors = torch.from_numpy(samples).double()
        return axis_angle_to_matrix(vectors.reshape(*vectors.shape[:-1], -1, 3))

    relative = matrices(actual) @ matrices(reference).mT
    sine = (
        relative[..., [2, 0, 1], [1, 2, 0]] - relative[..., [1, 2, 0], [2, 0, 1]]
    ).norm(dim=-1) * 0.5
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine).rad2deg().numpy()


def distribution(values):
    q50, q95 = np.quantile(values, [0.5, 0.95])
    return {"median": float(q50), "p95": float(q95), "max": float(values.max())}


def errors(actual, reference, space):
    def fetch(name):
        key = f"{space}_{name}"
        return actual[key], reference[key]

    metrics = {
        f"{name}_deg": rotation_degrees(*fetch(name))
        for name in ("body_pose", "global_orient")
    }
    for name, key, scale in (
        ("betas", "betas_abs", 1),
        ("transl", "transl_component_abs_mm", 1000),
        ("transl", "transl_l2_mm", 1000),
        ("joints", "coco17_l2_mm", 1000),
    ):
        before, after = fetch(name)
        diff = (before.astype(np.float64) - after) * scale
        metrics[key] = (
            np.abs(diff)
            if key in {"betas_abs", "transl_component_abs_mm"}
            else np.linalg.norm(diff, axis=-1)
        )
    return metrics


def compare(actual, reference, rows):
    if any(actual[key].shape != reference[key].shape for key in actual):
        raise ValueError("Accuracy output shapes differ from the FP32 reference")
    wanted_frames = np.array([row["length"] for row in rows])
    wanted_mask = (
        np.arange(actual["mask_out"].shape[1])[None, :] < wanted_frames[:, None]
    )
    valid = actual["mask_out"] & reference["mask_out"] & wanted_mask
    result = {
        "mask_matches_input": bool(np.array_equal(actual["mask_out"], wanted_mask)),
        "length_matches_input": bool(
            np.array_equal(actual["length_out"], wanted_frames)
        ),
        "mask_matches_fp32": bool(
            np.array_equal(actual["mask_out"], reference["mask_out"])
        ),
        "length_matches_fp32": bool(
            np.array_equal(actual["length_out"], reference["length_out"])
        ),
        "mask_disagreeing_frames": int(
            np.count_nonzero(actual["mask_out"] != reference["mask_out"])
        ),
        "compared_frames": int(valid.sum()),
        "expected_frames": int(wanted_frames.sum()),
        "segments": {},
    }
    if not valid.any():
        raise ValueError("No valid GVHMR frame in common with FP32")
    for space in SPACES:
        values = errors(actual, reference, space)
        result[space] = {
            name: distribution(value[valid]) for name, value in values.items()
        }
        for index, row in enumerate(rows):
            segment = result["segments"].setdefault(
                row["id"], {"valid_frames": int(valid[index].sum())}
            )
            segment[space] = {
                name: distribution(value[index][valid[index]])
                if valid[index].any()
                else None
                for name, value in values.items()
            }
    return result


def read_outputs(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def checked_measurement(directory, condition, frames, rows):
    report = json.loads((directory / "measurement.json").read_text())
    if (
        report["tensor_lock_sha256"] != sha256_file(HERE / "tensor-lock.json")
        or report["condition"] != condition
        or report["length"] != frames
        or report["segments"] != [row["id"] for row in rows]
    ):
        raise ValueError(f"Measurement does not match the input identity: {directory}")
    return report


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    segments, checkpoint = checked_segments(args.inputs, args.checkpoints)
    checkpoint_root = args.checkpoints.resolve()
    lock = json.loads((HERE / "tensor-lock.json").read_text())
    for asset in lock["assets"].values():
        if sha256_file(checkpoint_root / asset["path"]) != asset["sha256"]:
            raise ValueError(f"Unexpected body asset digest: {asset['path']}")
    selected = [
        (row, arrays)
        for row, arrays in segments
        if row["split"] == "eval" and row["length"] == args.length
    ]
    rows = [row for row, _ in selected]
    measure = checked_measurement(args.output, args.condition, args.length, rows)
    prediction_file = args.output / "raw-outputs.npz"
    if sha256_file(prediction_file) != measure["raw_outputs_sha256"]:
        raise ValueError("Measured raw outputs failed digest verification")
    raw = read_outputs(prediction_file)
    nonfinite = {name: int(np.count_nonzero(~np.isfinite(raw[name]))) for name in HEADS}
    if any(nonfinite.values()):
        write_json(args.output / "accuracy.json", {"head_nonfinite": nonfinite})
        raise ValueError(f"Nonfinite GVHMR heads: {nonfinite}")

    assets = smpl_assets(checkpoint_root)
    body_dir = checkpoint_root / "body_models"
    saved = SavedHeads()
    pipeline = (
        Pipeline(
            saved,
            EnDecoder(body_dir, smpl_assets=assets, stats_name=DEMO_ENDECODER_STATS),
        )
        .eval()
        .cuda()
    )
    joints = SmplCoco17Reconstructor(body_dir, device="cuda", smpl_assets=assets)
    native = load_network(checkpoint).cuda().network if args.condition == "0" else None
    native_parity = {}
    decoded = {f"{space}_{name}": [] for space in SPACES for name in PARAMETERS}
    for index, (_, arrays) in enumerate(selected):
        batch = {
            name: torch.from_numpy(arrays[name]).cuda()
            for name in ("length", "obs", "f_imgseq")
        }
        batch.update(
            {
                name: torch.from_numpy(arrays[name][None]).cuda()
                for name in ("bbx_xys", "K_fullimg", "cam_angvel")
            }
        )
        saved.values = {
            name: torch.from_numpy(raw[name][index : index + 1]).cuda().float()
            for name in HEADS
        }
        static_cam = bool(arrays["static_cam"].item())
        actual = pipeline(batch, postproc=True, static_cam=static_cam)
        if native is not None:
            pipeline.denoiser3d = native
            expected = pipeline(batch, postproc=True, static_cam=static_cam)
            pipeline.denoiser3d = saved
        for space in SPACES:
            key = f"pred_smpl_params_{space}"
            params = {name: value[0] for name, value in actual[key].items()}
            if native is not None:
                for name, value in params.items():
                    delta = float((value - expected[key][name][0]).abs().max())
                    name_with_space = f"{space}_{name}"
                    native_parity[name_with_space] = max(
                        native_parity.get(name_with_space, 0), delta
                    )
                    torch.testing.assert_close(
                        value, expected[key][name][0], rtol=1e-4, atol=2e-4
                    )
            for name, value in params.items():
                decoded[f"{space}_{name}"].append(value.cpu().numpy())
            decoded[f"{space}_joints"].append(joints.reconstruct(params).numpy())
    actual = {key: np.stack(value) for key, value in decoded.items()}
    actual.update({key: raw[key] for key in ("mask_out", "length_out")})
    decoded_file = args.output / "decoded-outputs.npz"
    np.savez(decoded_file, **actual)
    if args.condition == "0":
        reference = actual
        reference_hash = sha256_file(decoded_file)
    else:
        baseline = checked_measurement(args.reference, "0", args.length, rows)
        if baseline["device_uuid"] != measure["device_uuid"]:
            raise ValueError("FP32 reference used a different GPU")
        certificate = json.loads((args.reference / "accuracy.json").read_text())
        reference_file = args.reference / "decoded-outputs.npz"
        reference_hash = sha256_file(reference_file)
        if certificate["decoded_outputs_sha256"] != reference_hash:
            raise ValueError("FP32 reference predictions failed digest verification")
        reference = read_outputs(reference_file)
    result = {
        **compare(actual, reference, rows),
        "head_nonfinite": nonfinite,
        "decoded_outputs_sha256": sha256_file(decoded_file),
        "fp32_reference_sha256": reference_hash,
        "native_wrapper_max_abs": native_parity or None,
    }
    write_json(args.output / "accuracy.json", result)
    if not all(
        result[key]
        for key in (
            "mask_matches_fp32",
            "length_matches_fp32",
            "mask_matches_input",
            "length_matches_input",
        )
    ):
        raise ValueError(
            "Measured sequence length or mask differs from the locked input and FP32"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--checkpoints", default=Path("ckpt"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.condition != "0" and args.reference is None:
        parser.error("Conditions other than 0 require a measured FP32 reference")
    require_queue()
    evaluate(args)


if __name__ == "__main__":
    main()
