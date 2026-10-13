"""Compare four completed evaluations only after their conditions match."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

BACKBONES = {
    "dinov3_vits16": "ViT-S/16",
    "dinov3_vits16plus": "ViT-S+/16",
    "dinov3_vitb16": "ViT-B/16",
    "dinov3_vitl16": "ViT-L/16",
}


def comparison_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "court_ablation_evaluation_v1":
        raise ValueError("Unsupported evaluation schema")
    config = report["config"]
    encoder = config["model"]["encoder"]
    if encoder["train_mode"] != "frozen" or encoder["lora"]["enabled"]:
        raise ValueError("The comparison requires fully frozen DINO backbones")
    return {
        "data": config["data"],
        "mixed": config["mixed"],
        "training": config["training"],
        "seed": config["run"]["seed"],
        "loss": config["loss"],
        "downstream": {
            key: value for key, value in config["model"].items() if key != "encoder"
        },
        "target_bundle": report["target_bundle"],
        "data_manifest_sha256": report["data_manifest_sha256"],
        "precision": report["precision"],
        "evaluation_compile": report["evaluation_compile"],
        "qualitative_batch_size": report["qualitative_batch_size"],
        "downstream_parameters": {
            key: value
            for key, value in report["parameters"].items()
            if key not in {"backbone", "projection"}
        },
        "splits": {
            name: {key: value for key, value in split.items() if key != "metrics"}
            for name, split in report["splits"].items()
        },
    }


def compare_evaluations(inputs: list[Path], output: Path) -> Path:
    loaded = {}
    for path in inputs:
        report = json.loads(path.read_text())
        backbone = report["config"]["model"]["encoder"]["backbone_name"]
        if backbone not in BACKBONES or backbone in loaded:
            raise ValueError(
                "Provide exactly one evaluation for each S/S+/B/L backbone"
            )
        loaded[backbone] = (path.parent, report)
    if set(loaded) != set(BACKBONES):
        raise ValueError("Provide all four S/S+/B/L evaluations")
    contract = comparison_contract(loaded[next(iter(BACKBONES))][1])
    for _, report in loaded.values():
        if comparison_contract(report) != contract:
            raise ValueError(
                "Evaluation data, ordering, preprocessing, targets or downstream conditions differ"
            )
    output.mkdir(parents=True, exist_ok=False)
    text = [
        "# DINOv3サイズ比較：後続容量共通",
        "",
        "Transformer・DPT・headの構成と評価条件を照合済み。backboneは完全凍結。",
        "",
    ]
    columns = [
        ("kp_mean_distance_px", "KP px ↓"),
        ("seg_miou", "SEG mIoU ↑"),
        ("line_dice", "LINE Dice ↑"),
        ("semantic_line_miou", "semantic LINE mIoU ↑"),
    ]
    for split_name, split in contract["splits"].items():
        has_pose = split["pose_supervised"]
        selected_columns = columns + (
            [
                ("pose_reprojection_mean_distance_px", "pose再投影 px ↓"),
                ("pose_translation_l2_m", "位置 m ↓"),
                ("pose_rotation_geodesic_deg", "回転 deg ↓"),
                ("pose_focal_relative_error", "焦点相対誤差 ↓"),
            ]
            if has_pose
            else []
        )
        text += [
            f"## {split_name} ({split['count']} images)",
            "",
            "| Model | " + " | ".join(label for _, label in selected_columns) + " |",
            "|---|" + "---:|" * len(selected_columns),
        ]
        for backbone, label in BACKBONES.items():
            metrics = loaded[backbone][1]["splits"][split_name]["metrics"]
            text.append(
                "| "
                + label
                + " | "
                + " | ".join(f"{metrics[name]:.6g}" for name, _ in selected_columns)
                + " |"
            )
        text += [
            "",
            "実画像のposeは教師がないためN/A。"
            if not has_pose
            else "poseは合成V3の教師に対して評価。",
            "",
        ]
        for index, sample in enumerate(split["qualitative"]):
            text += [f"### {sample['sample_id']}", ""]
            for head in sample["files"]:
                images = []
                for backbone in BACKBONES:
                    directory, report = loaded[backbone]
                    filename = report["splits"][split_name]["qualitative"][index][
                        "files"
                    ][head]
                    with Image.open(directory / split_name / filename) as image:
                        images.append(image.convert("RGB"))
                if len({image.size for image in images}) != 1:
                    raise ValueError("Qualitative image dimensions differ")
                width, height = images[0].size
                canvas = Image.new("RGB", (width * 4, height + 28), "white")
                draw = ImageDraw.Draw(canvas)
                for column, (label, image) in enumerate(
                    zip(BACKBONES.values(), images, strict=True)
                ):
                    draw.text((column * width + 5, 5), label, fill="black")
                    canvas.paste(image, (column * width, 28))
                name = f"{split_name}-{index:02d}-{head}.png"
                canvas.save(output / name)
                text += [f"![{head}: {sample['sample_id']}]({name})", ""]
    text += [
        "## パラメータ数",
        "",
        "| Model | 凍結backbone | 学習射影 | 共通後続 |",
        "|---|---:|---:|---:|",
    ]
    for backbone, label in BACKBONES.items():
        counts = loaded[backbone][1]["parameters"]
        downstream = sum(
            value["trainable"]
            for key, value in counts.items()
            if key not in {"backbone", "projection"}
        )
        text.append(
            f"| {label} | {counts['backbone']['total']} | {counts['projection']['trainable']} | {downstream} |"
        )
    text += [
        "",
        "KPは可視点全体の距離、mIoUは背景を含む全クラスの累積intersection/union、LINEは画像ごとのcoverage Diceの平均。",
        "定性画像は固定IDをbatch 1で推論し、定量評価は記録した共通batch sizeを使用。",
        "",
    ]
    result = output / "comparison.md"
    result.write_text("\n".join(text))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "checkpoints": {
                    backbone: {
                        key: report[key]
                        for key in (
                            "checkpoint",
                            "checkpoint_sha256",
                            "epoch",
                            "global_step",
                            "parameters",
                        )
                    }
                    for backbone, (_, report) in loaded.items()
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    return result
