"""Public cohort and measurement-protocol summaries."""

from __future__ import annotations

from decimal import Decimal

from ..sources import Sources, array, count, obj, require, string, unique
from ..table_spec import Column, Row, Table

RUN = "run-meiji-multiview-player-evaluation/"
COHORT = RUN + "publication.json"
QUALITY = RUN + "quality_counts.csv"
TIMING = "run-person-model-speed-baseline/timing-protocol.json"
CROP_SPEED = "run-person-model-speed-low-precision/cases.csv"


def build(sources: Sources) -> tuple[Table, ...]:
    publication = sources.json(COHORT)
    groups = array(publication["clips"])
    require(len(groups) == 6, "Unexpected Meiji comparison cohort")
    raw = unique(sources.csv(QUALITY), "clip_id", "camera_id", "player_id")
    keys = ("box_eligible", "box_excluded", "joint_eligible", "joint_excluded")
    rows = []
    clips = set()
    for metadata in groups:
        clip = obj(metadata)
        clip_id = string(clip["clip_id"])
        cameras, players = array(clip["camera_ids"]), array(clip["player_ids"])
        addresses = {
            (clip_id, string(camera), str(player))
            for camera in cameras
            for player in players
        }
        clips.update(addresses)
        totals = tuple(
            Decimal(sum(count(raw[address][key]) for address in addresses))
            for key in keys
        )
        rows.append(Row(clip_id, (clip_id.split("/")[-1].replace("_", " "), *totals)))
    require(set(raw) == clips, "Unexpected Meiji cohort quality-count addresses")
    frames, images = (
        count(publication["evaluation_frames"]),
        count(publication["evaluation_images"]),
    )
    quality = Table(
        "evaluation-cohort",
        "疑似GT の採点対象と除外",
        f"{frames:,} 時刻・{images:,} 画像。各クリップの全カメラ・選手の件数を集計。"
        " 固定マスクの除外理由は正本を参照。参照のない人物のラベル状態は不明。",
        (
            Column("クリップ", 59),
            Column("採点 BBox", 54, 0),
            Column("除外 BBox", 54, 0),
            Column("採点関節", 56, 0),
            Column("除外関節", 56, 0),
        ),
        tuple(rows),
    )
    timing = sources.json(TIMING)
    schedules = obj(timing["schedules"])
    warmup, blocks, reps = (
        count(timing[key]) for key in ("warmup", "blocks", "iterations_per_block")
    )
    b1 = unique(sources.csv(CROP_SPEED), "workload", "batch_size", "backend")[
        "vitpose-flip", "1", "TRT16"
    ]
    names = {
        "frames": ("検出 / 検出＋pose", "ms/画像"),
        "crops": ("Top-down pose", "ms/人"),
        "body_sequences": ("身体復元", "ms/系列"),
    }
    units = []
    for key, (label, unit) in names.items():
        schedule = obj(schedules[key])
        size = count(schedule["records"])
        batches = " / ".join(
            f"B{count(batch)}" for batch in obj(schedule["batch_sizes"])
        )
        note = (
            f"B1 は先頭 {count(b1['distinct_inputs'])} crop"
            if key == "crops"
            else "固定順"
        )
        units.append(Row(key, (label, Decimal(size), batches, unit, note)))
    protocol = Table(
        "timed-inputs",
        "速度計測の入力と単位",
        f"RTX 5060 Ti・ウォームアップ {warmup} 回・{reps} 回 × {blocks} ブロック。"
        " 同期 wall は前処理から返却後のデバイス同期まで。GPU event と NVML は別指標。",
        (
            Column("workload", 84),
            Column("固定入力", 52, 0),
            Column("Batch", 63),
            Column("単位", 48),
            Column("備考", 89),
        ),
        tuple(units),
    )
    return quality, protocol
