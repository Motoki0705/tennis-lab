"""Small stage-by-stage projections for the person-model overview."""

from __future__ import annotations

from ..sources import Sources, fraction, number, require
from ..table_spec import Column, Row, Table
from . import body, detection, pose


def build(sources: Sources) -> tuple[Table, ...]:
    conditions, population = detection.load_conditions(sources)
    by_key = {c.key: c for c in conditions}
    pose_speed = pose.load_comparison(sources, pose.load_accuracy(sources))
    body_speed = body.load_current(sources)
    condition = Column("経路 / Backend", 120)
    ms_image = Column("同期 P50", 65, 2, "ms/画像")
    det_baseline = by_key["dino-swin-l/trt16"]
    det_candidate = by_key["rfdetr-l/trt16"]
    joint_baseline = by_key["dino-swin-l-vitpose-h/trt16"]
    joint_candidate = by_key["ecpose-l-o365/pt16"]
    require(
        all(
            case.exclusion is None
            for case in (det_baseline, det_candidate, joint_baseline, joint_candidate)
        ),
        "An overview condition is outside the paired comparison",
    )
    return (
        Table(
            "overview-detection",
            "代表条件 · 人物検出",
            f"画像 B1。{population}の部分疑似GT（SMPL）。速度は固定 96 画像。",
            (
                condition,
                ms_image,
                Column("Recall IoU .50", 73, 2, "%", 100),
                Column("NVML Δ", 58, 0, "MiB"),
            ),
            tuple(
                Row(
                    c.key,
                    (
                        c.label_with_backend,
                        c.speed("stage_p50_ms"),
                        c.accuracy("recall_50"),
                        c.speed("nvml_peak_delta_mib"),
                    ),
                    exclusion=c.exclusion,
                )
                for c in (det_baseline, det_candidate)
            ),
        ),
        Table(
            "overview-joint",
            "代表条件 · 検出＋pose",
            f"画像 B1。{population}の SMPL 由来疑似GT、欠測込み OKS。速度は固定 96 画像。",
            (condition, ms_image, Column("平均 OKS", 61, 5)),
            tuple(
                Row(
                    c.key,
                    (
                        c.label_with_backend,
                        c.speed("stage_p50_ms"),
                        c.accuracy("coverage_oks"),
                    ),
                    exclusion=c.exclusion,
                )
                for c in (joint_baseline, joint_candidate)
            ),
        ),
        Table(
            "overview-top-down",
            "代表条件 · 固定 BBox の pose",
            "速度 B1 は先頭 300 crop、B8 は固定 384 crop。同じ候補の精度は B8・SMPL 由来疑似GT。",
            (
                condition,
                Column("B1 P50", 58, 2, "ms/人"),
                Column("B8 P50", 58, 2, "ms/人"),
                Column("平均 OKS", 58, 5),
            ),
            tuple(
                Row(
                    model,
                    (
                        f"{pose.MODELS[model]} / TRT16",
                        number(pose_speed[model, "TRT16", "1"]["wall_p50_ms_person"]),
                        number(pose_speed[model, "TRT16", "8"]["wall_p50_ms_person"]),
                        fraction(pose_speed[model, "TRT16", "8"]["mean_oks"]),
                    ),
                )
                for model in ("vitpose-h", "rtmpose-x")
            ),
        ),
        Table(
            "overview-body",
            "代表条件 · 身体復元",
            f"系列 B1。{body.SCOPE} {body.CONTEXT} {body.QUALITATIVE}",
            (
                condition,
                Column("同期 P50", 65, 2, "ms/系列"),
                Column("NVML", 62, 1, "MiB"),
                Column("状態", 71),
            ),
            tuple(
                Row(c.key, (c.label, c.p50, c.memory, c.status), exclusion=c.exclusion)
                for c in body_speed
                if c.batch == 1
            ),
        ),
    )
