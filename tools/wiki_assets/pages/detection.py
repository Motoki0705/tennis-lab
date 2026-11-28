"""Meiji player detection and joint detection/pose projections."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..chart_spec import Scatter
from ..sources import (
    CsvRow,
    Sources,
    array,
    count,
    fraction,
    mib,
    number,
    obj,
    require,
    string,
    unique,
)
from ..table_spec import Column, Row, Table

RUN = "run-meiji-player-model-comparison/"
CONDITIONS = RUN + "conditions.csv"
EVIDENCE = RUN + "comparison-evidence.json"
TIMING = RUN + "joint-speed/cases.csv"
INITIAL = "run-person-model-speed-baseline/cases.csv"
MODELS = {
    "dino-swin-l": ("DINO Swin-L", "DINO"),
    "dino-swin-l-vitpose-h": ("DINO + ViTPose-H", "DINO + ViTPose-H"),
    "rfdetr-l": ("RF-DETR-L", "RF-DETR"),
    "rfdetr-2xl": ("RF-DETR-2XL", "RF-DETR"),
    "ecdet-l-coco": ("ECDet-L COCO", "ECDet · COCO"),
    "ecdet-x-coco": ("ECDet-X COCO", "ECDet · COCO"),
    "ecdet-l-o365": ("ECDet-L O365", "ECDet · O365"),
    "ecdet-x-o365": ("ECDet-X O365", "ECDet · O365"),
    "rtdetrv4-l": ("RT-DETRv4-L", "RT-DETRv4"),
    "rtdetrv4-x": ("RT-DETRv4-X", "RT-DETRv4"),
    "ecpose-l-o365": ("ECPose-L O365", "ECPose · O365"),
    "ecpose-x-o365": ("ECPose-X O365", "ECPose · O365"),
}
BACKENDS = {
    "pt32": "PT32",
    "pt16": "PT16",
    "trt16": "TRT16",
    "trt16-fp32attn": "TRT16+FP32 attn",
}
STATUS = {
    ("measured", "ok"): ("同条件で計測", "変換異常・診断"),
    ("measured", "unmatched_host"): ("旧2スレッド計時", "変換異常・旧2スレッド"),
    ("excluded", "historical"): ("範囲外・旧計時", "範囲外・旧計時"),
    ("measured", "excluded"): ("精度のみ・速度対象外", "精度のみ・速度対象外"),
    ("excluded", "excluded"): ("範囲外", "範囲外"),
    ("infeasible", "infeasible"): ("変換異常・実行不能", "変換異常・実行不能"),
    ("not_applicable", "not_applicable"): ("変換方式を適用不可", "変換方式を適用不可"),
}
QUALITY = ("recall_50", "recall_75", "small_recall_50", "recall_50_score_ge_0_5")
POSE_QUALITY = (
    "coverage_oks",
    "coverage_pck_0_10",
    "joint_coverage",
    "small_recall_50",
)
SPEED = ("stage_p50_ms", "stage_p95_ms", "images_per_s", "nvml_peak_delta_mib")
GPU = "RTX 5060 Ti"


@dataclass(frozen=True)
class Condition:
    source: CsvRow
    label: str
    backend: str
    group: str
    status: str
    exclusion: str | None

    @property
    def key(self) -> str:
        return f"{self.source['model_id']}/{self.source['backend']}"

    @property
    def label_with_backend(self) -> str:
        return f"{self.label} / {self.backend}"

    def accuracy(self, field: str) -> Decimal | None:
        value = self.source[field]
        if self.source["accuracy_status"] == "measured":
            return fraction(value)
        require(value == "", f"{self.key}: unmeasured accuracy has a value")
        return None

    def speed(self, field: str) -> Decimal | None:
        value = self.source[field]
        if self.source["speed_status"] == "ok":
            return number(value)
        require(value == "", f"{self.key}: unmatched speed has a value")
        return None


def _condition(row: CsvRow, diagnostic: bool) -> Condition:
    accuracy, speed = row["accuracy_status"], row["speed_status"]
    model = row["model_id"]
    label, group = MODELS[model]
    backend = BACKENDS[row["backend"]]
    require(
        row["workload"] in {"detection", "detection_and_pose"},
        "Unknown detector workload",
    )
    if accuracy == "measured":
        require(
            row["accuracy_batch_size"] == "1", f"{model}: accuracy must use image B1"
        )
    if speed == "ok":
        require(
            (
                accuracy,
                row["speed_batch_size"],
                row["host_threads"],
                row["opencv_threads"],
            )
            == ("measured", "1", "4", "1"),
            f"{model}: comparison requires matched B1 and CPU threads",
        )
    status = STATUS[accuracy, speed][int(diagnostic)]
    if model == "rfdetr-2xl":
        require(
            (accuracy, speed) == ("excluded", "excluded")
            and "license decision" in row["accuracy_reason"],
            "RF-DETR-2XL license decision differs",
        )
        status = "ライセンス除外"
    comparable = (accuracy, speed) == ("measured", "ok") and not diagnostic
    return Condition(row, label, backend, group, status, None if comparable else status)


def load_conditions(sources: Sources) -> tuple[list[Condition], str]:
    records = unique(sources.csv(CONDITIONS), "model_id", "backend", "workload")
    evidence = sources.json(EVIDENCE)
    index = {}
    for item in array(evidence["conditions"]):
        entry = obj(item)
        key = (string(entry["model_id"]), string(entry["backend"]))
        require(key not in index, "Duplicate detector evidence condition")
        index[key] = entry
    require(
        set(index) <= {key[:2] for key in records},
        "Detector evidence refers to a missing condition",
    )
    evaluation = obj(evidence["evaluation"])
    population = f"6クリップ・{count(evaluation['images']):,} 画像・{count(evaluation['eligible_boxes']):,} 人物枠"
    conditions = []
    for (model, backend, _), row in records.items():
        # Diagnostic reasons are cross-checked, but only the public status vocabulary is rendered.
        entry = index.get((model, backend), {})
        diagnosis = row["pareto_exclusion_reason"]
        require(
            entry.get("pareto_exclusion_reason", "") == diagnosis,
            f"{model}: diagnosis differs from evidence",
        )
        conditions.append(_condition(row, bool(diagnosis)))
    return conditions, population


def _pair_tables(
    conditions: list[Condition], *, joint: bool, population: str
) -> tuple[Table, Table, Scatter]:
    selected = "detection_and_pose" if joint else "detection"
    cohort = [
        condition
        for condition in conditions
        if condition.source["workload"] == selected
    ]
    slug = "joint-b1" if joint else "detection-b1"
    name = "人物検出＋pose" if joint else "人物検出"
    quality_note = (
        f"画像 B1・{population}の部分疑似GT（SMPL 由来）。未ラベル人物の precision / AP は未定義。"
        + (
            " OKS / PCK は欠測込み。"
            if joint
            else " 通常の Recall は各方式の推論閾値。共通閾値は score ≥ 0.50。"
        )
    )
    speed_note = (
        f"画像 B1・固定 96 画像・{GPU}・CPU 4 / OpenCV 1 thread。同期 wall。"
        " NVML はロード前からの増分。旧計時と変換診断は順位対象外。"
    )
    heading = Column("モデル / Backend", 111)
    status = Column("状態", 77)
    recall = Column("R50", 42, 2, "%", 100, "max")
    oks = Column("平均 OKS", 46, 5, best="max")
    small = Column("小 R50", 47, 2, "%", 100, "max")
    metrics = (
        (
            oks,
            Column("PCK .10", 48, 2, "%", 100, "max"),
            Column("関節有効", 48, 2, "%", 100),
            small,
        )
        if joint
        else (
            recall,
            Column("R75", 42, 2, "%", 100, "max"),
            small,
            Column("共通 R50", 50, 2, "%", 100),
        )
    )
    latency = Column("P50", 48, 2, "ms/画像", best="min")
    accuracy = Table(
        f"{slug}-quality",
        f"{name}の精度",
        quality_note,
        (heading, *metrics, status),
        tuple(
            Row(
                c.key,
                (
                    c.label_with_backend,
                    *(c.accuracy(k) for k in (POSE_QUALITY if joint else QUALITY)),
                    c.status,
                ),
                c.group,
                c.exclusion,
            )
            for c in cohort
        ),
    )
    timed = Table(
        f"{slug}-speed",
        f"{name}の速度と VRAM",
        speed_note,
        (
            heading,
            latency,
            Column("P95", 48, 2, "ms/画像", best="min"),
            Column("速度", 45, 2, "画像/s"),
            Column("NVML Δ", 50, 0, "MiB"),
            status,
        ),
        tuple(
            Row(
                c.key,
                (c.label_with_backend, *(c.speed(k) for k in SPEED), c.status),
                c.group,
                c.exclusion,
            )
            for c in cohort
        ),
    )
    axis = oks if joint else Column("Recall (IoU 0.50)", 57, 2, "%", 100)
    chart = Scatter(
        Table(
            f"{slug}-pareto",
            f"{name}の精度と遅延",
            f"画像 B1・疑似GT（SMPL）・{population}\n"
            f"RTX 5060 Ti・固定 96 画像の同期 wall（CPU 4 / OpenCV 1）。× は診断。\n"
            "対応する速度がない条件はテキスト表に記載。",
            (Column("モデル", 100), Column("Backend", 60), latency, axis, status),
            tuple(
                Row(
                    c.key,
                    (
                        c.label,
                        c.backend,
                        c.speed("stage_p50_ms"),
                        c.accuracy("coverage_oks" if joint else "recall_50"),
                        c.status,
                    ),
                    exclusion=c.exclusion,
                )
                for c in cohort
            ),
        ),
        x=2,
        y=3,
        zoom=None if joint else ((49, 60), (97.3, 98.5)),
    )
    return accuracy, timed, chart


def _joint_batch2(sources: Sources, conditions: list[Condition]) -> Table:
    index = unique(sources.csv(TIMING), "workload", "batch_size", "backend")
    require(
        set(index)
        == {
            ("dino-vitpose", batch, backend)
            for batch in ("1", "2")
            for backend in ("PT16", "TRT16")
        },
        "Unexpected DINO + ViTPose-H timing conditions",
    )
    baseline = {
        c.backend: c
        for c in conditions
        if c.source["model_id"] == "dino-swin-l-vitpose-h"
    }
    rows = []
    for (_, batch, backend), raw in index.items():
        require(
            (
                raw["item_unit"],
                raw["status"],
                raw["samples"],
                raw["distinct_inputs"],
                raw["timed_clips"],
            )
            == ("image", "ok", "300", "96", "6"),
            "Expected fixed 96-image synchronized timing",
        )
        p50, p95 = number(raw["wall_p50_ms"]), number(raw["wall_p95_ms"])
        memory = mib(raw["nvml_peak_minus_before_load_bytes"])
        if batch == "1":
            require(
                (p50, p95, memory)
                == tuple(baseline[backend].speed(k) for k in (*SPEED[:2], SPEED[-1])),
                "B1 values differ between the comparison and joint-speed receipts",
            )
            continue
        status = "速度のみ・精度未計測"
        rows.append(
            Row(backend, (backend, p50 / 2, p95 / 2, memory, status), exclusion=status)
        )
    return Table(
        "joint-b2-speed",
        "DINO + ViTPose-H の画像 B2 参考計時",
        "固定 96 画像・同期 wall をバッチ内の 2 画像で割る。NVML は増分。画像 B2 の精度は未計測。",
        (
            Column("Backend", 56),
            Column("P50", 55, 2, "ms/画像"),
            Column("P95", 55, 2, "ms/画像"),
            Column("NVML Δ", 56, 0, "MiB"),
            Column("状態", 83),
        ),
        tuple(rows),
    )


def _old_pt32(sources: Sources) -> Table:
    records = unique(sources.csv(INITIAL), "workload", "backend", "batch_size")
    status = "同期境界の修正前"
    rows = []
    for workload, label in (
        ("dino", "DINO Swin-L"),
        ("dino-vitpose", "DINO + ViTPose-H"),
    ):
        for batch in (1, 2):
            case = records[workload, "PT32", str(batch)]
            require(
                (case["status"], case["item_unit"], case["distinct_inputs"])
                == ("ok", "image", "96"),
                "Unexpected legacy image timing condition",
            )
            rows.append(
                Row(
                    f"{workload}/PT32/{batch}",
                    (
                        f"画像 B{batch}",
                        number(case["wall_p50_ms"]) / batch,
                        number(case["wall_p95_ms"]) / batch,
                        status,
                    ),
                    label,
                    status,
                )
            )
    return Table(
        "detection-pt32-historical",
        "人物検出と検出＋pose の旧 PT32 計時",
        "固定 96 画像。バッチ wall を画像数で割った参考値。同期境界の修正後の値とは比較対象を分ける。",
        (
            Column("Batch", 52),
            Column("P50", 57, 2, "ms/画像"),
            Column("P95", 57, 2, "ms/画像"),
            Column("状態", 92),
        ),
        tuple(rows),
    )


def build(sources: Sources) -> tuple[Table | Scatter, ...]:
    conditions, population = load_conditions(sources)
    return (
        *_pair_tables(conditions, joint=False, population=population),
        *_pair_tables(conditions, joint=True, population=population),
        _joint_batch2(sources, conditions),
        _old_pt32(sources),
    )
