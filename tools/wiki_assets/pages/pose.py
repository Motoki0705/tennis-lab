"""Fixed-box top-down pose projections, including the unscored attempts."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from ..chart_spec import Bars, Scatter, pareto_frontier
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
    unique,
)
from ..table_spec import Column, Row, Table, Value

RUN = "run-meiji-topdown-pose-comparison/"
ACCURACY = RUN + "accuracy.csv"
COMPARISON = RUN + "comparison.csv"
DIFFICULTY = RUN + "difficulty_pt32.csv"
ATTEMPTS = RUN + "sapiens2_attempts.json"
SPEED = "run-person-model-speed-low-precision/cases.csv"
BATCHES = ("1", "4", "8")
BACKENDS = ("PT16", "TRT16")
MODELS = {
    "vitpose-h": "ViTPose-H（flip あり）",
    "rtmpose-m": "RTMPose-M",
    "rtmpose-x": "RTMPose-X",
    "pmpose-l": "PMPose-L",
    "pmpose-h": "PMPose-H",
    "vitpose-h-no-flip": "ViTPose-H（flip なし）",
    "sapiens2-1b": "Sapiens2-1B",
    "sapiens2-5b": "Sapiens2-5B",
}
MAIN_MODELS = tuple(MODELS)[:5]
DIFFICULTY_GROUPS = {
    "all": "全条件",
    "small_projected_person": "小人物",
    "overlapping_boxes": "BBox 重なり",
    "image_edge": "画像端",
    "likely_inter_player_occlusion": "選手間遮蔽の疑い",
    "incomplete_reference": "参照関節不足",
    "fast_wrist": "手首の高速移動",
}
ACCURACY_VALUES = ("oks", "pck_0_10", "pck_0_20", "p95_px", "missing_joints")
POPULATION = "6 クリップ・26,276 人物枠・437,714 関節"
PSEUDO = "固定 BBox・SMPL 由来の疑似GT。参照生成に ViTPose / GVHMR を使用。"


def load_accuracy(sources: Sources) -> dict[tuple[str, ...], CsvRow]:
    rows = unique(sources.csv(ACCURACY), "model", "backend")
    require(
        set(rows)
        == {(model, backend) for model in MODELS for backend in ("PT32", *BACKENDS)},
        "Unexpected pose accuracy conditions",
    )
    for (model, _), raw in rows.items():
        if model in MAIN_MODELS or model == "vitpose-h-no-flip":
            scope = "primary" if model in MAIN_MODELS else "supplemental"
            require(
                (raw["comparison"], raw["status"], raw["rows"], raw["eligible_joints"])
                == (scope, "ok", "26276", "437714"),
                f"{model}: mismatched pose accuracy cohort",
            )
    return rows


def load_comparison(
    sources: Sources, accuracy: dict[tuple[str, ...], CsvRow]
) -> dict[tuple[str, ...], CsvRow]:
    rows = unique(sources.csv(COMPARISON), "model", "backend", "speed_batch_size")
    require(
        set(rows)
        == {
            (model, backend, batch)
            for model in MAIN_MODELS
            for backend in BACKENDS
            for batch in BATCHES
        },
        "Unexpected top-down speed comparison conditions",
    )
    for (model, backend, _), raw in rows.items():
        reference = accuracy[model, backend]
        require(
            (raw["speed_status"], raw["accuracy_batch_size"], raw["parity"])
            == ("ok", "8", "PT32未計測"),
            f"{model}: unexpected speed or accuracy condition",
        )
        for left, right in (
            ("accuracy_rows", "rows"),
            ("eligible_joints", "eligible_joints"),
            ("mean_oks", "oks"),
            *((field, field) for field in ACCURACY_VALUES[1:]),
        ):
            require(
                number(raw[left]) == number(reference[right]),
                f"{model}: {left} differs from accuracy.csv",
            )
    return rows


def _accuracy_tables(
    accuracy: dict[tuple[str, ...], CsvRow],
) -> tuple[Table, Table, Table]:
    ranked: list[Row] = []
    reference: list[Row] = []
    parity: list[Row] = []
    for (model, backend), raw in accuracy.items():
        if model.startswith("sapiens2-"):
            continue
        label = f"{MODELS[model]} / {backend}"
        key = f"{model}/{backend}"
        values: tuple[Value, ...] = (
            label,
            *(fraction(raw[k]) for k in ACCURACY_VALUES[:3]),
            number(raw["p95_px"]),
            Decimal(count(raw["missing_joints"])),
        )
        if model in MAIN_MODELS and backend in BACKENDS:
            ranked.append(Row(key, values))
        else:
            group = "保存済み PT32" if model in MAIN_MODELS else "flip なし（補助条件）"
            reference.append(Row(key, values, group, group))
        if backend in BACKENDS:
            parity.append(
                Row(
                    key,
                    (
                        label,
                        number(raw["vs_pt32_p95_px"]),
                        number(raw["vs_pt32_max_px"]),
                    ),
                    "flip なし（補助条件）"
                    if model == "vitpose-h-no-flip"
                    else "主比較",
                )
            )
    columns = (
        Column("モデル / Backend", 115),
        Column("OKS", 43, 5, best="max"),
        Column("PCK .10", 48, 2, "%", 100, "max"),
        Column("PCK .20", 48, 2, "%", 100, "max"),
        Column("誤差 P95", 55, 2, "px", best="min"),
        Column("欠測関節", 51, 0),
    )
    note = f"精度 B8・{POPULATION}。{PSEUDO} OKS / PCK は欠測込み。"
    return (
        Table("pose-b8-quality", "Top-down pose の精度", note, columns, tuple(ranked)),
        Table(
            "pose-b8-reference",
            "Top-down pose の参考精度",
            note + " 主比較の順位対象外。",
            columns,
            tuple(reference),
        ),
        Table(
            "pose-pt32-difference",
            "保存済み PT32 との座標差",
            f"精度 B8・{POPULATION}。同じモデルの PT32 との保存出力差。同期速度計測とは別の記録。",
            (
                columns[0],
                Column("座標差 P95", 68, 2, "px"),
                Column("座標差 最大", 68, 2, "px"),
            ),
            tuple(parity),
        ),
    )


def _speed_tables(comparison: dict[tuple[str, ...], CsvRow]) -> tuple[Table, ...]:
    tables = []
    for batch in BATCHES:
        rows = []
        for (model, backend, speed_batch), raw in comparison.items():
            if speed_batch != batch:
                continue
            rows.append(
                Row(
                    f"{model}/{backend}/{batch}",
                    (
                        f"{MODELS[model]} / {backend}",
                        number(raw["wall_p50_ms_person"]),
                        number(raw["wall_p95_ms_person"]),
                        number(raw["persons_per_s"]),
                        number(raw["nvml_peak_mib"]),
                    ),
                )
            )
        crops = 300 if batch == "1" else 384
        tables.append(
            Table(
                f"pose-b{batch}-speed",
                f"Top-down pose の速度と VRAM · B{batch}",
                f"固定 {crops} crop・RTX 5060 Ti・同期 wall、速度 B{batch}。精度は別表の B8。"
                " NVML はデバイス全体のサンプル最大値。速度計測の PT32 差は未計測。",
                (
                    Column("モデル / Backend", 115),
                    Column("P50", 52, 2, "ms/人", best="min"),
                    Column("P95", 52, 2, "ms/人", best="min"),
                    Column("速度", 52, 2, "人/s"),
                    Column("NVML", 61, 0, "MiB", best="min"),
                ),
                tuple(rows),
            )
        )
    return tuple(tables)


def _pareto(comparison: dict[tuple[str, ...], CsvRow]) -> tuple[Scatter, ...]:
    charts = []
    for batch in BATCHES:
        rows = []
        saved = set()
        for (model, backend, speed_batch), raw in comparison.items():
            if speed_batch != batch:
                continue
            key = f"{model}/{backend}/{batch}"
            require(raw["pareto"] in {"True", "False"}, "Unexpected pose Pareto flag")
            on_frontier = raw["pareto"] == "True"
            if on_frontier:
                saved.add(key)
            rows.append(
                Row(
                    key,
                    (
                        MODELS[model],
                        backend,
                        number(raw["wall_p50_ms_person"]),
                        fraction(raw["mean_oks"]),
                        number(raw["nvml_peak_mib"]),
                        "非支配" if on_frontier else "支配あり",
                    ),
                )
            )
        crops = 300 if batch == "1" else 384
        chart = Scatter(
            Table(
                f"pose-b{batch}-pareto",
                f"Top-down pose の精度と遅延 · 速度 B{batch}",
                f"速度 B{batch}・固定 {crops} crop・RTX 5060 Ti の同期 wall。\n"
                f"精度 B8・{POPULATION}の SMPL 由来疑似GT。\n"
                "外枠は OKS・P50・NVML（全体）の 3 指標で非支配。",
                (
                    Column("モデル", 105),
                    Column("Backend", 54),
                    Column("P50", 54, 2, "ms/人"),
                    Column("平均 OKS", 49, 5),
                    Column("NVML", 60, 0, "MiB"),
                    Column("3 指標 Pareto", 65),
                ),
                tuple(rows),
            ),
            x=2,
            y=3,
            memory=4,
        )
        require(
            {row.key for row in pareto_frontier(chart)} == saved,
            f"Pose B{batch} Pareto differs from comparison.csv",
        )
        charts.append(chart)
    return tuple(charts)


def _viTPose_speed(
    sources: Sources, accuracy: dict[tuple[str, ...], CsvRow]
) -> tuple[Bars, Table]:
    raw = unique(sources.csv(SPEED), "workload", "batch_size", "backend")
    heading = Column("Batch", 43)
    backend_column = Column("Backend", 56)
    latency = Column("P50", 54, 2, "ms/人")
    panels = []
    for workload in ("vitpose-flip", "vitpose"):
        rows = []
        for batch in BATCHES:
            for backend in BACKENDS:
                case = raw[workload, batch, backend]
                n = "300" if batch == "1" else "384"
                require(
                    (
                        case["status"],
                        case["item_unit"],
                        case["samples"],
                        case["distinct_inputs"],
                    )
                    == ("ok", "person-crop", "300", n),
                    "Unexpected ViTPose fixed-crop timing condition",
                )
                p50 = number(case["wall_p50_ms"]) / int(batch)
                cells: tuple[Value, ...]
                if workload == "vitpose-flip":
                    cells = (f"B{batch}", backend, p50)
                else:
                    cells = (
                        f"B{batch}",
                        backend,
                        p50,
                        number(case["wall_p95_ms"]) / int(batch),
                        mib(case["nvml_device_peak_sampled_bytes"]),
                        fraction(accuracy["vitpose-h-no-flip", backend]["oks"]),
                    )
                rows.append(Row(f"{backend}/{batch}", cells))
        panels.append(tuple(rows))
    note = "RTX 5060 Ti・同期バッチ wall を crop 数で割る。B1 は先頭 300 crop、B4 / B8 は固定 384 crop。"
    chart = Bars(
        Table(
            "vitpose-flip-backend",
            "ViTPose-H（flip あり）の backend 別速度",
            note.replace("。B1", "。\nB1"),
            (heading, backend_column, latency),
            panels[0],
        ),
        metrics=(2,),
        series=1,
    )
    supplemental = Table(
        "vitpose-no-flip",
        "ViTPose-H（flip なし）の補助条件",
        note + " 精度 B8 は固定 BBox（疑似GT）。flip なしは主比較の順位対象外。",
        (
            heading,
            backend_column,
            latency,
            Column("P95", 54, 2, "ms/人"),
            Column("NVML", 57, 0, "MiB"),
            Column("平均 OKS", 54, 5),
        ),
        panels[1],
    )
    return chart, supplemental


def _sapiens(sources: Sources, accuracy: dict[tuple[str, ...], CsvRow]) -> Table:
    evidence = sources.json(ATTEMPTS)
    scope = obj(evidence["scope_decision"])
    models = ("sapiens2-1b", "sapiens2-5b")
    backends = ("PT32", "PT16", "TRT16")
    require(
        tuple(array(scope["models"])) == models
        and tuple(array(scope["backends"])) == backends
        and scope["status"] == "excluded"
        and scope["reason"] == "GPU time",
        "Unexpected Sapiens2 scope",
    )
    attempts = obj(evidence["attempts"])
    require(
        array(obj(attempts["1b-pt32-accuracy"])["published_accuracy_artifacts"]) == [],
        "Sapiens2-1B PT32 has published accuracy",
    )
    load = obj(attempts["5b-pt32-load"])
    require(
        load["queue_state"] == "done" and bool(obj(load["load_receipt"])),
        "Sapiens2-5B load is not confirmed",
    )
    rows = []
    for model, size in zip(models, ("1b", "5b"), strict=True):
        conversion = obj(attempts[f"{size}-trt16-build"])
        require(
            conversion["queue_state"] == "failed"
            and conversion["operator"] == "ReduceMean",
            "Sapiens2 TRT16 conversion outcome differs",
        )
        for backend in backends:
            expected = "cancelled"
            label = "打ち切り・未計測" if backend == "PT32" else "取消・未計測"
            if backend == "PT32" and size == "5b":
                expected, label = "load_only", "重みロードのみ"
            else:
                attempt = obj(attempts[f"{size}-{backend.lower()}-accuracy"])
                require(
                    attempt["queue_state"] == "cancelled",
                    "Sapiens2 accuracy cancellation differs",
                )
            if backend == "TRT16":
                expected, label = "conversion_failed", "変換停止・計測取消"
            raw = accuracy[model, backend]
            require(
                raw["comparison"] == "excluded"
                and raw["status"] == expected
                and not any(
                    raw[k] for k in ("rows", "eligible_joints", *ACCURACY_VALUES)
                ),
                "Sapiens2 has unexpected measured accuracy or scope",
            )
            rows.append(
                Row(
                    f"{model}/{backend}",
                    (MODELS[model], backend, None, None, label),
                    exclusion=label,
                )
            )
    return Table(
        "sapiens2-status",
        "Sapiens2 の試行と採点状況",
        "GPU 時間を理由に比較範囲から除外。TRT16 は ReduceMean の変換で停止。5B PT32 はロードの完了を確認。",
        (
            Column("モデル", 79),
            Column("Backend", 52),
            Column("平均 OKS", 50, 5),
            Column("P50", 52, 2, "ms/人"),
            Column("状態", 93),
        ),
        tuple(rows),
    )


def _difficulty(sources: Sources) -> Table:
    raw = unique(sources.csv(DIFFICULTY), "model", "group", "joint")
    joints = {key[2] for key in raw}
    require(
        len(joints) == 17
        and set(raw)
        == {
            (model, group, joint)
            for model in (*MAIN_MODELS, "vitpose-h-no-flip")
            for group in DIFFICULTY_GROUPS
            for joint in joints
        },
        "Unexpected pose difficulty matrix",
    )
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for (model, group, _), values in raw.items():
        eligible = count(values["eligible_joints"])
        missing = count(values["missing_joints"])
        exact_hits = fraction(values["pck_0_10"]) * eligible
        hits = int(exact_hits.to_integral_value())
        require(
            abs(exact_hits - hits) <= Decimal("1e-8") and hits + missing <= eligible,
            "Difficulty PCK does not match the eligible/missing counts",
        )
        totals = counts[model, group]
        totals[0] += eligible
        totals[1] += missing
        totals[2] += hits
    rows = []
    for group, name in DIFFICULTY_GROUPS.items():
        for model in (*MAIN_MODELS, "vitpose-h-no-flip"):
            eligible, missing, hits = counts[model, group]
            rows.append(
                Row(
                    f"{model}/{group}",
                    (
                        MODELS[model],
                        Decimal(eligible),
                        Decimal(missing),
                        Decimal(hits) / eligible if eligible else None,
                    ),
                    name,
                    "保存済み PT32 診断",
                )
            )
    return Table(
        "pose-pt32-difficulty",
        "難条件別の PT32 診断",
        "固定 BBox・SMPL 由来疑似GT。各群で 17 関節の PCK .10 適合数を合計し、欠測込みの有効参照数で割る。"
        " 群は重複可。手首の高速移動は参照速度の上位 10% を使用。",
        (
            Column("モデル", 112),
            Column("有効参照", 58, 0),
            Column("欠測", 53, 0),
            Column("PCK .10", 62, 2, "%", 100),
        ),
        tuple(rows),
    )


def build(sources: Sources) -> tuple[Table | Scatter | Bars, ...]:
    accuracy = load_accuracy(sources)
    comparison = load_comparison(sources, accuracy)
    return (
        *_accuracy_tables(accuracy),
        *_speed_tables(comparison),
        *_pareto(comparison),
        *_viTPose_speed(sources, accuracy),
        _difficulty(sources),
        _sapiens(sources, accuracy),
    )
