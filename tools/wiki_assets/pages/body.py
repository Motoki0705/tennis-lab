"""Body-recovery timing and memory. The benchmark supplies no accuracy metric."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..chart_spec import Bars
from ..sources import Sources, array, count, mib, number, obj, require
from ..table_spec import Column, Row, Table

CURRENT = "run-person-model-speed-low-precision/cases.csv"
INITIAL = "run-person-model-speed-baseline/cases.csv"
FAST = "run-fasthmr-camerahmr-pt16-speed/speed-summary.json"
EXPORT = "run-fasthmr-camerahmr-trt-export-probe/trt-export.json"
SMOKE = "run-fasthmr-camerahmr-smoke/observations.json"
BATCHES = (1, 2, 4)
CURRENT_PATH = "HMR2 + GVHMR"
FAST_PATH = "FastHMR CameraHMR"
SCOPE = "固定 24 系列（各 16 フレーム）・RTX 5060 Ti・同期 wall。NVML はデバイス全体。"
CONTEXT = (
    "FastHMR は CPU 2・特徴 B4、現行は CPU 4・特徴 B16。計時経路が異なるため参考比較。"
)
QUALITATIVE = "身体復元は定性評価のみ。目視評価は未実施。"
REFERENCE = "参考・経路差"
WALL = ("wall_p50_ms", "wall_p95_ms")


@dataclass(frozen=True)
class BodyCase:
    key: str
    label: str
    batch: int
    p50: Decimal | None
    p95: Decimal | None
    memory: Decimal
    status: str
    exclusion: str | None = None


def _csv_cases(
    sources: Sources, source: str, backends: tuple[str, ...]
) -> list[BodyCase]:
    cases = {}
    old = source == INITIAL
    for row in sources.csv(source):
        if row["workload"] != "body":
            continue
        backend, batch = row["backend"], count(row["batch_size"])
        key = batch, backend
        require(key not in cases, "Duplicate body speed condition")
        require(
            (
                row["item_unit"],
                row["samples"],
                row["distinct_inputs"],
                row["timed_clips"],
            )
            == ("person-frame", "300", "24", "6"),
            "Unexpected body speed workload",
        )
        status = row["status"]
        times: tuple[Decimal | None, ...]
        if status == "ok":
            times = tuple(number(row[k]) / batch for k in WALL)
        else:
            require(
                old
                and backend == "TRT16"
                and status == "parity_failed"
                and not any(row[k] for k in WALL),
                "Unexpected body timing outcome",
            )
            times = (None, None)
        label = f"{CURRENT_PATH} {backend}"
        note = (
            ("旧計時" if status == "ok" else "初回の出力差基準不成立")
            if old
            else "PT32 差未判定"
        )
        cases[key] = BodyCase(
            f"gvhmr/{backend}/{batch}",
            label,
            batch,
            times[0],
            times[1],
            mib(row["nvml_device_peak_sampled_bytes"]),
            note,
            note if old else None,
        )
    require(
        set(cases) == {(batch, backend) for batch in BATCHES for backend in backends},
        "Missing body speed condition",
    )
    return [cases[batch, backend] for batch in BATCHES for backend in backends]


def load_current(sources: Sources) -> list[BodyCase]:
    current = _csv_cases(sources, CURRENT, ("PT16", "TRT16"))
    gvhmr = {case.batch: case for case in current if case.key.startswith("gvhmr/PT16/")}
    fast = sources.json(FAST)
    require(count(fast["fixed_body_sequences"]) == 24, "Unexpected FastHMR input count")
    seen = set()
    for item in array(fast["cases"]):
        raw = obj(item)
        case = obj(raw["case"])
        baseline = obj(raw["gvhmr_pt16"])
        baseline_case = obj(baseline["case"])
        settings, baseline_settings = (
            obj(case["settings"]),
            obj(baseline_case["settings"]),
        )
        batch = count(case["batch_size"])
        require(batch in gvhmr and batch not in seen, "Unexpected FastHMR series batch")
        seen.add(batch)
        require(
            (raw["status"], case["name"], case["backend"], case["item_unit"])
            == ("ok", "fasthmr-camerahmr", "PT16", "person-frame"),
            "Unexpected FastHMR timing condition",
        )
        require(
            (
                baseline_case["name"],
                baseline_case["backend"],
                count(baseline_case["batch_size"]),
            )
            == ("body", "PT16", batch),
            "Invalid GVHMR reference condition",
        )
        require(
            case["input_manifest_sha256"]
            == baseline_case["input_manifest_sha256"]
            == fast["input_manifest_sha256"]
            and settings["timing_protocol_sha256"]
            == baseline_settings["timing_protocol_sha256"]
            == fast["timing_protocol_sha256"]
            and settings["timed_input_ids_sha256"]
            == baseline_settings["timed_input_ids_sha256"],
            "FastHMR and GVHMR use different fixed inputs or report ordering",
        )
        require(
            (
                settings["body_window_frames"],
                baseline_settings["body_window_frames"],
                settings["host_threads"],
                baseline_settings["host_threads"],
                settings["feature_batch_size"],
                baseline_settings["hmr2_crop_batch"],
                settings["sequence_execution"],
                settings["use"],
            )
            == (16, 16, 2, 4, 4, 16, "serial", "speed_reference_only"),
            "Unexpected body timing boundary",
        )
        stage = obj(obj(raw["summary"])["stage_wall"])
        saved = obj(baseline["stage_wall"])
        require(
            count(stage["samples"]) == count(saved["samples"]) == 300
            and count(stage["items"]) == count(saved["items"]) == 300 * 16 * batch,
            "Unexpected body timing population",
        )
        other = gvhmr[batch]
        require(
            (
                number(saved["p50_ms"]) / batch,
                number(saved["p95_ms"]) / batch,
                mib(baseline["nvml_device_peak_sampled_bytes"]),
            )
            == (other.p50, other.p95, other.memory),
            "GVHMR records disagree between FastHMR JSON and the low-precision CSV",
        )
        current.append(
            BodyCase(
                f"fasthmr/PT16/{batch}",
                f"{FAST_PATH} PT16",
                batch,
                number(stage["p50_ms"]) / batch,
                number(stage["p95_ms"]) / batch,
                mib(raw["nvml_device_peak_sampled_bytes"]),
                REFERENCE,
                REFERENCE,
            )
        )
    require(seen == set(BATCHES), "Missing FastHMR batch")
    return current


def _batch_assets(batch: int, cases: list[BodyCase]) -> tuple[Table, Bars]:
    rows = [case for case in cases if case.batch == batch]
    table = Table(
        f"hmr-b{batch}-speed",
        f"身体復元の速度と VRAM · 系列 B{batch}",
        f"{SCOPE} {CONTEXT} {QUALITATIVE}",
        (
            Column("経路 / Backend", 105),
            Column("P50", 55, 2, "ms/系列"),
            Column("P95", 55, 2, "ms/系列"),
            Column("NVML", 54, 1, "MiB"),
            Column("状態", 72),
        ),
        tuple(
            Row(
                c.key,
                (c.label, c.p50, c.p95, c.memory, c.status),
                exclusion=c.exclusion,
            )
            for c in rows
        ),
    )
    graph = Bars(
        Table(
            f"hmr-b{batch}-wall-vram",
            f"身体復元の遅延と VRAM · 系列 B{batch}",
            f"{SCOPE}\n{CONTEXT}\n{QUALITATIVE}",
            (
                table.columns[0],
                Column("P50", 72, 2, "ms/人・フレーム"),
                table.columns[3],
                table.columns[4],
            ),
            tuple(
                Row(
                    c.key,
                    (
                        c.label,
                        c.p50 / 16 if c.p50 is not None else None,
                        c.memory,
                        c.status,
                    ),
                    exclusion=c.exclusion,
                )
                for c in rows
            ),
        ),
        metrics=(1, 2),
    )
    return table, graph


def _history(sources: Sources) -> Table:
    cases = _csv_cases(sources, INITIAL, ("PT32", "PT16", "TRT16"))
    probe = sources.json(EXPORT)
    require(
        all(
            obj(probe[k])["status"] == "failed"
            for k in ("torch_export", "torchscript_export")
        )
        and probe["trt_engine"] == "not_built",
        "FastHMR TRT16 export outcome differs",
    )
    smoke = sources.json(SMOKE)
    frames = len(array(smoke["source_frames"]))
    require(frames > 0, "FastHMR PT32 smoke has no frames")
    require(
        "pt32" in obj(smoke["prediction_sha256"]), "FastHMR PT32 smoke is not confirmed"
    )
    rows = [
        Row(
            c.key,
            (c.label, c.p50, c.p95, c.memory, c.status),
            f"初回 · 系列 B{c.batch}",
            c.exclusion,
        )
        for c in cases
    ]
    rows.extend(
        [
            Row(
                "fasthmr/PT32",
                (
                    f"{FAST_PATH} PT32",
                    None,
                    None,
                    None,
                    f"{frames} フレームの動作確認のみ",
                ),
                "FastHMR の補助試行",
                "速度未計測",
            ),
            Row(
                "fasthmr/TRT16",
                (f"{FAST_PATH} TRT16", None, None, None, "書き出し失敗・実行不能"),
                "FastHMR の補助試行",
                "書き出し失敗・実行不能",
            ),
        ]
    )
    return Table(
        "hmr-historical",
        "身体復元の旧計時と未計測条件",
        "初回の壁時計はバッチ終端同期の修正前。NVML は全体。初回の出力差基準と再計時の判定は別。"
        " FastHMR HMR 2.0 ベースは公開 Mask-ToMe の適用外。",
        (
            Column("経路 / Backend", 110),
            Column("P50", 55, 2, "ms/系列"),
            Column("P95", 55, 2, "ms/系列"),
            Column("NVML", 55, 1, "MiB"),
            Column("状態", 88),
        ),
        tuple(rows),
    )


def build(sources: Sources) -> tuple[Table | Bars, ...]:
    cases = load_current(sources)
    assets = [asset for batch in BATCHES for asset in _batch_assets(batch, cases)]
    return (*assets, _history(sources))
