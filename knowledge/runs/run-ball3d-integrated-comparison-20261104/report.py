"""Foundation scoring, cohort accounting and streamed portable comparison media."""

import argparse
import csv
import html
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from common import (
    BUNDLE_SHA,
    REFERENCE_SHA,
    code_fingerprint,
    load_script,
    write_json,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import error_samples, score_comparison
from src.tennis_scene.visualization.ball3d.rendering import (
    BallComparisonRenderer,
    render_video,
)
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_comparison,
    render_extremes,
)
from src.utils.checksum import dual_sha256

SEGMENTS = ("evaluation", "tuning", "all")


def load_results(output, plan):
    runs = {}
    for row in [*plan["methods"], *plan["diagnostics"]]:
        directory = "diagnostics" if row["kind"] == "scoring_only" else "results"
        receipt = json.loads(
            (output / "completed" / (row["name"] + ".json")).read_text()
        )
        runs[row["name"]] = tuple(
            load_run(
                output / directory / row["name"] / key,
                expected_sha256=receipt["manifests"][key],
            )
            for key in row["folds"]
        )
    return runs


def metric_rows(scores, plan):
    rows = []
    categories = {
        r["name"]: r["category"] for r in [*plan["methods"], *plan["diagnostics"]]
    }
    for name, method in scores["methods"].items():
        for segment in SEGMENTS:
            for camera in ("pooled", "cam0", "cam1", "cam2", "all_views"):
                if camera == "pooled":
                    metrics = method["lovo_reprojection_px"][segment]
                else:
                    key = "all_views" if camera == "all_views" else "without_" + camera
                    metrics = method["folds"][key][segment]["reprojection_px"]
                rows.append(
                    {
                        "method": name,
                        "category": categories[name],
                        "segment": segment,
                        "held_out_camera": camera,
                        "count": metrics["count"],
                        "eligible": metrics["label_count"],
                        "coverage": metrics["scored_label_coverage"],
                        "median_px": metrics["median"],
                        "p90_px": metrics["p90"],
                        "above20_rate": metrics["above_threshold_rate"],
                        "all_view_valid_rate": method["folds"]["all_views"][segment][
                            "valid_3d_rate"
                        ],
                    }
                )
    return rows


def cohort_reports(runs, bundle, annotations, scores, output):
    # Reuse the already reviewed #1560 partition implementation and its tests.
    cohort = load_script(
        "run-ball3d-topk-multihypothesis-20261103",
        "cohort_report.py",
        "integrated_cohorts",
    )
    indices = {}
    for name, values in runs.items():
        samples = [
            error_samples(r, bundle, annotations)
            for r in values
            if r.held_out_camera is not None
        ]
        merged = {key: np.concatenate([s[key] for s in samples]) for key in samples[0]}
        indices[name] = cohort.index_samples(
            merged, bundle.inputs.camera_ids, bundle.inputs.num_frames
        )
    reports, table = {}, []
    for name, index in indices.items():
        strata = cohort.summarize(
            index, indices["pipeline_baseline"], bundle.inputs.camera_ids
        )
        report = {"condition": name, "strata": strata}
        for row in strata:
            # Every marginal must also match the unmodified foundation scorer.
            segment, camera, kind = (
                row["segment"],
                row["held_out_camera"],
                row["reference_kind"],
            )
            method = scores["methods"][name]
            metrics = (
                method["lovo_reprojection_by_reference_kind"][segment][kind]
                if camera == "all"
                else method["folds"]["without_" + camera][segment][
                    "reprojection_by_reference_kind"
                ][kind]
            )
            actual = row["totals"]["method"]
            for key in ("count", "median", "p90", "above_threshold_rate"):
                if actual[key] != metrics[key]:
                    raise ValueError(
                        f"Cohort/scorer marginal mismatch: {name}/{segment}/{camera}/{kind}/{key}"
                    )
        reports[name] = report
        table.extend(cohort.table_rows(report))
    write_json(output / "cohorts.json", reports)
    write_csv(output / "cohorts.csv", table)
    with (output / "cohort-samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                "held_out_camera",
                "frame",
                "reference_kind",
                "cohort",
                "method_error_px",
                "baseline_error_px",
            ]
        )
        baseline = indices["pipeline_baseline"]
        for name, index in indices.items():
            for key in sorted(index.keys() | baseline.keys()):
                a, b = index.get(key), baseline.get(key)
                writer.writerow(
                    [
                        name,
                        *key,
                        (a or b)[1],
                        "common"
                        if a and b
                        else "method_only"
                        if a
                        else "baseline_only",
                        None if a is None else a[0],
                        None if b is None else b[0],
                    ]
                )
    return reports


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value, rate=False):
    return "—" if value is None else f"{value:.2%}" if rate else f"{value:.3f}"


def metric_table(rows):
    return (
        "<table><thead><tr><th>方式</th><th>median px</th><th>p90 px</th><th>&gt;20 px</th><th>採点/対象</th><th>coverage</th><th>全view有効3D率</th></tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in [
                    r["method"],
                    number(r["median_px"]),
                    number(r["p90_px"]),
                    number(r["above20_rate"], True),
                    f"{r['count']}/{r['eligible']}",
                    number(r["coverage"], True),
                    number(r["all_view_valid_rate"], True),
                ]
            )
            + "</tr>"
            for r in rows
        )
        + "</tbody></table>"
    )


def cohort_table(reports, names, segment):
    def cell(metric):
        return f"{metric['count']}; {number(metric['median'])}/{number(metric['p90'])}; {number(metric['above_threshold_rate'], True)}"

    rows = []
    for name in names:
        r = next(
            r
            for r in reports[name]["strata"]
            if r["segment"] == segment
            and r["held_out_camera"] == "all"
            and r["reference_kind"] == "all"
        )
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(v) + "</td>"
                for v in [
                    name,
                    cell(r["common"]["method"]),
                    cell(r["common"]["baseline"]),
                    cell(r["method_only"]["method"]),
                    cell(r["baseline_only"]["baseline"]),
                ]
            )
            + "</tr>"
        )
    return (
        "<table><thead><tr><th>方式</th><th>共通:方式</th><th>共通:baseline</th><th>追加:方式のみ</th><th>喪失:baselineのみ</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def plot_metrics(rows, output, segment, names):
    figure = Figure(figsize=(14, max(7, 0.3 * len(names) + 3)), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    for axis, camera in zip(axes, ("pooled", "cam0", "cam1", "cam2"), strict=True):
        selected = [
            next(
                r
                for r in rows
                if r["method"] == n
                and r["segment"] == segment
                and r["held_out_camera"] == camera
            )
            for n in names
        ]
        y = np.arange(len(selected))
        median = np.array(
            [np.nan if r["median_px"] is None else r["median_px"] for r in selected]
        )
        p90 = np.array(
            [np.nan if r["p90_px"] is None else r["p90_px"] for r in selected]
        )
        axis.plot(median, y, "o", label="median")
        axis.plot(p90, y, "x", label="p90")
        axis.axvline(20, color="gray", linewidth=0.7)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_yticks(
            y,
            [
                f"{r['method']}  {r['count']}/{r['eligible']} ({r['coverage']:.0%})"
                for r in selected
            ]
            if camera == "pooled"
            else [
                f"{r['count']}/{r['eligible']} ({r['coverage']:.0%})" for r in selected
            ],
            fontsize=7,
        )
        axis.invert_yaxis()
        axis.set_title(camera)
        axis.set_xlabel("LOVO error px; n/eligible (coverage)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        segment
        + ": errors conditional on scored predictions; empty errors are unmeasured"
    )
    figure.tight_layout()
    figure.savefig(output)
    figure.clear()


def restrict_run(run, start, stop):
    result = run.result
    mask = (
        result.valid & (result.frame_indices >= start) & (result.frame_indices < stop)
    )
    return replace(
        run,
        result=replace(
            result,
            valid=mask,
            positions=np.where(mask[:, None], result.positions, 0),
            used_views=result.used_views & mask,
            covariance=None
            if result.covariance is None
            else np.where(mask[:, None, None], result.covariance, 0),
        ),
    )


def write_page(output, plan, rows, cohorts, verification, images, events, shortlist):
    """Portable relative references; never embed video bytes in HTML."""
    names = [r["name"] for r in plan["methods"]]
    diagnostics = [r["name"] for r in plan["diagnostics"]]
    parts = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>球3D 統合比較 #1568</title>',
        "<style>body{font:15px system-ui;margin:2em auto;max-width:1500px;padding:0 1em;background:#111820;color:#e8edf1}a{color:#95d0ff}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:.45em;border:1px solid #536170}th{background:#293846}img,video{max-width:100%;height:auto}details{margin:1em 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow-x:auto}</style>",
        "<h1>球3D 全方式・組合せの統合比較</h1>",
        "<p>対象: meiji_3cam/video_000/clip_000、3 camera・1010 frame。主評価は後半 [505,1010)、前半 [0,505) の選定済み条件を再調整せずCPUで再生した。<strong>人間が採否を決める。既定は変更していない。</strong></p>",
        "<p>指標はLOVO: 保持する2 cameraから推定し、除外cameraの元校正・offset 0で採点。全viewの自己再投影と独立3D精度は異なる。各方式の採点集合が違うため、誤差と採点数・対象数・coverageを併せて読む。欠測は0誤差ではない。固定校正の出自自体は全viewを使っており、校正段階まで独立したLOVOではない。</p>",
        "<p>緑の十字=直接観測、白い菱形=人間承認の推定、色付きring=全view方式の再投影。cropの色付き斜め十字=当該cameraを除外したLOVO予測。各記号は推定の出自を保つ。末尾の除外外挿・座標なしには正解を描かない。</p>",
        "<p>BA camera組合せは各foldの保持viewだけでfitし、cameraだけを次段へ渡し、offsetは全て0。clockも用いるcourt_ball_ba単独行と区別する。全方式共通LOVO集合は失敗したheatmap物理方式を含むため空で、以下ではbaselineとの2組の共通集合を使う。</p>",
        '<nav><a href="#evaluation">後半主表</a> · <a href="#cohorts">共通/追加/喪失</a> · <a href="#videos">比較動画</a> · <a href="#crops">イベント・外れcrop</a> · <a href="metrics.csv">全区間/camera CSV</a> · <a href="cohorts.csv">全cohort CSV</a> · <a href="execution-plan.json">実行一覧</a> · <a href="prior-comparison.json">保存予測照合</a></nav>',
    ]
    for segment in SEGMENTS:
        parts.append(
            f'<h2 id="{segment}">{segment}: LOVO 主表</h2><div class="scroll">'
            + metric_table(
                [
                    r
                    for r in rows
                    if r["method"] in names
                    and r["segment"] == segment
                    and r["held_out_camera"] == "pooled"
                ]
            )
            + "</div>"
        )
        for camera in ("cam0", "cam1", "cam2", "all_views"):
            parts.append(
                f"<details><summary>{segment} / {camera}</summary>"
                + metric_table(
                    [
                        r
                        for r in rows
                        if r["method"] in names
                        and r["segment"] == segment
                        and r["held_out_camera"] == camera
                    ]
                )
                + "</details>"
            )
        parts.append(
            f'<img src="metrics-{segment}.png" alt="{segment} errors, counts and coverage by held-out camera">'
        )
    parts.append(
        '<h2 id="cohorts">baselineとの共通・追加・喪失</h2><p>各cellは n; median/p90 px; &gt;20px率。共通の2列は同じcamera/frame。共通+追加=方式全採点数、共通+喪失=baseline全採点数を全区間/camera/kindで検算し、基盤scorerの周辺件数とも一致した。0件の誤差は未測定。共通集合にも可用性による選択が残る。</p>'
    )
    for segment in SEGMENTS:
        parts.append(
            f"<h3>{segment}</h3><div class='scroll'>"
            + cohort_table(cohorts, names, segment)
            + "</div>"
        )
    parts.append(
        "<details><summary>正解を使う参考診断（実運用方式ではない）</summary><p>annotation / candidate / relative-clock oracleは採点専用であり、方式入力には渡していない。相対clockは前半の全camera注釈を使う診断で、blind LOVOでも同期真値でもない。oracleは数学的に保証した下限ではない。</p>"
    )
    for segment in SEGMENTS:
        parts.append(
            f"<h3>{segment}</h3>"
            + metric_table(
                [
                    r
                    for r in rows
                    if r["method"] in diagnostics
                    and r["segment"] == segment
                    and r["held_out_camera"] == "pooled"
                ]
            )
            + cohort_table(cohorts, diagnostics, segment)
        )
    parts.append(
        '</details><h2 id="videos">比較動画（全viewによる推定）</h2><p>動画は隣の相対pathから開く。HTMLへ動画を埋め込まない。全frameを1枚ずつ読み書きした。</p>'
    )
    for directory, title in (
        ("all-methods", "全20予測条件（LOESSなし参考を含む、正解oracleを除く）"),
        ("shortlist", "後半LOVO主指標による表示候補"),
    ):
        parts.append(
            f'<h3>{title}</h3><a href="{directory}/comparison.mp4">MP4を開く</a><video controls preload="metadata" src="{directory}/comparison.mp4"></video>'
        )
    parts.append(
        "<p>絞り込み: "
        + html.escape(", ".join(shortlist))
        + "。baselineを必ず含み、残る3条件を後半LOVO中央値→p90→20px超率の順で選んだ。表示の選択だけで、設定は変えていない。</p>"
    )
    parts.append(
        '<h2 id="crops">誤差上位・イベント近傍</h2><p>cropは参照中心の160×120 px。大きく外れた予測はcrop外になる。完全な座標と両側の残差はcohort-samples.csvに残す。camera別のイベント行・注記・1frame差は統合せず保持する。</p>'
    )
    for title, path in images:
        parts.append(
            f'<h3>{html.escape(title)}</h3><img src="{path.as_posix()}" alt="{html.escape(title)}">'
        )
    for event in events:
        parts.append(
            "<details><summary>"
            + html.escape(
                f"{event['event']} f{event['frame_index']} / source rows {len(event['source_rows'])}"
            )
            + "</summary><pre>"
            + html.escape(
                json.dumps(event["source_rows"], ensure_ascii=False, indent=2)
            )
            + f'</pre><img src="{event["image"]}" alt="Event neighbourhood with all-view and LOVO predictions"></details>'
        )
    parts.append(
        '<h2>制限・推奨の読み方</h2><p>1 clip、固定の近似pinhole校正、推定を含む2D参照。独立3D真値・物理clock真値・独立court真値はない。3Dの滑らかさや全view再投影の改善だけで採用を決めない。各方式の長所・短所と推奨案は隣の conclusions.md / knowledgeノードを参照。人間が採否を決める。</p><p><a href="scores.json">全採点・参照kind・イベント・運動診断 JSON</a> · <a href="cohort-samples.csv">camera/frame別の所属と残差</a> · <a href="verification.json">decode・出自検証</a></p><pre>'
        + html.escape(json.dumps(verification, ensure_ascii=False, indent=2))
        + "</pre></html>"
    )
    page = output / "index.html"
    page.write_text("\n".join(parts))
    if broken_html_references(page):
        raise ValueError(
            f"Broken portable page references: {broken_html_references(page)}"
        )
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    started = time.perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    plan = json.loads((output / "execution-plan.json").read_text())
    if code_fingerprint() != plan["code_fingerprint"]:
        raise ValueError("Report executable differs from committed execution list")
    bundle = load_bundle(
        Path(plan["inputs"]) / "fixed_input_bundle_v1", expected_sha256=BUNDLE_SHA
    )
    annotations = load_annotations(
        Path(plan["inputs"]) / "human_annotations/ground_truth_v1/ground_truth.json",
        bundle,
        expected_sha256=REFERENCE_SHA,
    )
    runs = load_results(output, plan)
    scores = score_comparison(runs, bundle, annotations)
    write_json(output / "scores.json", scores)
    rows = metric_rows(scores, plan)
    write_csv(output / "metrics.csv", rows)
    cohorts = cohort_reports(runs, bundle, annotations, scores, output)
    eligible_names = [
        r["name"]
        for r in plan["methods"]
        if r["category"] not in ("baseline", "reference_no_labels")
        and scores["methods"][r["name"]]["lovo_reprojection_px"]["evaluation"]["count"]
    ]
    shortlist = [
        "pipeline_baseline",
        *sorted(
            eligible_names,
            key=lambda n: tuple(
                scores["methods"][n]["lovo_reprojection_px"]["evaluation"][k]
                for k in ("median", "p90", "above_threshold_rate")
            ),
        )[:3],
    ]
    write_json(
        output / "shortlist.json",
        {
            "methods": shortlist,
            "policy": plan["display_selection"],
            "settings_changed": False,
        },
    )
    print(
        "Scoring/cohort conservation passed for every row/split/camera/reference kind",
        flush=True,
    )
    if args.score_only:
        return
    videos = {c: args.videos / f"{c}.mp4" for c in bundle.inputs.camera_ids}
    video_inputs = verify_videos(bundle, videos)
    predicted = {r["name"]: runs[r["name"]] for r in plan["methods"]}
    for segment in SEGMENTS:
        plot_metrics(rows, output / f"metrics-{segment}.png", segment, tuple(predicted))
    plot_comparison(predicted, bundle, output / "trajectories.png", annotations)
    media = {}
    for directory, names in (
        ("all-methods", list(predicted)),
        ("shortlist", shortlist),
    ):
        displayed = {
            f"{i + 1:02d} {name}": runs[name][0] for i, name in enumerate(names)
        }
        with BallComparisonRenderer(
            bundle.inputs, displayed, videos, annotations=annotations
        ) as renderer:
            media[directory] = render_video(
                renderer, output / directory / "comparison.mp4"
            )
        print(f"Streamed and decoded {directory}: 1010 frames", flush=True)
    images = [
        ("全方式: 全view軌道・LOVO残差（欠測を跨がない）", Path("trajectories.png"))
    ]
    for label, chosen in (
        ("all", predicted),
        ("shortlist", {n: runs[n] for n in shortlist}),
    ):
        path = Path(f"extremes-{label}.png")
        receipt = render_extremes(
            chosen, bundle, videos, output / path, annotations, count=8
        )
        write_json(output / f"extremes-{label}.json", receipt)
        images.append(
            (f"{label}: 最大LOVO誤差frame、斜め十字は除外cameraへの予測", path)
        )
    (output / "events").mkdir(exist_ok=False)
    events = []
    groups = {}
    for event in annotations.reference.events:
        groups.setdefault((event["event"], event["frame_index"]), []).append(
            dict(event)
        )
    for (kind, frame), source_rows in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        # Existing crop renderer selects the largest LOVO error inside the fixed event window.
        restricted = {
            n: tuple(
                restrict_run(r, max(0, frame - 3), min(1010, frame + 4)) for r in values
            )
            for n, values in predicted.items()
        }
        path = Path("events") / f"{kind}-{frame:04d}.png"
        receipt = render_extremes(
            restricted, bundle, videos, output / path, annotations, count=1
        )
        if not receipt["frames"]:
            raise ValueError(f"No event crop support: {kind}/{frame}")
        events.append(
            {
                "event": kind,
                "frame_index": frame,
                "source_rows": source_rows,
                "image": path.as_posix(),
                **receipt,
            }
        )
    write_json(output / "event-crops.json", events)
    verification = {
        "bundle_sha256": BUNDLE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "execution": json.loads((output / "execution-verification.json").read_text()),
        "videos": media,
        "input_videos": video_inputs,
        "common_all_method_count": len(
            scores["common_lovo_cohort"]["camera_frame_pairs"]
        ),
        "cohort_conservation_and_scorer_marginals": True,
        "shortlist": shortlist,
        "event_source_rows": len(annotations.reference.events),
        "event_crop_groups": len(events),
        "html_video_policy": "relative files; no embedded video bytes",
        "report_seconds": time.perf_counter() - started,
        "report_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "broken_html_references": [],
    }
    write_json(output / "verification.json", verification)
    write_page(output, plan, rows, cohorts, verification, images, events, shortlist)
    inventory = {
        str(p.relative_to(output)): {
            "size_bytes": p.stat().st_size,
            "sha256": dual_sha256(p),
        }
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "artifacts.json"
    }
    write_json(
        output / "artifacts.json",
        {
            "schema": "ball3d_integrated_artifacts.v1",
            "files": inventory,
            "execution_plan_sha256": dual_sha256(output / "execution-plan.json"),
            "bundle_sha256": BUNDLE_SHA,
            "reference_sha256": REFERENCE_SHA,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "artifacts_sha256": dual_sha256(output / "artifacts.json"),
                "files": len(inventory),
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
