"""Record the complete execution list, without running reconstruction or scoring."""

import argparse
from dataclasses import asdict

from common import (
    BUNDLE_SHA,
    DEFAULT_INPUT,
    HERE,
    REFERENCE_SHA,
    ROOT,
    SOURCE_PINS,
    code_files,
    code_fingerprint,
    git_head,
    prior_run,
    source_inventory,
    write_json,
)
from hydra import compose, initialize_config_dir
from methods import make_method
from omegaconf import OmegaConf

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=type(DEFAULT_INPUT), default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.inputs
    output = root / "integrated_comparison_v1"
    if (HERE / "execution-plan.json").exists() or output.exists():
        raise ValueError("Execution list/output already exists")
    bundle = load_bundle(root / "fixed_input_bundle_v1", expected_sha256=BUNDLE_SHA)
    for directory in SOURCE_PINS:
        source_inventory(root, directory)
    with initialize_config_dir(
        config_dir=str(ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        config = compose(config_name="pipeline")
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    rows = []

    def single(name, kind, directory, category="method"):
        previous = prior_run(root, directory, f"results/{name}/all_views")
        row = {
            "name": name,
            "kind": kind,
            "category": category,
            "config": previous.config,
            "prior": {"directory": directory, "prefix": f"results/{name}"},
        }
        # Constructor/config equivalence is checked before any replay, no defaults inferred.
        method = make_method(row, bundle)
        if method.name != name or dict(method.config) != previous.config:
            import json

            if json.loads(json.dumps(dict(method.config))) != previous.config:
                raise ValueError(
                    f"Frozen adapter differs from child configuration: {name}"
                )
        rows.append(row)
        return row["config"]

    single("pipeline_baseline", "baseline", "foundation_v2", "baseline")
    single("without_loess", "baseline", "foundation_v2", "reference_no_labels")
    single("quadratic_logit_3", "decode", "heatmap_decoding_v4")
    decoder = single("selected_affine", "decode", "heatmap_decoding_v4")
    topk = single("topk_multihypothesis", "topk", "topk_multihypothesis_v2")
    for estimator, temporal in (
        ("map", "none"),
        ("mean", "none"),
        ("map", "rts"),
        ("mean", "rts"),
    ):
        single(f"candidate_gmm_{estimator}_{temporal}", "gmm", "candidate_gmm_v2")
    dense = single("dense_selected", "dense", "dense_likelihood_v2")
    single("dense_temporal", "dense", "dense_likelihood_v2")
    single("segment_physics_candidates", "physics", "segment_physics_offset_v3")
    single("segment_physics_heatmap", "physics", "segment_physics_offset_v3")
    ba = single("court_ball_ba", "ba", "court_ball_ba_v2")
    if any(topk[k] != v for k, v in asdict(runtime.ball_topk).items()):
        raise ValueError("Integrated composition changed selected TopK values")
    if (
        OmegaConf.to_container(config.ball_reconstruction.bundle_adjustment)
        != ba["adjustment"]
    ):
        raise ValueError("Integrated composition changed selected BA values")
    for downstream in ("baseline", "topk", "dense"):
        rows.append(
            {
                "name": f"ba_cameras_{downstream}",
                "kind": "composition",
                "category": "combination",
                "config": {
                    "composition": "ba_cameras",
                    "downstream": downstream,
                    "ba": ba,
                    **({"topk": topk} if downstream == "topk" else {}),
                    **({"dense": dense} if downstream == "dense" else {}),
                    "time_policy": "BA cameras only; all offsets zero; full BA is separate",
                    "fold_policy": "fit BA from the retained views of this fold only",
                },
            }
        )
    rows.extend(
        [
            {
                "name": "selected_affine_topk",
                "kind": "composition",
                "category": "combination",
                "config": {
                    "composition": "decode_candidates",
                    "downstream": "topk",
                    "topk": topk,
                    "decoding": decoder["decoding"],
                    "candidate_policy": "same raw cells/scores/masks; existing fit at each cell; normalized affine then original pixels",
                },
            },
            {
                "name": "topk_loess",
                "kind": "composition",
                "category": "combination",
                "config": {
                    "composition": "topk_then_loess",
                    "downstream": "topk",
                    "topk": topk,
                    "smoothing": "selected raw candidates through unchanged BallPointsModule and triangulation; no gap filling",
                },
            },
            {
                "name": "selected_affine_ba",
                "kind": "composition",
                "category": "combination",
                "config": {
                    "composition": "decode_then_ba",
                    "ba": ba,
                    "decoding": decoder["decoding"],
                    "fold_policy": "decode, threshold/gate, consensus/LOESS then BA inside the same fold",
                },
            },
        ]
    )
    diagnostics = []
    for name in (
        "annotation_oracle",
        "annotation_time_oracle",
        "candidate1_oracle",
        "candidate8_oracle",
    ):
        prefix = f"scoring_diagnostics_only/{name}"
        previous = prior_run(root, "error_decomposition_v2", f"{prefix}/all_views")
        diagnostics.append(
            {
                "name": name,
                "kind": "scoring_only",
                "category": "diagnostic_labels_used",
                "config": previous.config,
                "prior": {"directory": "error_decomposition_v2", "prefix": prefix},
            }
        )
    for row in [*rows, *diagnostics]:
        row.update(
            code_commit=git_head(),
            input_bundle_sha256=BUNDLE_SHA,
            reference_sha256=REFERENCE_SHA,
            output=str(
                output
                / ("diagnostics" if row["kind"] == "scoring_only" else "results")
                / row["name"]
            ),
            folds=["all_views", "without_cam0", "without_cam1", "without_cam2"],
        )
    plan = {
        "schema": "ball3d_integrated_execution_plan.v1",
        "issue": 1568,
        "integration_boundary_commit": "cecb863797d3e5468f26005d1286e29c7eaf997a",
        "code_commit": git_head(),
        "code_fingerprint": code_fingerprint(),
        "code_commit_policy": "This commit owns every executable byte. The later plan-only commit is the checkout HEAD at execution; both are recorded.",
        "inputs": str(root),
        "output": str(output),
        "bundle_sha256": BUNDLE_SHA,
        "reference_sha256": REFERENCE_SHA,
        "selection": "Use each child front-half selection/calibration unchanged. Do not refit any settings.",
        "source_manifest_sha256": SOURCE_PINS,
        "methods": rows,
        "diagnostics": diagnostics,
        "excluded_combinations": [
            {
                "combination": "BA clock + top-K/dense",
                "reason": "Existing top-K/dense kernels assume simultaneous frames. Applying nonzero clocks requires a new asynchronous likelihood/candidate model; camera-only rows explicitly use zero clocks.",
            },
            {
                "combination": "GMM K/support/failure-policy changes",
                "reason": "Child next proposals require changing the model or retuning; not a composition of existing frozen parts.",
            },
            {
                "combination": "physics segment continuity/event support/dense convergence changes",
                "reason": "Child proposals require new objectives or tuning, not existing component composition. Unconverged heatmap row remains in all tables.",
            },
            {
                "combination": "top-K null/motion cost changes",
                "reason": "Requires retuning, prohibited. The existing selected-candidate + LOESS composition is included.",
            },
            {
                "combination": "LINE-based BA",
                "reason": "Pinned bundle contains no raw LINE coordinates. Re-inference/new bundle is outside scope.",
            },
        ],
        "display_selection": "After freezing and scoring, baseline plus the three non-diagnostic nonempty rows with smallest evaluation LOVO (median,p90,>20 rate); only chooses video display, never settings.",
        "resource_policy": {
            "gpu": False,
            "method_workers": 1,
            "pytest_workers_max": 2,
            "video_streaming": True,
        },
    }
    write_json(HERE / "execution-plan.json", plan)
    write_json(HERE / "code-files.json", code_files())
    lines = [
        "# 実行前に固定する一覧",
        "",
        f"実行コード: `{plan['code_commit']}`。この後の一覧だけのcommitはcheckout出自として別記する。",
        "各行は同一bundle/GT checksum（execution-plan.json）と全view+3 LOVO。既存の選定条件をそのまま再生する。",
        "GTを使う4 oracleは別の診断で、方式・組合せの順位へ含めない。LOESSなしも参考診断。",
        "",
        "| 行 | 種類 | 設定の正本/照合元 | 出力先（integrated_comparison_v1/以下） |",
        "|---|---|---|---|",
    ]
    for row in [*rows, *diagnostics]:
        prior = row.get("prior", {})
        setting = prior.get("directory", "execution-plan.json の凍結合成設定")
        lines.append(
            f"| {row['name']} | {row['category']} | {setting} | {row['output'].split('integrated_comparison_v1/')[1]} |"
        )
    lines.extend(["", "合成対象外（別入力へ置換しない）:", ""])
    lines.extend(
        f"- {r['combination']}: {r['reason']}" for r in plan["excluded_combinations"]
    )
    (HERE / "execution-list.md").write_text("\n".join(lines) + "\n")
    print(
        f"Frozen {len(rows)} prediction rows and {len(diagnostics)} diagnostic rows; no reconstruction/scoring executed."
    )


if __name__ == "__main__":
    main()
