"""Publish bounded CPU summaries, comparison tables and a standalone research figure."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from support import GROUPS, HERE, SCENES, stats, write

from src.utils.data.image_record_store import file_sha256


def percentage(n: float, total: float) -> str:
    return f"{n / total * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    comparison = json.loads((args.plans / "comparison.json").read_text())
    val = json.loads((args.analysis / "val-summary.json").read_text())
    baseline = json.loads((args.analysis / "baseline-geometry.json").read_text())
    audit = json.loads(args.audit.read_text())
    assert audit["images_verified"] == 1102
    assert audit["previous_counts_and_histograms_match"]
    costs = json.loads((HERE / "cost-scenarios.json").read_text())
    for source, target in (
        (args.analysis / "val-summary.json", "val-summary.json"),
        (args.analysis / "baseline-geometry.json", "baseline-geometry.json"),
        (args.plans / "comparison.json", "comparison.json"),
        (args.audit, "saved-pilot.json"),
        (args.analysis / "val-examples.png", "val-examples.png"),
    ):
        shutil.copyfile(source, args.output / target)
    row_paths = [
        args.analysis / name
        for name in (
            "baseline-geometry-rows.json",
            "val-camera-rows.json",
            "capture-rows.json",
        )
    ]
    write(
        args.output / "source-files.json",
        {
            "files": [
                {
                    "path": p.as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
                for p in row_paths
                + [args.plans / name / "rows.json" for name in comparison["variants"]]
            ]
        },
    )
    text = [
        "# D2のCPU比較（L4描画・QAは未実施）",
        "",
        "seed42、B00〜B03。testの幾何・写り方・精度は集計せず、valだけを参照した。",
        "",
        "## 前回データの再検査",
        "",
        "Driveから別rootへstrict fetchし、1,102 JPEG・全store・教師を再検査した。全1,209候補のIDとKP数、保存計画とQA ledgerのcameraを照合した。",
        "",
        "| 段階 | N | 全面 | ほぼ全面 | 部分 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, data in (
        ("幾何検査前", baseline["all_candidates"]),
        ("幾何通過", baseline["geometry_pass"]),
        ("幾何拒否", baseline["geometry_rejected"]),
    ):
        text.append(
            f"| {label} | {data['n']:,} | {data['framing']['full']} | {data['framing']['near_full']} | {data['framing']['partial']:,} |"
        )
    text.extend(
        [
            "| QA採択 | 1,102 | 8 | 10 | 1,084 |",
            "| 実画像val | 4,468 | 4,400 | 55 | 13 |",
            "",
            "幾何拒否86件は全て部分、QA拒否21件も全て部分。前回のscene別KP数ヒストグラムまで一致した。12件はnear/farが未定義で保存projectionがnullだが、診断では物理14点を直接投影し、幾何拒否のまま写り方を数えた。ラベルを補完・再採択していない。",
            "",
            "QAは前回のD5 ledgerを検査したもので、JPEGからalpha/depthを再判定した結果ではない。採択/幾何通過98.13%、採択/全候補91.15%。全QA拒否理由は対象可視KP不足。採択内ROI穴率0、α<0.9のp90は2.53%、最大38.03%。αは診断のままで閾値化していない。",
            "",
            "既存の画質代表図は[前回のreview](../../run-court-scale-pilot-l4-20261023/review/review.json)と同runのscene別PNGを参照する。低視点のぼけとB03の暗さを再確認し、今回の候補画質に読み替えない。",
            "",
            "## 原因とvalの支持範囲",
            "",
            "現D2の1,209点のうち854点は対象court内、1,079点で対象14点の一部がcamera後方だった。全面8点は全てbaseline外側。全14点が前方にある130点でも、必要HFOVの中央値は約159°。現在のcamera配置を固定した広角化だけでは大半を全面にできない。",
            "",
            "固定poseでHFOVを110/125/140/155°へ変えた全面数は21/32/43/54。固定camera位置・jitter XYで注視高を0/0.75/1.5mへ揃えても全面数は全て8だった。この診断と、通常recipe変更では点ID・jitterも変わることを区別する。",
            "",
            "val cameraは各動画の辞書順先頭1枚を既存fit_planar_cameraで推定した。fx=fy、画像中心principal point、歪みなし、既知court寸法の仮定付き。249動画中248成功、1件はfocal境界で失敗。成功例の残差/対角長は最大0.00146。低残差は仮定の一意性や真のcamera位置を保証しない。",
            "",
            "| 配置 | 高さp10 / 中央 / p90（m） | 中心距離p10 / 中央 / p90（m） | HFOV p10 / 中央 / p90（°） |",
            "|---|---|---|---|",
        ]
    )
    for label, selected, fields in (
        ("現D2・全候補", baseline["all_candidates"], ("height_m", "distance_xy_m")),
        (
            "val推定・動画各1枚",
            val["camera_estimation"]["all_successful"],
            ("height_m", "distance_xy_m"),
        ),
    ):
        values = [selected[f]["min_p10_median_p90_max"] for f in fields]
        fov = (
            "75 / 92.5 / 110（層の値）"
            if label.startswith("現")
            else " / ".join(
                f"{v:.2f}" for v in selected["hfov_deg"]["min_p10_median_p90_max"][1:4]
            )
        )
        text.append(
            f"| {label} | "
            + " | ".join(" / ".join(f"{v:.2f}" for v in vs[1:4]) for vs in values)
            + f" | {fov} |"
        )
    text.extend(
        [
            "",
            "valは高く遠い配置・比較的狭い画角を示し、今回の低い閉軌道とは異なる。camera高さ1.5〜3mの分布はstrategy実装の固定値でGeometryRecipeの変更項目ではない。今回そのsrcを変更していない。高さ・距離・hull境界からの距離による[要因別集計](factor-bins.json)も保存した。これらの周辺分布の相関を独立した因果効果と見なさない。",
            "",
            "## 同一128軌道prefixの比較",
            "",
            "全条件で4scene・seed42・密度0。128本は96層を一巡して32本追加する。Kは現行のbirth-level／最大64候補cacheの数。QA採択率・Drive容量・新視点の画質は、これらの未描画候補では未測定。",
            "",
            "| 候補 | 全候補C | 幾何通過R（率） | 全面 | ほぼ全面 | 部分 | K | 幾何通過のhull外 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, candidate in comparison["variants"].items():
        c, passed = candidate["all_candidates"], candidate["geometry_pass"]
        f = passed["framing"]
        k = sum(s["caches"] for s in candidate["per_scene"].values())
        text.append(
            f"| {name} | {c['n']:,} | {passed['n']:,}（{percentage(passed['n'], c['n'])}） | {f['full']}（{percentage(f['full'], passed['n'])}） | {f['near_full']}（{percentage(f['near_full'], passed['n'])}） | {f['partial']:,} | {k} | {passed['outside_captured_hull']}（{percentage(passed['outside_captured_hull'], passed['n'])}） |"
        )
    text.extend(
        [
            "",
            "currentは本番recipeの幾何をin-memoryで使用。wide125はHFOV 92.5/110/125°、hull5は拡張5%・margin0.5m、hull25_margin0は拡張25%・margin0m、末尾wide125はその組合せ。その他、D5/D6は固定。0%/5%の差が小さいため、court外側へ届く25%を探索値とした。安全な観測範囲と認定した値ではない。",
            "",
            "| 候補 | B00 全面/ほぼ全面/R | B01 | B02 | B03 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, candidate in comparison["variants"].items():
        text.append(
            f"| {name} | "
            + " | ".join(
                f"{candidate['geometry_pass_by_scene'][s]['framing']['full']}/{candidate['geometry_pass_by_scene'][s]['framing']['near_full']}/{candidate['geometry_pass_by_scene'][s]['n']}"
                for s in SCENES
            )
            + " |"
        )
    text.extend(
        [
            "",
            "25%拡張の2候補だけが全4sceneで全面を持つ。それでも組合せ案のB02/B03全面率は3.06%/1.63%、全scene合計9.38%で、valの98.48%との大きな差が残る。hull外・未観測領域のぼけ／破綻リスクを新規L4で確かめるまで採用を推奨しない。",
            "",
            "## CPU計画へ前回の費用モデルを適用",
            "",
            "次表は同じ128本prefixの見積もり。QA採択率・JPEG/tar単価・cache処理費は前回のscene別実測を仮定して移植しており、新候補の実測ではない。Drive永続化を含む秒数には意図的な中断/復旧由来の固定費も含まれる。",
            "",
            "| 候補 | 描画/QA/local公開 秒/R | 永続化込み 秒/採択想定 | 生成 分 | Drive MB想定 | QA率・画質（新規） |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, candidate in comparison["variants"].items():
        cost = candidate["cost_estimate_using_previous_model"]
        text.append(
            f"| {name} | {cost['render_seconds_per_geometry_pass']:.3f} | {cost['generation_seconds_per_accepted_if_old_qa']:.3f} | {cost['generation_with_persistence_seconds'] / 60:.1f} | {cost['durable_bytes_if_old_jpeg_sizes'] / 1e6:.1f} | 未測定 |"
        )
    text.extend(
        [
            "",
            "## 24k/48k/96k費用表の条件付き再計算",
            "",
            "前回の10k共通更新、seed43/44、代表3条件の独立40k、cold setup/full fetch、全checkpoint保持の想定を維持した。人間が合意した値ではない。hull25の候補は基準G96を仮案とし、少なくとも96層を一巡させる。基準prefixのC/R/N0はCPU実測、QAと将来prefix/高密度の率は外挿。",
            "",
            "| 候補 | 基準G/scene | 6kへ必要なGの線形推定 | 基準採択見込み |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, c in costs["variants"].items():
        text.append(
            f"| {name} | {c['baseline_trajectories_per_scene_proposal']} | {c['minimum_integer_G_for_6000_from_128_mean_if_old_qa']} | {c['baseline_estimate']['accepted_if_old_scene_qa_rates']:.0f} |"
        )
    text.extend(
        [
            "",
            "以下の24k/48k/96kは前回と同じ倍率・比較条件数の呼称で、exact枚数ではない。組合せ案は基準約6,379枚なので端点は約25.5k/51.0k/102.1kになる。QA後に補充して枚数を合わせる計画ではない。幅はpreflight固定/線形の感度ケースで、信頼区間ではない。",
            "",
            "| 候補 | 旧最大目安 | A1/A2最大cache | 主比較のみ L4時間 | 追加seed・長時間込み L4時間 | Drive増分GB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, c in costs["variants"].items():
        for b in c["budgets"]:
            text.append(
                f"| {name} | {b['previous_nominal_max_images']:,} | {b['A1_max_cache_estimate']:.0f}/{b['A2_max_cache_exact_for_base_curves']:.0f} | "
                + "〜".join(f"{v:.1f}" for v in b["main_only_l4_hold_hours"])
                + " | "
                + "〜".join(f"{v:.1f}" for v in b["with_additions_l4_hold_hours"])
                + f" | {b['drive_increment_bytes'] / 1e9:.1f} |"
            )
    text.extend(
        [
            "",
            "cache数の減少は基準G128→96という別の計画変更も含む。画角の変更自体がこの節約を保証するわけではない。前回currentの費用表は全項目を再現した。新候補の採択率・JPEG容量をL4で測った後、costs.pyの仮定を置き換え、規模・G・U・seedは親が決める。",
            "",
            "## 次の192枚と停止理由",
            "",
            "current、wide125、hull25_margin0、hull25_margin0_wide125を各scene12枚、合計192枚で凍結した。full/near_full/partialを各4枚、足りない枠は規則で再配分。候補ごとの上限ではなく、描画総数が192。D5をそのまま適用し、拒否分は補充しない。",
            "",
            "層別にraw件数・QA拒否理由・α<0.9・代表画像を報告し、全体採択率の参考推定には幾何母集団N_h/n_hを使う。バランスを変えた12枚のraw採択率を母集団率と呼ばない。全面・ほぼ全面は各層最大4枚、部分は不足枠の再配分で増えるため、本番QA率の確定には精度が低い。",
            "",
            "比較scriptはこの新規commitに含まれ、現在のpush済み版には無い。依頼の停止条件に従い、Colab開始前で停止した。今回の新規描画0・学習0・Colab新規session0。親がpush後、[resume.md](../resume.md)で一意L4を開始し、全jobのDrive保存確認後に必ずstopする。採用D2・最終費用・本実験条件は未確定。",
            "",
            "推奨はD2の全面不足を制約として放置せず、まず上記192枚のQA・画質を確認すること。25%/margin0/92.5–125°は描画比較の第一候補だが採用案ではない。valに近い高く遠いcameraを検討する場合、strategyの高さや軌道設計の変更が必要になり得るため、別途親の判断へ返す。",
            "",
        ]
    )
    (args.output / "comparison.md").write_text("\n".join(text))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current_rows = json.loads(
        (args.analysis / "baseline-geometry-rows.json").read_text()
    )
    factor_bins: dict[str, Any] = {}
    for field, edges in (
        ("height_m", [0, 2, 2.5, 4]),
        ("distance_xy_m", [0, 5, 10, 15, 25]),
        ("hull_clearance_m", [0, 2, 5, 10, 25]),
    ):
        factor_bins[field] = {}
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            selected = [r for r in current_rows if low <= r[field] < high]
            factor_bins[field][f"[{low},{high})"] = {
                "n": len(selected),
                "framing": {
                    g: sum(r["framing"] == g for r in selected) for g in GROUPS
                },
                "with_kp_behind_camera": sum(r["behind_camera"] for r in selected),
            }
    write(
        args.output / "factor-bins.json",
        {
            "scope": "all 1209 original candidates; marginal associations, not isolated causal effects",
            "factors": factor_bins,
        },
    )
    val_rows = json.loads((args.analysis / "val-camera-rows.json").read_text())
    captures = json.loads((args.analysis / "capture-rows.json").read_text())
    capture_summary = {
        s: {
            f: stats([r[f] for r in captures if r["scene"] == s])
            for f in ("height_m", "distance_xy_m")
        }
        for s in SCENES
    }
    write(args.output / "captured-camera-summary.json", capture_summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].scatter(
        [r["distance_xy_m"] for r in current_rows],
        [r["height_m"] for r in current_rows],
        s=4,
        alpha=0.2,
        label="current D2, 1209 planned",
    )
    axes[0].scatter(
        [r["distance_xy_m"] for r in captures],
        [r["height_m"] for r in captures],
        s=3,
        alpha=0.15,
        label="captured cameras (nearest court)",
    )
    axes[0].scatter(
        [r["distance_xy_m"] for r in val_rows],
        [r["height_m"] for r in val_rows],
        s=7,
        alpha=0.6,
        label="val estimates, one frame/video",
    )
    axes[0].set(
        xlabel="Distance to court centre (m)",
        ylabel="Height above court (m)",
        title="Position support",
    )
    axes[0].legend(fontsize=6)
    labels = [
        "current",
        "wide125",
        "hull5",
        "hull25\nmargin0",
        "hull25\n+wide125",
        "val",
    ]
    distributions = [
        c["geometry_pass"]["framing"] for c in comparison["variants"].values()
    ] + [val["framing"]]
    bottom = np.zeros(len(distributions))
    for g, color in zip(GROUPS, ("#217346", "#9bc36b", "#c8cdd5"), strict=True):
        values = np.array([d[g] / sum(d.values()) * 100 for d in distributions])
        axes[1].bar(
            range(len(distributions)), values, bottom=bottom, label=g, color=color
        )
        bottom += values
    axes[1].set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylabel="Percent",
        title="Framing after geometric gates",
    )
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].legend(fontsize=7)
    x = np.arange(4)
    for shift, name, color in (
        (-0.2, "current", "#497cae"),
        (0.2, "hull25_margin0_wide125", "#217346"),
    ):
        c = comparison["variants"][name]
        full = [
            c["geometry_pass_by_scene"][s]["framing"]["full"]
            / c["geometry_pass_by_scene"][s]["n"]
            * 100
            for s in SCENES
        ]
        axes[2].bar(x + shift, full, width=0.4, label=name, color=color)
    axes[2].set(
        xticks=x,
        xticklabels=SCENES,
        ylabel="Full court (%)",
        title="Residual scene imbalance",
    )
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(args.output / "support.png", dpi=130)
    plt.close(fig)
    assert all(
        p.stat().st_size <= (512 * 1024 if p.suffix == ".png" else 256 * 1024)
        for p in args.output.iterdir()
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(list(args.output.iterdir())),
                "source_test_geometry_processed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
