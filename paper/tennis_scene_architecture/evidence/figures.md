# 図の来歴と再生成

元ファイルと掲載ファイルは同じバイト列です。SHA-256欄は双方の値を示します。パスは、元がリポジトリroot基準、コピー先が原稿ディレクトリ基準です。`scripts/prepare_figures.py` が各固定コミットから抽出し、コピー前後を照合します。

| コピー先 | 元ファイル | 元コミット | 元・コピー先の SHA-256 |
| --- | --- | --- | --- |
| `figures/court/method_ransac.png` | `paper/court_robustness/figures/method_ransac.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `6db91889d3969b9c3765073bb8759c34a23ab441e948a772a006b403325576f2` |
| `figures/court/method_projection.png` | `paper/court_robustness/figures/method_projection.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `b1bb9b29c756b7736572244c8b976e2cc020d4b797d53f96ff96b05623d2f5dc` |
| `figures/court/sfm_temporal_drift.png` | `paper/court_robustness/figures/sfm_temporal_drift.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `2246b4a1c7a11f136504d1751186fa901d4770a56cfcabfbfdc3daf684066711` |
| `figures/court/B00_000423_alignment.png` | `paper/court_robustness/figures/B00_000423_alignment.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `c5bda24fbbe0148f4fb1de510ddb90864bcc7ee31e73e60c2414d2ba2cd434e0` |
| `figures/court/B01_001403_alignment.png` | `paper/court_robustness/figures/B01_001403_alignment.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `a263d1dd84c820c97d27ccbacb3d63270c6a2c1ed2d2bfdecebb2928aa04591a` |
| `figures/court/B02_000445_alignment.png` | `paper/court_robustness/figures/B02_000445_alignment.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `853d761fcfc447a3803b40a1dcf6900cd6b5886af50e39d228bf1e66ff971388` |
| `figures/court/B03_001091_alignment.png` | `paper/court_robustness/figures/B03_001091_alignment.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `28f939ac4889967c4ac88314acb686ec6a5555c4b09b9edf16c5048d0cc121ba` |
| `figures/court/local01.jpg` | `paper/court_robustness/images/local01.jpg` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `7d4fc062c2ee1480b04d77c5ed79e79d0983446c80119063aa9436d6e6c2f016` |
| `figures/court/local01_baseline_common.png` | `paper/court_robustness/figures/local01_baseline_common.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `5153e9314532998d5452adc97bb52ae8f769fa13332872765a1261cbb3c29ec6` |
| `figures/court/local01_ours.png` | `paper/court_robustness/figures/local01_ours.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `a6df40a18205639fb7e55af3b75b26e93ccb184aa8c0d4b78f2d0d6dce83299e` |
| `figures/court/local02.jpg` | `paper/court_robustness/images/local02.jpg` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `e76a1b63ff1e6aef6db63c5ea2f237ce7afd3de4be723131859aace12457e31f` |
| `figures/court/local02_baseline_common.png` | `paper/court_robustness/figures/local02_baseline_common.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `0af90c486bc1bdda563792fffaafc5c73af5840ce90715c4c75c92280e5ba6c2` |
| `figures/court/local02_ours.png` | `paper/court_robustness/figures/local02_ours.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `d67f19e78f98c8ef122d355abfaef869c877444b4bbd6584290fadc7f3a93bef` |
| `figures/court/local03.jpg` | `paper/court_robustness/images/local03.jpg` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `64aa07abf02b6ae27e1da6f2eb4d195d1cec55469db4d20a6996bf8c8e63c1fa` |
| `figures/court/local03_baseline_common.png` | `paper/court_robustness/figures/local03_baseline_common.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `faed81482710f0b0d25d311ea7fd19e1b130215ea6525857d4d14e19c10c9f62` |
| `figures/court/local03_ours.png` | `paper/court_robustness/figures/local03_ours.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `22011a2cf0cf7f2a1a8b2cc2a967998fe82a834bddf394662f8902d60639ad90` |
| `figures/court/local04.jpg` | `paper/court_robustness/images/local04.jpg` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `fff92432ea6e17c83c6c1e7d1dbb8917c9421a6b0408b17b3848122fbf93610b` |
| `figures/court/local04_baseline_common.png` | `paper/court_robustness/figures/local04_baseline_common.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `cd9f270b255290dfb29d4673ea9fb98275a6f138e6a7e603153878b74bec7bd3` |
| `figures/court/local04_ours.png` | `paper/court_robustness/figures/local04_ours.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `081ee40de8d1505d7b9881f4bb133263f2b58d181c0c8edc2c20fdae8ab8e464` |
| `figures/court/test-effects.png` | `knowledge/runs/run-i2633-court-viewpoint-comparison/test-effects.png` | `d292aa707ab7874119698ad6191e7d7704e0f1e6` | `86b3a59aba37d26978396f55ae7dd62a7e7a4594b3f32bb10a85f8459466818b` |
| `figures/players/clip_004-f0282.pdf` | `knowledge/runs/run-meiji-stopped-clip-observations/figures/clip_004-f0282.pdf` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `3a14fcc3c572d1c7d760ce17609c25b46c56876ea32cce56caba4ee2b83112e2` |
| `figures/players/clip_020-f0131.pdf` | `knowledge/runs/run-meiji-stopped-clip-observations/figures/clip_020-f0131.pdf` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `2ee7ffe39a4f9c267fde5325c45c28ff10b92a8a386af54ab84d7fc65dbcb649` |
| `figures/ball/footprint-distributions.png` | `knowledge/runs/run-i2637-ball-image-difficulty/footprint-distributions.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `27fcf320ca2ccc78e03a304d951e826f99d7f865e4a3817e24435c0987401eab` |
| `figures/ball/visibility.png` | `knowledge/runs/run-i2637-ball-image-difficulty/visibility.png` | `a1a9931aadd533fda757968aa10249aae12aaeb6` | `806e14b7c6aeb6069ec8ee109a8af3f50a8079bb7801dfbc6e9af0365fca51cb` |
| `figures/ball/temporal_training_ablation.pdf` | `knowledge/runs/run-i2697-ball-temporal-training/temporal_training_ablation.pdf` | `3c0083802aa118d46cd739109d3ba250004c182f` | `d7429265fbc43dd1f721022cc1e284a3e9ffba51dfa5225c5a70a934950161e0` |
| `figures/execution/comparison.pdf` | `knowledge/runs/run-i2696-exec-system-ablation/comparison.pdf` | `f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1` | `4b6d11ae0c5f7c7b70e77d832b8ff02456ee880ffc2248cfbd4337e10b400881` |

## 元図の生成方法

| 掲載図 | 固定証拠と生成方法 |
| --- | --- |
| `figures/pipeline.pdf`・`figures/execution.pdf` | `scripts/{pipeline,execution}.tex` と `scripts/diagram-style.tex` によるTikZ模式図。リポジトリrootで `bash paper/tennis_scene_architecture/scripts/make_diagrams.sh` |
| コートの地面推定・LINE投影 | `paper/court_robustness/evidence/alignment_method/` の実SfM点群と実画像LINE。元報告の `.venv/bin/python paper/court_robustness/make_alignment_figures.py` |
| コートのアライメント | `paper/court_robustness/evidence/scene_sources/` の保存レンダリング・カメラ・教師。元報告の `.venv/bin/python paper/court_robustness/make_scene_figures.py` |
| コートの実画像比較 | 同ディレクトリの `images/` と `evidence/predictions/`、`evidence/homography/results.json`。元報告の `.venv/bin/python paper/court_robustness/make_comparisons.py` |
| SfMドリフト | 同ディレクトリの `evidence/sfm_drift/` の点・観測トラック。元報告の `.venv/bin/python paper/court_robustness/make_drift_figure.py` |
| `test-effects.png` | #2633の固定キャンペーンの採点行を `report.py --campaign <campaign> --output <out>` で集計。 `metrics.csv` と `comparison.json` に差と条件を保存。[#2633の再現手順](https://github.com/Motoki0705/tennis-lab/blob/d292aa707ab7874119698ad6191e7d7704e0f1e6/knowledge/runs/run-i2633-court-viewpoint-comparison/README.md) |
| 停止診断PDF | #2636の保存画像、選手枠ラベル、検出・追跡・選別。 `figures.py` が重畳・レイアウトを生成。[#2636の再現手順](../../../knowledge/runs/run-meiji-stopped-clip-observations/README.md#再現) |
| `footprint-distributions.png`・`visibility.png` | #2637の `measure.py` の保存計測と `visual-review.csv` を `summarize.py --input <size> --review <csv> --output <out>` で集計。[000097の再現手順](../../../knowledge/nodes/ball_detection/000097-run-i2637-ball-image-difficulty.md#実行記録) |
| `temporal_training_ablation.pdf` | #2697の保存予測の集計。 `reproduce.sh collect --training <run> --output <out>` で集計し、`reproduce.sh plot --table <comparison.csv> --output <out>` で描画する。[#2697の再現手順](https://github.com/Motoki0705/tennis-lab/blob/3c0083802aa118d46cd739109d3ba250004c182f/knowledge/runs/run-i2697-ball-temporal-training/README.md) |
| `comparison.pdf` | #2696の `metrics.json` を `make_figures.py <metrics.json> <out>` で描画。[#2696の再現手順](https://github.com/Motoki0705/tennis-lab/blob/f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1/knowledge/runs/run-i2696-exec-system-ablation/README.md#再現) |

TikZ図は同じ組版環境で再生成し、バイト一致を確認しました。

| 生成PDF | SHA-256 |
| --- | --- |
| `figures/pipeline.pdf` | `e2bebffac390a3825caf0d02985998779d1c5663712a87ba726b9f3dc5958d16` |
| `figures/execution.pdf` | `b26822b4fd04e9bff19f5eb4c1215f94727fd7112332f2ba0046bce1f9c486b5` |
