# 3DGSによる視点拡張と幾何教師を用いたテニスコート検出

[report.pdf](report.pdf) は日本語・A4縦・8ページの技術報告です。手法、実験条件、結果、課題点の正本は [report.tex](report.tex)。指定された `data/samples/tennis_court` の**全4枚**で旧資料の外部写真6枚を置き換え、B00〜B03各3視点の実3DGSレンダリング＋アライメントを収録しています。方法の実例としてB00の点群によるRANSAC地面推定、視点ごとのLINE投影から集約までを掲載しています。第6節「課題点」では実SfMトラックによる地面高さの時間依存の不整合と、合成側4シーンの多様性・汎化の制約を扱います。模式図や生成AI画像は使用していません。

## 同梱物

| ファイル | 用途 |
|---|---|
| `images/`, `evidence/inputs.json` | 指定画像のバイト一致コピー、元ファイル名・サイズ・SHA-256 |
| `evidence/predictions/` | 両モデルのKP座標、H生成可否、TCD確率マップ・argmax、提案モデルのLINE確率・姿勢生出力。数値はfloat32/float64のまま保存 |
| `evidence/homography/`, `tables/homography.tex` | 保存KP・スコア・LINEからの共同推定。参照テンプレート、3段階のHと線支持率、KP採用履歴・除外理由、残差、コード・元推論のハッシュ |
| `evidence/inference_both.json`, `checkpoint_config.json`, `target_bundle.json` | 重み・入力・出力のハッシュ、CPU実行、保存設定と出力契約 |
| `evidence/training_run*.{json,yaml}`, `training_provenance.json` | mixed学習の保存設定と既存queue runの出典。checkpointの正規化設定から落ちる混合比を補う |
| `evidence/dataset_audit.json` | TCD全8,841画像＋B00〜B03全8,415レンダリングとのバイト/RGB/知覚ハッシュ照合 |
| `evidence/scene_sources/` | 実レンダリングPNG、選択サンプルのカメラ・教師、現存アライメントと分割集計。各sourceのハッシュ付き |
| `evidence/alignment_method/`, `alignment_figures.json` | 元のSfM点群、保存平面・設定、48視点の投影証拠、掲載3視点のRGBとLINE生出力、32視点の集約。RANSAC・射影の再計算と元データ照合に使用 |
| `evidence/sfm_drift/`, `drift_figures.json`, `tables/sfm_drift.tex` | B00〜B03の全点に対応するSfMトラックの観測区間・再投影誤差・登録画像ID、前半／後半の共通セルの高さ差、格子幅の感度確認、4区間診断の成立状況。図と表を同じ数値から生成。元COLMAPモデル・座標変換のハッシュ付き。B00点群は方法図バンドルを共有し、他3シーンは同梱 |
| `figures/`, `evidence/*figures.json`, `scene_visualization.json` | PDF図版と数値・実画像からの生成結果のハッシュ |
| `evidence/build.json` | PDF・TeX・図版・生成表のハッシュとLuaLaTeXの組版検証記録 |
| `evidence/validation.json` | PDFと掲載成果物の検証結果。精度評価ではない |

同梱物だけの検証は、git管理されたビルド記録を読み、一時的なLaTeXログを要求しません。通常は読み取り専用で、`--write-report` を付けた場合だけ `validation.json` を更新します。`--check-local-sources` は両モデルの重み、checkpoint内の設定・出力契約・epoch/stepも現物と照合します。

現物照合の状態は [evidence/checks.json](evidence/checks.json) に記録しています。`local_source_validation` に残す2026-09-19の確認では、以前の未学習推論に使用したcheckpointの現物が保存SHA-256と一致せず、全現物照合は失敗しました。今回の全4シーンのSfM元データ照合は `four_scene_source_validation` に分けています。既存の推論結果・重みの記録は保持しており、元のバイト列が必要な再推論ではこの不一致を解消する必要があります。

方法図はB00の保存済みv2 heatmap archiveを明示的に読みます。当時の入力は校正済み実画像で、現在のNHTレンダリング入力cacheへの差し替えは行いません。全48視点の確率配列と入力画像ハッシュが当時のcacheと一致することを確認しています。LINE専用のepoch 19重みと、未学習写真に使う多タスクepoch 17重みの出典も分けて保存しています。

使わないSEG・全KPヒートマップ（本モデル）は保存対象から除き、図版に必要な生出力を保持しています。重み・動画・3DGS全体は大きいため同梱しません。写真の撮影者・原公開URL・ライセンスは提供されていないため、出典を推測せずユーザー指定画像と記録しています。画像の権利を本repoのMITへ変更するものではありません。

## PDFを再ビルド

リポジトリrootで実行します。LuaLaTeX、luatexja、Noto Serif/Sans CJK JP、DejaVu Serif/Sansを使用し、2回の組版と欠け・文字・参照の検証後にビルド記録を保存します。

```bash
.venv/bin/python paper/court_robustness/build_paper.py
```

## 同梱証拠から図版を再生成・検証（GPU・重み不要）

リポジトリrootで、プロジェクトの `.venv` を使用します。NumPy、OpenCV、Pillow、Matplotlib、PyTorch（CPU）・SciPyを含むプロジェクト依存関係、pytest、およびPDF検証用の `pdfinfo` / `pdftotext` が必要です。

```bash
.venv/bin/python paper/court_robustness/homography_evidence.py
.venv/bin/python paper/court_robustness/make_comparisons.py
.venv/bin/python paper/court_robustness/make_scene_figures.py
.venv/bin/python paper/court_robustness/make_alignment_figures.py
.venv/bin/python paper/court_robustness/make_drift_figure.py
# 図版を変更した場合は build_paper.py でPDFとビルド記録を更新
.venv/bin/python -m pytest -n0 paper/court_robustness/test_artifacts.py
.venv/bin/python paper/court_robustness/verify_artifacts.py
```

KP・LINEのHは `src/tasks/court_detection/geometry/hybrid_homography.py` と `line_evidence.py` を共通実装として、保存済み `raw_kp`・`kp_scores`・`line_probability` からCPUで再推定します。元の推論NPZ・重みSHAは保持し、3段階のHと診断を別の証拠へ保存します。`infer.py` で今後推論する場合も同じ推定を呼びます。OpenCV・NumPy・SciPyのバージョン、推定コード、入力、設定が記録と異なると検証は停止します。方法と閾値の正本は本文3.4・4.2です。互換フィールド `fit_inliers` と `inliers` はともに最後の共同最適化に使ったKP集合で、誤差閾値内の全点を意味しません。`kp_rejection_reasons` は幾何外れ値・LINE支持不足・点数上限などを区別します。旧PROSACは比較段階 `kp_only` に保存し、TCD公式処理は保持しています。

3DGS図は保存レンダリングの原寸RGBに、保存カメラから全コートを再投影します。キャプチャ画像への置換はありません。画像の切り抜き・補修は行わず、線分だけを画面/near planeでclipします。`--collect` なしではデータセットも不要です。テストは、カメラやアライメントを誤って組み合わせた場合の拒否、near planeを跨ぐ線分、全図版の画素一致を確認します。

方法図のRANSACは元の全点群からproduction関数で再計算し、保存平面への一致を要求します。投影もproduction関数で再計算します。共通グリッドへの集約は当時のreducerを数値的に再現し、保留視点の混入を拒否します。元データから方法図の証拠を再収集する場合だけ、`make_alignment_figures.py --collect` を使用してください。未学習写真のLINEは、同一の保存確率から二値マスク・RGB重畳・確率マップを生成します。

SfM診断は `drift_evidence.py` が公開COLMAP binary形式の点・観測トラックを読み、公開 `scene_from_sfm` で既存点群へ対応付けます。点の全単射・座標許容差・RGB一致を要求し、ファイル内の並びや連番IDには依存しません。登録画像の欠番を保持して撮影順の順位で分割し、地面基準は全シーンで保存コートの共通平面から構成します。コート間の非共面性は拒否します。全シーン共通の2区間診断を本文に掲載し、4区間で共通セルがない場合は0誤差にせず `no_shared_cells` / `null` と記録します。

`make_drift_figure.py --collect` の場合だけ元SfMモデルが必要です。`--check-local-sources` は全4シーンのモデルを再読してトラックの全件一致も確認します。通常の再生成は同梱証拠のみで行い、区間をまたぐ点の除外、同じセル同士の比較、欠番を持つ区間分割、図の画素一致と表の数値をテストします。測定条件・解釈・交絡要因と、SfM／SLAMのドリフト研究との関係は本文第2・6節を参照してください。

## 指定入力から推論を再実行

専用worktree内で実行します。`data` と `.venv` は元repoのものを使用できます。重みとDINOv3の外部資産はgit共通rootの既存ファイルを参照します（具体的な重みは `inference_both.json`）。`infer.py` はCUDAを不可視にしてCPU実行し、終了時にCUDA未初期化を検証します。

```bash
mkdir -p .cache/court-report
git clone https://github.com/yastrebksv/TennisCourtDetector.git .cache/court-report/TennisCourtDetector
git -C .cache/court-report/TennisCourtDetector checkout e5cd4f1ce26b15361700d3d89e068cbf0e82749e
.venv/bin/python -m gdown 'https://drive.google.com/uc?id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG' -O .cache/court-report/baseline.pth
.venv/bin/python paper/court_robustness/prepare_inputs.py
.venv/bin/python paper/court_robustness/infer.py
.venv/bin/python paper/court_robustness/audit_dataset.py
.venv/bin/python paper/court_robustness/make_scene_figures.py --collect
.venv/bin/python paper/court_robustness/make_alignment_figures.py --collect
.venv/bin/python paper/court_robustness/make_drift_figure.py --collect
.venv/bin/python paper/court_robustness/homography_evidence.py
.venv/bin/python paper/court_robustness/make_comparisons.py
.venv/bin/python paper/court_robustness/build_paper.py
# 元画像・カメラ・アライメント・重み・保存設定まで照合
.venv/bin/python paper/court_robustness/verify_artifacts.py --check-local-sources --write-report
```

互換処理は推論専用です。現行validatorが保存済みの多解像度学習設定を拒否するため、使わない学習用 `train_scales` のみを `[val_short_side]` に置換し、全state dictをstrictに読み込みます。元設定も保存し、推論解像度・architecture・重みは変更しません。重みはSHA-256を固定し、読み込み前後と推論終了時に一致を要求します。TCDはNumPy/SciPy互換のため12候補のH選択を等価に記述し、特異行列・評価点欠落を明示的に棄却します。

新規学習・GPUレンダリングは不要です。今後GPU処理を追加する場合は、repoのtraining-queue規約に従ってください。
