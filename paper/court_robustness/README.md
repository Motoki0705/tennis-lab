# 3DGSによる視点拡張と幾何教師を用いたテニスコート検出

[report.pdf](report.pdf) は日本語・A4縦・5ページの技術報告です。手法、実験条件、結果、限界の正本は [report.tex](report.tex)。指定された `data/samples/tennis_court` の**全4枚**で旧資料の外部写真6枚を置き換え、B00〜B03各3視点の実3DGSレンダリング＋アライメントを収録しています。模式図や生成AI画像は使用していません。

## 同梱物

| ファイル | 用途 |
|---|---|
| `images/`, `evidence/inputs.json` | 指定画像のバイト一致コピー、元ファイル名・サイズ・SHA-256 |
| `evidence/predictions/` | 両モデルのKP座標、H生成可否、TCD確率マップ・argmax、提案モデルのLINE確率・姿勢生出力。数値はfloat32/float64のまま保存 |
| `evidence/inference_both.json`, `checkpoint_config.json`, `target_bundle.json` | 重み・入力・出力のハッシュ、CPU実行、保存設定と出力契約 |
| `evidence/training_run*.{json,yaml}`, `training_provenance.json` | mixed学習の保存設定と既存queue runの出典。checkpointの正規化設定から落ちる混合比を補う |
| `evidence/dataset_audit.json` | TCD全8,841画像＋B00〜B03全8,415レンダリングとのバイト/RGB/知覚ハッシュ照合 |
| `evidence/scene_sources/` | 実レンダリングPNG、選択サンプルのカメラ・教師、現存アライメントと分割集計。各sourceのハッシュ付き |
| `figures/`, `evidence/*figures.json`, `scene_visualization.json` | PDF図版と数値・実画像からの生成結果のハッシュ |
| `evidence/build.json` | PDF・TeX・図版のハッシュとLuaLaTeXの組版検証記録 |
| `evidence/validation.json` | PDFと掲載成果物の検証結果。精度評価ではない |

同梱物だけの検証は、git管理されたビルド記録を読み、一時的なLaTeXログを要求しません。通常は読み取り専用で、`--write-report` を付けた場合だけ `validation.json` を更新します。`--check-local-sources` は両モデルの重み、checkpoint内の設定・出力契約・epoch/stepも現物と照合します。

使わないSEG・全KPヒートマップ（本モデル）は保存対象から除き、図版に必要な生出力を保持しています。重み・動画・3DGS全体は大きいため同梱しません。写真の撮影者・原公開URL・ライセンスは提供されていないため、出典を推測せずユーザー指定画像と記録しています。画像の権利を本repoのMITへ変更するものではありません。

## PDFを再ビルド

リポジトリrootで実行します。LuaLaTeX、luatexja、Noto Serif/Sans CJK JP、DejaVu Serif/Sansを使用し、2回の組版と欠け・文字・参照の検証後にビルド記録を保存します。

```bash
.venv/bin/python paper/court_robustness/build_paper.py
```

## 同梱証拠から図版を再生成・検証（GPU・重み不要）

リポジトリrootで、プロジェクトの `.venv` を使用します。NumPy、OpenCV、Pillow、pytest、およびPDF検証用の `pdfinfo` / `pdftotext` が必要です。

```bash
.venv/bin/python paper/court_robustness/make_comparisons.py
.venv/bin/python paper/court_robustness/make_scene_figures.py
# 図版を変更した場合は build_paper.py でPDFとビルド記録を更新
.venv/bin/python -m pytest -n0 paper/court_robustness/test_artifacts.py
.venv/bin/python paper/court_robustness/verify_artifacts.py
```

3DGS図は保存レンダリングの原寸RGBに、保存カメラから全コートを再投影します。キャプチャ画像への置換はありません。画像の切り抜き・補修は行わず、線分だけを画面/near planeでclipします。`--collect` なしではデータセットも不要です。テストは、カメラやアライメントを誤って組み合わせた場合の拒否、near planeを跨ぐ線分、全図版の画素一致を確認します。

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
.venv/bin/python paper/court_robustness/make_comparisons.py
.venv/bin/python paper/court_robustness/build_paper.py
# 元画像・カメラ・アライメント・重み・保存設定まで照合
.venv/bin/python paper/court_robustness/verify_artifacts.py --check-local-sources --write-report
```

互換処理は推論専用です。現行validatorが保存済みの多解像度学習設定を拒否するため、使わない学習用 `train_scales` のみを `[val_short_side]` に置換し、全state dictをstrictに読み込みます。元設定も保存し、推論解像度・architecture・重みは変更しません。重みはSHA-256を固定し、読み込み前後と推論終了時に一致を要求します。TCDはNumPy/SciPy互換のため12候補のH選択を等価に記述し、特異行列・評価点欠落を明示的に棄却します。

新規学習・GPUレンダリングは不要です。今後GPU処理を追加する場合は、repoのtraining-queue規約に従ってください。
