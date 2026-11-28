# 委任指示 (#2697): ボール検出の単フレーム(T=1) vs 時系列(T=8・T=32) 統制学習アブレーション

親 Issue #2627（テニスシーン3D再構成の技術報告）の子 Issue #2697 として、第7章の根拠となる単フレーム (`T=1`) と時系列 (`T=8`, `T=32`) をゼロから同一条件で学習する統制アブレーションを実施する（`#2637` の推論時入力介入を補完する）。

## 必須の前提・制約
- `AGENTS.md`、`.agents/memory/SUMMARY.md`、`knowledge/nodes/ball_detection/000098-run-i2637-ball-temporal-context.md`、`knowledge/nodes/ball_detection/000068-run-ball-sequential-length-20261025.md`、`knowledge/runs/run-ball-sequential-length-20261025/run.py` を読んで既存の学習・評価契約を確認する。
- **常設 `src/` は変更しない**。一度限りの学習・評価・集計・描画スクリプトと軽量な検証テストはすべて `knowledge/runs/run-i2697-ball-temporal-training/` に同梱し、**記録専用 PR** として作成する。
- GPU 実行は必ずメイン checkout 共有の training queue（`TRAINING_QUEUE_DIR=[local path]`、`--resource all`、`--provider codex`、`--session "$CODEX_THREAD_ID"`、`--issue 2697`）経由で投入する。queue worker の `start`[local path]`serve` は禁止（既に systemd サービスで稼働中）。
- GPU ジョブを投入したら、このターンで待たずに `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":[...]}}` を返して終了する。プローブ用と本学習用で複数ターンに分けてもよい（合計 GPU 時間は **6時間以内** を目安にする）。
- 最終的に ready の記録専用 PR を push したら、PR CI 完了を待たずに PR 番号・完全な HEAD・検証結果を報告し `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す（PR 本文には `## 実行証拠` JSON と `Closes #2697` を必ず記載する）。

## 実験設計（固定条件）
- **データセット**: ローカル保存済みの固定データセット `ball-mix1800-720p-all-frame-train-val-v1`（`[local path]` 配下の `dataset/` および `prepared/{train_manifest.json, evaluation_manifest.json}`）。入力の SHA-256 とパック数・評価フレーム数を照合して固定する。
- **比較する3条件**（同一アーキテクチャ系統・同一解像度・同一事前学習初期化・同一 seed `42`・同一オプティマイザ[local path]）:
  1. `T1_rgb`: 単フレーム入力（`frames=1`、RGB 3ch、時間文脈なし・MDD なし）。単フレーム専用に最初から学習する。
  2. `T8_rgb_mdd`: 短時系列入力（`frames=8`、5ch RGB+MDD）。
  3. `T32_rgb_mdd`: 長時系列入力（`frames=32`、5ch RGB+MDD）。
  - 既存の `mdd_stage_dpt_heatmap`（X3D/CSN + DPT）等、`T=1 / 8 / 32` を同一骨格（`T=1` は入力層のみ 3ch・時間長 1、それ以外のチャネル幅・DPT デコーダ・ヘッドは完全同一）で公平に比較できる構成を選ぶ。もし解像度 720p の 3条件で 6時間に収まらない場合は、全3条件共通の解像度（例: 512×288 または 720p）と共通の `frames_per_update`（勾配累積で `T=1/8/32` 全条件の 1更新あたりフレーム数と総最適化更新数を完全一致させる）を事前に固定する。
  - **重要**: `T=1` と `T=8`[local path]`T=32` で 1更新あたりの観測フレーム総数（例: `frames_per_update=64` または `128`）および総最適化更新数（例: 3,000〜6,000更新）を**厳密に揃え**、データ量や更新回数の差が交絡しないようにする。まず短い GPU プローブ（数分）または既存計測で 1更新あたり秒数と VRAM 余裕（16GB RTX 5060 Ti で安全域内）を確認し、全3条件の学習＋全 validation 評価が合計 6時間以内に収まる更新数を確定して子 issue に実験設計コメントを投稿してから本学習を投入すること。

## 評価・成果物
1. **全固定 validation 評価**（observed 85,473 + reviewed_empty 29,173 + out_of_frame 2,712 = 117,358 スコア対象フレーム、および source 別: Chat / Meiji 3cam / TrackNet game9）:
   - 位置誤差（元画像 px）: mean / median / P95 / 20px 以内率（存在閾値適用前の全 observed 85,473 件）。
   - 存在判定: 最適バランス閾値および固定 0.5 閾値での FPR（確認済み不在 31,885 件分母）・FNR（observed 85,473 件分母）・F1。
2. **#2637 固定 144 アンカー部分集合での評価**:
   - `knowledge/runs/run-i2637-ball-temporal-context/anchors.json` の同一 144 アンカー（observed 88 / reviewed_empty 32 / out_of_frame 24）でも3条件を採点し、`#2637` の推論時反復・形状基準と直接並べられる表を作る。
3. **論文掲載用の図 (`temporal_training_ablation.png`, `temporal_training_ablation.pdf`)**:
   - 保存済み `metrics.json` / `comparison.csv` から再現生成できるスクリプトを同梱する。
   - (a) T=1 / T=8 / T=32 の位置精度（20px 以内率・mean/median 誤差）と存在判定（F1 / FPR / FNR）、(b) 距離誤差 CDF または source 別比較を日本語（Noto Sans CJK JP / DejaVu Sans）で明瞭に描画する。
4. **ナレッジ記録と検証**:
   - 結果が主張を支持するか否かにかかわらず、観測値と条件をそのまま記録し最終報告で親へ伝える。
   - `knowledge-control` skill に従い `knowledge/nodes/ball_detection/` に run/group ノードを作成し、`knowledge/summary.md` を更新する。
   - 集計・図生成スクリプトの再実行バイト一致、同梱チェック、`ruff`、`pre-commit`、`kg_validate.py --check-summary`（現 `main` 既存の `deployments.yaml` 2件以外のエラー・警告が 0 件であること）を確認する。
