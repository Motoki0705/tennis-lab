# 委任指示 (#2696): 実行システムのモデル共有・横断バッチ・TensorRT FP16 統制アブレーション

親 Issue #2627（テニスシーン3D再構成の技術報告）の子 Issue #2696 として、第8–9章の根拠となる同一 HEAD・同一固定3クリップ上での実行システム統制アブレーション（モデル共有・横断バッチ・TensorRT FP16）と出力整合照合を実施する。

## 必須の前提・制約
- `AGENTS.md`、`.agents/memory/SUMMARY.md`、`src/tennis_scene/pipeline/README.md`、`knowledge/runs/run-i2363-trt-player-video/README.md`、`knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md` を読んで仕様を確認する。
- **常設 `src/` は変更しない**。一度限りの実験・集計・描画スクリプトと軽量な検証テストはすべて `knowledge/runs/run-i2696-exec-system-ablation/` に同梱し、**記録専用 PR** として作成する。
- GPU 実行は必ずメイン checkout 共有の training queue（`TRAINING_QUEUE_DIR=[local path]`、`--resource all`、`--provider codex`、`--session "$CODEX_THREAD_ID"`、`--issue 2696`）経由で投入する。queue worker の `start`[local path]`serve` は禁止（既に systemd サービスで稼働中）。
- GPU ジョブを投入したら、このターンで待たずに `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":[...]}}` を返して終了する。結果を受け取った再開ターンで集計・図生成・検証・PR 作成まで完了する。
- 最終的に ready の記録専用 PR を push したら、PR CI 完了を待たずに PR 番号・完全な HEAD・検証結果を報告し `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す（PR 本文には `## 実行証拠` JSON と `Closes #2696` を必ず記載する）。

## 実験設計（固定条件）
- **入力**: `meiji_4cam/video_000` の固定3クリップ `clip_000`・`clip_037`・`clip_027`（356・292・235 frame、計 883 source frame/camera × 4 カメラ = 3,532 フレーム）。
- **固定共有コート**: `#2363` と同一の保存済み共有コート（`artifact_digest=4ac8e386763687fcb6c81f46abf0774e38fc118af4f79b31cdb6e3c2283e3f75`、`drift_digest=7a4be8bf20397552a7b287131c3cb0e5fa57ba0dedc16e4a05fa23287c81a7f6`、既存保管元 `[local path]` を参照または worktree 出力先へコピーして SHA-256 照合）。
- **共通パイプライン設定**: `device=cuda:0`、`players_per_side=2`、`court_side.mode=clip`、`court_side.evidence=player`、`ball_detection.enabled=false`、`ball_reconstruction.enabled=false`、`record_cuda=true`、`record_cuda_synchronize=false`（`#2363` 全件計測と揃えたホスト計時）、`record_sample_interval=0.5`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`。各条件は独立した空の component store / 出力ディレクトリで実行し、キャッシュ再利用を一切行わない。
- **比較する4条件**（すべて同一 HEAD・同一入力・同一設定）:
  1. `C1_serial_pt16`: 1クリップずつ独立プロセスで逐次実行（`clip_000` → `clip_037` → `clip_027` の3回別実行、`batching.enabled=false`、PyTorch FP16）。モデル共有なし・横断バッチなしの基準。
  2. `C2_shared_pt16`: 3クリップを単一 `run_video_batch` 実行、`batching.enabled=false`、PyTorch FP16。**モデル共有の単独効果**を分離。
  3. `C3_batched_pt16`: 3クリップを単一 `run_video_batch` 実行、`batching.enabled=true`（`#2363` と同じバッチ上限: `vitpose=8, hmr2=16, clip=16, dino=1, gvhmr=4`）、PyTorch FP16。**横断バッチの単独効果**を分離。
  4. `C4_batched_trt16`: 3クリップを単一 `run_video_batch` 実行、`batching.enabled=true`（同バッチ上限）、**TensorRT FP16**。既存 `#2363` の検証済み plan（`[local path]`）が現在のコード・重み・manifest SHA-256 と整合すれば再利用し、不整合なら同一条件でビルドする（ビルド時間は推論 wall 時間とは分離して記録）。**TensorRT FP16 の追加効果**を分離。
  - 注意: 現在の `main` の `pipeline.yaml` の既定人物モデルが `#2363`（DINO + ViTPose-H + CLIP-ReID + HMR2 + GVHMR）と異なる場合は、4条件すべてで `#2363` の標準5モデル構成を明示固定すること。
  - 計測前に設計と所要時間見積もりを子 issue コメントへ記録すること。

## 計測・照合・成果物
1. **性能指標 (`metrics.json`, `comparison.csv`)**:
   - 各条件の CLI wall 秒（`C1` は3クリップ合計および各クリップ内訳）、recorder wall 秒、`C1` 比の高速化倍率（speedup）、隣接段階ごとの短縮秒・倍率。
   - stage 別 wall 配分、モデル別 wall 区間和、管理対象モデルの load/unload 回数、`C3`[local path]`C4` のバッチ充填率と横断バッチ回数。
   - NVML Peak VRAM (MiB / GiB)、Peak RSS (MiB / GiB)、GPU 稼働区間率・時間平均使用率。
2. **出力整合 (`parity.json`, `parity.csv`)**:
   - 全4条件 × 3クリップで `scene status`、3D 関節有効フレーム数、SMPL 有効フレーム数、player ID 対応数を照合。
   - `C1_serial_pt16` を基準に、`C2`・`C3`・`C4` の3D関節座標差（median / P95 / max L2 m）と有効マスク一致率を算出。
3. **論文掲載用の図 (`comparison.png`, `comparison.pdf`)**:
   - 保存済み `metrics.json` から再現生成できるスクリプト（`make_figures.py` 等）を同梱する。
   - (a) C1〜C4 の E2E wall 時間とモデル別/段階別内訳＋高速化倍率、(b) Peak VRAM / RSS・モデル load 回数の比較を日本語（Noto Sans CJK JP / DejaVu Sans）で明瞭に描画する。
4. **ナレッジ記録と検証**:
   - `knowledge-control` skill に従い `knowledge/nodes/tennis_scene/` に run ノードを作成し、`knowledge/summary.md` を更新する。
   - `report.py` / `make_figures.py` の再実行で CSV・JSON・図がバイト一致することを確認し、同梱のユニット確認・`ruff`・`pre-commit` を通す。
   - `kg_validate.py --check-summary` を実行する（なお現 `main` には #2681 起因の `deployments.yaml` 未登録エラー2件が既存するため、**本 PR の新規ノード・run・summary 由来エラーが 0 件であること**を確認・明記する）。
