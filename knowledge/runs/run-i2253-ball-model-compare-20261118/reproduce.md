# meiji_4camのボールモデル比較

## 検証設計

同じsource media、4カメラの順序、保存済みcamera alignmentと人物を固定する。入力のSHA-256とscene indexの依存鎖を照合し、保存済みの検出を再入力するCPU試験でball_points、ball_triangulation、scene_assemblyの基準出力を確かめる。GPUでは現行モデルの生のargmax、gate後の観測、平滑化、三角測量、sceneを保存結果と比較する。各NPZで時間軸・型・欠測・座標単位を検査し、再投影残差を保存済みカメラで照合する。

既存の `test_ball_detection.py::test_centre_policy_matches_training_owners_without_score_selection`、`test_ball_evidence.py::test_trajectory_gate_does_not_erase_raw_evidence`、`test_ball_points.py::test_filled_and_rejected_points_cannot_supply_the_second_real_view`、`test_assembly.py` を再利用する。追加の実験コードはこのrunに置き、3件の実storeと全出力の検査を行う。pytestはCPUでcollect-onlyとdurationsを記録する。

## 実行手順

1. [#1752の保存済み試走](../run-i1752-shared-fp16-20261118/reproduce.md)と同じ3クリップのmedia、store、標準モデルをローカルに用意する。[選定設定](models.json)の `drive_generation` を持つ3モデルは、tennis-driveの `download --verify` でその世代の `weights-update-00006000.pt` と `run.json` を取得し、それぞれ `--root` 配下の `checkpoint` と `training_record` に置く。実験の配置はOUTPUTの `tennis_scene/evaluate/ball-model-compare-20261118`。
2. 固定した学習コードは各 `model_source_commit` の `src` から展開する。`prepare` が学習記録に列挙されたソース、重みと設定のSHA-256を照合し、拡張のないCPU厳密ロードを実行する。基準2モデルと再構成にはworktreeの環境（Torch 2.13.0+cu130）、他の3モデルには学習時の環境（Torch 2.14.1+cu130）を使う。後者の `site-packages` は #1899 の `ball-query-desktop/ball_detection/train/query_controls/desktop-resume-20261112/env/lib/python3.11/site-packages` をローカル保持先から読み込む。標準環境のパッケージを変更する必要はない。
3. クリップとモデルの来歴を固定し、保存済みの検出を用いるCPU前処理を行う。

   ```bash
   cmp_run=knowledge/runs/run-i2253-ball-model-compare-20261118
   cmp_output=outputs/tennis_scene/evaluate/ball-model-compare-20261118
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     .venv/bin/python -B -u "$cmp_run/compare.py" prepare --root "$cmp_output" --historical-site "$cmp_training_site"
   ```

4. 共有training queueの `all` 枠へ次の処理を一度登録する。既存のhealth guardでCUDA/NVMLと推論中のheartbeatを監視し、異常時はstack・GPU情報を記録して終了する。

   ```bash
   cmp_commit=$(git rev-parse HEAD)
   cmp_command="env PYTHONPATH=. PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python -B -u $cmp_run/compare.py run --root $cmp_output --expect-commit $cmp_commit"
   TRAINING_QUEUE_DIR="$cmp_shared_queue" bash .agents/skills/training-queue/scripts/training_queue.sh add "$cmp_command" \
     --name i2253-ball-model-compare --provider codex --session "$CODEX_THREAD_ID" --issue 2253 --resource all \
     --idempotency-key i2253/ball-model-compare/20261118
   ```

## 数値の意味

- 全モデルで動画の元FPS、全frameを使用する。時間窓は中心距離で所有frameを決め、同点は早い窓、末尾は最後の完全窓で覆う。T64の学習・評価はJPEG入力で、今回のMP4はOpenCVでdecode・線形resizeしたRGB uint8をモデル入力とする。RGB＋MDDの作成は復元したモデル内で行う。
- `uv_px/score/visible` は `ball_points` に入る直前の観測。3Dは同じsmoothingを通した観測から求める。queryには位置の出力だけがあるため `raw_score` はNaN。事前の存在閾値はなく、有限な座標提案の重みを1として標準trajectory gateへ入力する。queryの可視率は幾何ゲート通過率として解釈する。
- `ball_rejection_code` はPointRejectionの整数値をfloat32で保存する。3D無効点はゼロ、再投影はNaN。再投影の中央値は全有効frameのうち、そのカメラに平滑化済みの実観測と前向き投影がある組を使う。基準点との照合は `baseline-parity.json`、read-onlyで読み込んだノードは `loaded-stages.json` に保存する。
