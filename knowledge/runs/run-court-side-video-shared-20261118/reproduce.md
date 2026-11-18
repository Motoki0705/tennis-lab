# meiji_4cam のcourt sideを再現する

## 入力

対象は `data/tennis_multiview/processed/meiji_4cam/dataset` の `video_000`。固定した共有コートとdriftは #1752 の `full69-command.txt` と同じです。DATAを読み取り、今回の入力補完は担当worktreeのoutputsへ書き込みます。`summary.json` にdataset.json・モデル・校正・動画判定・単独診断のdigestを記録します。

| clip | ball points | 校正 |
|---|---|---|
| 000、027、037 | #1752のdoubles batchで保存した4camera分 | 同じstoreの共有コート校正 |
| 018 | 11月16日のX3D検証の4camera storeをE:から作業領域へ複製 | 指定の共有コート・driftから作成 |
| 残る65 clip | 共有queueでX3D/CSN+DPTの `ball_detection → ball_points` を実行 | 指定の共有コート・driftから作成 |

GPUで実行したのは不足した65×4 cameraのボール検出です。校正・平滑化と保存を含む前段のGPU予約は `all`。共有Hの再推定、人物、3D sceneの処理は対象外です。

## 既存の前段storeからCPUで再集計する

担当worktreeで次を実行します。共有artifactのrootはメインcheckoutの `.worktree-outputs` です。解析には `prefix-results.json` と完了した前段の `health/prefix.json` が必要です。

```bash
main_checkout=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
common_outputs="$main_checkout/.worktree-outputs"
run_directory="$common_outputs/court-side-video-shared/tennis_scene/evaluate/court-side-20261118"
evidence=knowledge/runs/run-court-side-video-shared-20261118
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python "$evidence/evaluate.py" --outputs "$common_outputs" --run "$run_directory"
```

スクリプトは全storeのsource・系譜とball/checkpoint/smoothingの設定を照合して `side-inputs.json` を固定します。設定から動画判定を作成・再読込し、全clipで実nodeの採否を確認します。さらに各clipで静止観測を除いた後の全frameを連結し、元のDLTで全仮説を直接採点して加重平均、support、frame数、カメラ対の件数を照合します。000/027/037の全単独仮説も以前の保存値と照合します。

`clips.csv` がclip単位の表、`standalone-rankings.csv` が各clipの全仮説、`summary.json` が動画全体の順位・件数です。入力参照と結果オブジェクトの場所は `side-inputs.json` と `summary.json` のdigestから特定できます。

## 前段の再現

前段の実装はcommit `68b3bb4a491f6a7cca014c72d790a230a6a87894` の `src/` を `$run_directory/frozen` に展開して固定しました。入力補完の実コードは `collect_prefix.py` と `guarded_prefix.py`。ヘルス監視は #1752 のcommit `9a727005fd735dbdc76417cb54d476beb81cd2b3` にある `knowledge/runs/run-i1751-runtime-gpu-a5/health_guard.py` を使用しています。実投入のコマンドとqueue IDは `queue-run.json` にあります。

元の実行環境で前段を再現する場合は新しい出力ディレクトリを使います。絶対パスは同じデータを置いた場所に合わせてください。`guarded_prefix.py` は自分の置き場所をrun directoryとし、その3階層上を共通outputs rootとします。GPUは共有training queueを通じて起動します。

```bash
run_directory="$common_outputs/court-side-video-shared/tennis_scene/evaluate/court-side-2241-reproduction"
mkdir -p "$run_directory/frozen" "$run_directory/prefix"
git archive 68b3bb4a491f6a7cca014c72d790a230a6a87894 src | tar -x -C "$run_directory/frozen"
git show 9a727005fd735dbdc76417cb54d476beb81cd2b3:knowledge/runs/run-i1751-runtime-gpu-a5/health_guard.py > "$run_directory/health_guard.py"
cp "$evidence"/{collect_prefix.py,guarded_prefix.py} "$run_directory/"
export TRAINING_QUEUE_DIR="$main_checkout/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh
bash "$queue" add "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$run_directory/frozen .venv/bin/python $run_directory/guarded_prefix.py" \
    --name court-side-meiji-prefix-reproduction --provider codex --session "$CODEX_THREAD_ID" --issue 2241 --resource all
bash "$queue" start
```

元の実行ではCUDA異常時の再試行を無効にしました。外部監視はNVML応答と前段のheartbeatを監視し、異常時に全thread stackとGPU診断を保存して停止します。Python側の例外もstackとGPU状態を保存します。

## 保存済み判定を使う

`side-store/objects/` の2ファイルは、実際にloadして全69 nodeで検証した内容アドレス付きの動画判定と単独診断です。この `side-store` を任意のARTIFACT root直下の子ディレクトリに置き、共有コート・driftは `summary.json` の版を指定します。たとえば #1752 の既存コートstoreを使う場合、共通の `.worktree-outputs` をARTIFACT rootとしたときの追加設定は次のとおりです。

```text
court_kp.mode=video_shared
court_kp.video_shared.action=load
court_kp.video_shared.dataset_id=meiji_4cam
court_kp.video_shared.video_id=video_000
court_kp.video_shared.directory=court-video-shared/tennis_scene/evaluate/court-video-shared-acceptance-20261117/store
court_kp.video_shared.artifact_digest=4ac8e386763687fcb6c81f46abf0774e38fc118af4f79b31cdb6e3c2283e3f75
court_kp.video_shared.drift_directory=court-video-shared/tennis_scene/evaluate/court-video-drift-lines-20261117/drift-store
court_kp.video_shared.drift_digest=7a4be8bf20397552a7b287131c3cb0e5fa57ba0dedc16e4a05fa23287c81a7f6
court_side.mode=video_shared
court_side.video_shared.action=load
court_side.video_shared.directory=court-side-video-shared/tennis_scene/evaluate/court-side-20261118/side-store
court_side.video_shared.artifact_digest=a5f987bd59d40a2f20d19d7802a4435748f9e23fb61cddc0bc3a7b704d4cd668
```

このworktree自体から上記の共通ARTIFACT rootを選ぶ際は、`evaluate.py` と同じく `paths.project_root` をメインcheckoutに、`paths.output_root` をrun内の `entry-results` に明示します。PathResolverは設定したrootのbasenameを子パスの先頭に置くことを拒否します。CPU集計の初回はこの組合せで停止し、出力先を明示した集計で結果を取得しました。
