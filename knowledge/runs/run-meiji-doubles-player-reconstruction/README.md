# Meiji 4camのダブルス幅による選手再構成

[#2460](https://github.com/Motoki0705/tennis-lab/issues/2460) の対象は、`i2363-all-v1` に保存された69 clip。`player_selection` の標準設定（横5.485 m、滞在25%）を適用し、各視点で [#2440 の選別mask](../run-i2440-player-selection/README.md) と照合する。成功55件は保存済みの `camera_alignment`、停止14件は保存済みの共有コート向きと校正による幾何を用いる。ボールは [#2458](../run-meiji-external-ball-20261124/README.md) の `ball_triangulation` を束縛する。

選別に使用する検出・pose・外見特徴は、#2363で保存した追跡成果物から読み、配列のSHA-256を検査する。検出・追跡・pose・外見特徴のdescriptorとartifact IDを旧参照に照合する。選別後にcarrierが変わった視点は、保存済みposeを新しいcarrierへ対応付ける。カメラ間対応にはcarrierに対応する保存済みの検出単位の外見特徴を使う。元のcarrierが一致する場合は、対応済みの旧成果物を再利用する。

選手ID、2D座標、confidence、可視maskが一致する場合は三角測量結果を再利用する。身体推定の入力clip、視点、フレーム、box、pose、内部パラメータが一致する場合はHMR2・GVHMRを再利用する。変わった成果物には `doubles_players_*` のノード名を与える。これにより以前のカメラ幾何が参照する依存関係を保ったまま、新しいsceneから選手成果物を参照できる。

3D選手フレームは、有効な3D関節が1点以上ある選手と元フレームの組の数。身体配置後は有効なrootとメッシュの選手フレームも集計する。選手が復元できなかったclipは、空の選手軸、停止理由、#2458のボールを持つsceneとして公開する。

## CPUとGPUの実行

同じworktreeから、`MEIJI_FIXED_DATASET` に #2458版の `i2363-all-v1/inputs/dataset`、`I2440_REPLAY_ROOT` に #2440 の `i2440/replay` を指定する。CPUの出力を後続の段でも使う。

```bash
bundle=knowledge/runs/run-meiji-doubles-player-reconstruction
out=outputs/meiji-doubles-player-reconstruction-final
reconstruct=(.venv/bin/python -B "$bundle/reconstruct.py")
inputs=(--dataset "$MEIJI_FIXED_DATASET" --comparison "$I2440_REPLAY_ROOT" --output "$out")
pilot=(--clip clip_000 --clip clip_020)
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" cpu "${inputs[@]}" "${pilot[@]}" --workers 2
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" cpu "${inputs[@]}" --workers 2
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" cpu-report "${inputs[@]}"
```

`cpu` はclip別の `cpu.json` と採用前のcomponentを保存する。`cpu-report` は、固定した参照と集計を `cpu-clips.csv`、`cpu-upstream.csv`、`cpu-summary.json` に書く。完了済みの `cpu.json` は同じ入力・コード・#2440 のmaskで再利用できる。バッチを再開する際は同じ `--output` を使う。

GPUロックの読み書きができ、親が共有queueの稼働を確認してから、まず代表2件を投入する。#2363と同じTensorRT planと身体モデルを使い、モデル資産を当時のSHA-256と照合する。stageが成功したclipに `gpu.json` が残る。

```bash
(exec 9<>/var/lib/tennis-lab-actions/gpu.lock)
repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
export TRAINING_QUEUE_DIR="$repo/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh
printf -v command '%q ' env "PYTHONPATH=$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "${reconstruct[@]}" gpu "${inputs[@]}" "${pilot[@]}"
bash "$queue" add "$command" --name meiji-doubles-players-pilot --issue 2460 --resource all
```

代表2件の結果を受け取ったら `publish` と `verify` を同じ対象で実行する。確認後は `--clip` を外して共有queueへ全件の `gpu` を投入する。全件の `publish` は対象のCPU成果物と必要な身体配置が揃ってから始まる。

```bash
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" publish "${inputs[@]}"
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" verify "${inputs[@]}"
```

`publish` は先にimmutableなcomponentとNPZを書き、ボール配列と参照を照合してから `scene.json` と `annotation.json` を個別にrenameする。clipごとの `updates/doubles_players/<scene_assembly_ID>.json` に前後の参照、入力、集計を残す。renameの中断後は、同じ成果物で `publish` を再実行できる。`restore` は記録した#2458版の参照を復元する。`verify` は公開された両方の参照、NPZ内のボール5配列のバイト列、固定したカメラ・ボール・前段のartifact IDを検査し、clip別の結果を `clips.csv` と `upstream.csv` に書く。

## 検証

- 選別の幅・滞在率・外見モデルの契約は、既存の `test_membership_boundary_is_doubles_width_and_baseline_plus_five_metres`、`test_doubles_width_and_shorter_dwell_recover_different_real_observations`、`test_tracking_features_require_the_same_model_identity` を再利用する。実データでは両パイロットから始め、#2440 のmaskとartifact IDを全4視点で照合する。 [選別テスト](../../../tests/unit/tennis_scene/pipeline/components/test_player_selection.py)
- 既存の `test_the_decided_side_places_the_camera_local_courts` と `test_separated_features_preserve_body_parameters_placement_and_asset_loads` で、半回転の向きと身体推定の入力契約を検証する。 [人物対応テスト](../../../tests/unit/tennis_scene/pipeline/test_player_association_component.py)、[身体推定テスト](../../../tests/integration/tennis_scene/test_gvhmr_input_from_stored_selection.py)
- 参照切替用のintegrationテストを1件追加する。#2458の参照から切り替える途中で2つ目のrenameを中断し、再実行で公開できることを確認する。ボール配列と幾何のartifact ID、#2458版への復元も確認する。GPUや実データ資産を使わず、既存の短いclip fixtureで実行する。 [公開テスト](../../../tests/integration/tennis_scene/test_external_ball_update.py)
