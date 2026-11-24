# Meiji 4camのダブルス幅による選手再構成

[#2460](https://github.com/Motoki0705/tennis-lab/issues/2460) の対象は、`i2363-all-v1` に保存された69 clip。`player_selection` の標準設定（横5.485 m、滞在25%）を適用し、各視点で [#2440 の選別mask](../run-i2440-player-selection/README.md) と照合する。成功55件は保存済みの `camera_alignment`、停止14件は保存済みの共有コート向きと校正による幾何を用いる。ボールは [#2458](../run-meiji-external-ball-20261124/README.md) の `ball_triangulation` を束縛する。

選別に使用する検出・pose・外見特徴は、#2363で保存した追跡成果物から読み、配列のSHA-256を検査する。検出・追跡・pose・外見特徴のdescriptorとartifact IDを旧参照に照合する。選別後にcarrierが変わった視点は、保存済みposeを新しいcarrierへ対応付ける。カメラ間対応にはcarrierに対応する保存済みの検出単位の外見特徴を使う。元のcarrierが一致する場合は、対応済みの旧成果物を再利用する。

選手ID、2D座標、confidence、可視maskが一致する場合は三角測量結果を再利用する。身体推定の入力clip、視点、フレーム、box、pose、内部パラメータが一致する場合はHMR2・GVHMRを再利用する。変わった成果物には `doubles_players_*` のノード名を与える。これにより以前のカメラ幾何が参照する依存関係を保ったまま、新しいsceneから選手成果物を参照できる。

3D選手フレームは、有効な3D関節が1点以上ある選手と元フレームの組の数。身体配置後は有効なrootとメッシュの選手フレームも集計する。選手が復元できなかったclipは、空の選手軸、停止理由、#2458のボールを持つsceneとして公開する。

## パイロット2件の読戻し

| clip | 選手数 #2458 → 試行版 | 3D選手フレーム | 有効root | 有効メッシュ |
| --- | ---: | ---: | ---: | ---: |
| clip_000 | 4 → 4 | 1,416 → 1,416 | 1,410 → 1,410 | 1,407 → 1,407 |
| clip_020 | 0 → 4 | 0 → 2,579 | 0 → 2,575 | 0 → 2,550 |

| clip | CPU準備 | HMR2 | GVHMR | 身体配置 | Scene組立から読戻し |
| --- | ---: | ---: | ---: | ---: | ---: |
| clip_000 | 50.14秒 | 再利用 | 再利用 | 14.02秒 | 12.33秒 |
| clip_020 | 86.67秒 | 82.36秒 | 14.55秒 | 17.41秒 | 20.53秒 |

共有training queueのパイロットは `done`。取得した実行版は `bac14b71cfbbf0a9cd6bda9c5efef1fc97c84d61` で、再現情報の記録から完了までは約176秒だった。両clipのGVHMRの選手・カメラ・元フレームと、身体配置とsceneのroot・メッシュmaskを照合した。

別ストアで新しい `scene_assembly` とNPZを書いて読み戻した。各clipのボール5配列は #2458 とバイト一致し、カメラとボールのartifact IDも一致した。共有datasetの `scene.json` と `annotation.json` は、検証の前後とも69件すべてが #2458 の切替記録とバイト一致した。 [パイロットのID・選手数・処理時間](pilot-summary.json)

2026-11-24、ユーザーの判断で #2460 の身体工程をこの2件の試行で止め、共有GPUは #2446 の新経路比較に配分した。残るclipの身体工程と実datasetの参照切替は、再開の対象として記録した。 [判断記録 #1739](https://github.com/Motoki0705/tennis-lab/issues/1739)

## 全69 clipのCPU結果

| 指標 | #2458 | ダブルス幅のCPU結果 |
| --- | ---: | ---: |
| 3D選手フレーム | 133,887 | 147,105 |
| 選手の3D関節が得られたclip | 55 | 58 |
| 停止clip | 14 | 11 |

clip_012（4,060選手フレーム）、clip_020（2,579）、clip_058（3,052）が回復し、各clipで4人を復元した。残る11件は選手の確定が曖昧な9件と、視点間の対応が曖昧な2件。全276視点の選別maskを読み戻し、#2440のダブルス幅と一致した。旧幅から変化したのは72視点だった。

28件は保存済みの選手結果を再利用できる。22件は保存済みのGVHMRから身体配置へ進み、8件はHMR2から実行する。clip別に選手数、フレーム数、停止理由と処理時間を記録した。CPUのclip別処理時間の合計は3,812.03秒で、並列実行時間も加算している。 [clip別のCPU記録](cpu-clips.csv)、[集計](cpu-summary.json)

69件の公開参照を#2458の記録に照合し、前段1,104件のartifact ID、成功55件と停止14件で固定したカメラ幾何、ボール成果物を確認した。 [前段のartifact ID](cpu-upstream.csv)

## 保存済み成果物の検証と再開

同じworktreeから、`MEIJI_FIXED_DATASET` に #2458版の `i2363-all-v1/inputs/dataset`、`I2440_REPLAY_ROOT` に #2440 の `i2440/replay` の絶対パスを指定する。`out` には保存済みの `cpu.json` と `gpu.json` があるディレクトリを指定する。同じコードSHA-256で照合できるよう、実行時点のソースを展開する。 #2458 のCLI入口の修正を取り込む前にパイロットを実行した。

```bash
worktree="$PWD"
bundle=knowledge/runs/run-meiji-doubles-player-reconstruction
out="$worktree/outputs/meiji-doubles-player-reconstruction-final"
code_dir=$(mktemp -d "$out/replay-code.XXXXXX")
git archive bac14b71cfbbf0a9cd6bda9c5efef1fc97c84d61 src "$bundle" | tar -x -C "$code_dir"
reconstruct=("$worktree/.venv/bin/python" -B "$code_dir/$bundle/reconstruct.py")
inputs=(--dataset "$MEIJI_FIXED_DATASET" --comparison "$I2440_REPLAY_ROOT" --output "$out")
export PYTHONPATH="$code_dir" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

pilot_report=$(mktemp -d "$out/pilot-review.XXXXXX")
CUDA_VISIBLE_DEVICES='' "$worktree/.venv/bin/python" -B "$bundle/verify_pilot.py" "${inputs[@]}" \
    --readback "$pilot_report/stores" \
    --job-id 0-8204502006022507738_2026-11-24T05-26-33-977492261Z_204466_meiji-doubles-players-pilot.job

# 69件のCPU成果物の再読込と集計
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" cpu "${inputs[@]}" --workers 2
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" cpu-report "${inputs[@]}"
```

`cpu` はclip別の `cpu.json` と採用前のcomponentを保存する。`cpu-report` は、固定した参照と集計を `cpu-clips.csv`、`cpu-upstream.csv`、`cpu-summary.json` に書く。バッチを再開する際は同じ `--output` を使う。保存済み結果は入力、コードの版、#2440 のmaskを照合して再利用する。

初回のCPUバッチで025・054・059・062・064・065の6件に、選別前の追跡行で外見特徴の欠測が計31件あった。31件はいずれも #2440 の旧幅とダブルス幅の選別maskの外だった。保存済みの特徴で選別し、再対応付けで使用する観測の特徴を必須とする契約に直した。初回の完了結果は、準備コードのSHA-256と固定入力を確認して引き継いだ。欠測のあった6件を修正版で実行した。

#2460 の全件反映を再開する指示があり、親が共有queueとGPUロックの稼働を確認した時点で、上の設定を使って全69件を対象にキューへ登録する。保存済みCPU69件とGPUパイロット2件を照合して引き継ぐ。GPUは身体工程が残る28件で使用する。#2363と同じTensorRT planと身体モデルを使い、モデル資産を当時のSHA-256と照合する。

```bash
for suffix in '' .gate .slot-0 .slot-1; do
    (exec 9<>"/var/lib/tennis-lab-actions/gpu.lock$suffix") || exit 1
done
repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
export TRAINING_QUEUE_DIR="$repo/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh
printf -v command '%q ' env "PYTHONPATH=$code_dir" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "${reconstruct[@]}" gpu "${inputs[@]}"
bash "$queue" add "$command" --name meiji-doubles-players-full --issue 2460 --resource all \
    --provider codex --session "$CODEX_THREAD_ID"
```

再開後の全件GPUの結果を受け取り、参照切替の指示で `publish` と `verify` を実行する。`publish` は対象のCPU成果物と必要な身体配置が揃ってから始まる。

```bash
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" publish "${inputs[@]}"
CUDA_VISIBLE_DEVICES='' "${reconstruct[@]}" verify "${inputs[@]}"
```

`publish` は先にimmutableなcomponentとNPZを書き、ボール配列と参照を照合してから `scene.json` と `annotation.json` を個別にrenameする。clipごとの `updates/doubles_players/<scene_assembly_ID>.json` に前後の参照、入力、集計を残す。renameの中断後は、同じ成果物で `publish` を再実行できる。`restore` は記録した#2458版の参照を復元する。`verify` は公開された両方の参照、NPZ内のボール5配列のバイト列、固定したカメラ・ボール・前段のartifact IDを検査し、clip別の結果を `clips.csv` と `upstream.csv` に書く。

## 検証

- 選別の幅・滞在率・外見モデルの契約は、既存の `test_membership_boundary_is_doubles_width_and_baseline_plus_five_metres`、`test_doubles_width_and_shorter_dwell_recover_different_real_observations`、`test_tracking_features_require_the_same_model_identity` を再利用する。実データでは両パイロットから始め、#2440 のmaskとartifact IDを全4視点で照合する。 [選別テスト](../../../tests/unit/tennis_scene/pipeline/components/test_player_selection.py)
- 既存の `test_the_decided_side_places_the_camera_local_courts` と `test_separated_features_preserve_body_parameters_placement_and_asset_loads` で、半回転の向きと身体推定の入力契約を検証する。 [人物対応テスト](../../../tests/unit/tennis_scene/pipeline/test_player_association_component.py)、[身体推定テスト](../../../tests/integration/tennis_scene/test_gvhmr_input_from_stored_selection.py)
- 参照切替用のintegrationテストを1件追加する。#2458の参照から切り替える途中で2つ目のrenameを中断し、再実行で公開できることを確認する。ボール配列と幾何のartifact ID、#2458版への復元も確認する。GPUや実データ資産を使わず、既存の短いclip fixtureで実行する。 [公開テスト](../../../tests/integration/tennis_scene/test_external_ball_update.py)

関連pytestは391件が66.34秒で通過した。増分は1件で、単独実行のpytest計時は7.09秒だった。一時ストアを使った実データ検査では、clip_002とclip_006で切替、ボール5配列のバイト一致、復元を確認した。GPUで使用するHMR2・GVHMR・身体モデル資産16ファイルのSHA-256は#2363の記録と一致した。

#2458 のCLI入口の修正を取り込んだ後、関連するpipeline、公開、設定のpytest 788件が281.48秒で通過した。パイロット読戻し用スクリプトはruffとmypyで検査した。
