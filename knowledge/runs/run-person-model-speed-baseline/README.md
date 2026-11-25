# Meiji 4camの現行人物モデル速度基準

[#2496](https://github.com/Motoki0705/tennis-lab/issues/2496) の固定入力、初回計測の記録、再計測用スクリプトを収める。再計測はRTX 5060 Ti 16 GB上で、DINO 4-scale Swin-L、ViTPose-Hのflipありとなし、HMR2特徴＋GVHMRをPT16・TRT16で測る。初回のPT32とDINO＋ViTPose-H合計は修正前の計時プログラムによる暫定値。[共通harness](../../../src/utils/inference_benchmark.md)

## 固定入力と条件

- `meiji_4cam/multiview-v1` の代表6 clipを使う。全4人のGVHMR実推論frameの共通集合から、各clipで20・40・60・80%の位置を選ぶ。各時点の4視点を含む96画像で、clip内の時刻と視点を揃え、DINOのbatch 1/2を全条件で比較する。
- 固定cropは、初回に取得した各画像のPT32 DINO検出boxから作った。同frameの `multiview_scene_inputs.people.raw` にある実観測のsquare boxとの最大IoU、次いで検出score順で並べ、最大4人をstandalone ViTPoseのcropに採った。共通crop集合はbatch 8の完全バッチを作れる件数に揃え、端数のIDもmanifestに記録した。
- `body_view_selection` から各clip・各選手に、全4人同時にGVHMRが実推論した連続16サンプルの窓を1本ずつ取る。24系列で、選択済みview・校正結果・観測keypoint・boxを使う。各系列内は2 source frame間隔。HMR2の16 crop batchで特徴を毎回作り、同長の1/2/4系列をGVHMR denoiserで処理して人物ごとに静止カメラ補正とIKを行う。
- 30回warm-up、100回×3ブロック、TF32無効。比較軸はDINOで画像batch 1/2、ViTPose単体でcrop batch 1/4/8、身体で系列batch 1/2/4。身体の平均単位は有効person-frame。各ケースは独立プロセスで動かす。素材は事前にBGRへdecodeし、計測の前処理でresize・normalize・人物cropを行う。
- PT16ではCUDA autocast FP16を使う。DINOのCUDA deformable attentionとGVHMRの復号・補正・IKはFP32。TRT16では既存のDINO GridSample・ViTPose・HMR2・GVHMR denoiserのplanを使い、GVHMRの復号・補正・IKは共通のPyTorch FP32を使う。ViTPoseのflipなしbatch 1は既存ONNXから内部batch 1/1/2の別planを構築し、buildのwall timeを記録する。他のViTPose条件は既存planを使う。
- 再計測の全22条件は「未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）」と記録する。初回のPT32参照出力とparity指標は保存済みの結果から参照できる。[共通parityの契約](../../../src/utils/inference_benchmark.md#出力のparity)

## 再現

worktreeのrootで実行する。`PERSON_TRT_ROOT` は #2363 の `tensorrt/`、`PERSON_EXTERNAL_ASSETS` は検証済み `DINO/` checkoutの親、`TENNIS_LAB_DINO_EXTENSION` は同じPyTorch/CUDAでビルド済みのCUDA拡張を指す。`PERSON_FIXED_INPUTS` は公開済みの固定入力、`PERSON_POSE_ENGINE_RUN` はflipなしB=1の追加planを構築した初回計測の出力を指す。`output` には新規ディレクトリを指定する。

```bash
bundle=knowledge/runs/run-person-model-speed-baseline
output=outputs/person-model-speed-low-precision
PERSON_FIXED_INPUTS=outputs/person-model-speed-baseline-v1/inputs
PERSON_POSE_ENGINE_RUN=outputs/person-model-speed-baseline-v1
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"

runner=(env "PYTHONPATH=$PWD" "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  .venv/bin/python -B -u "$bundle/run.py" --output "$output"
  --input-dir "$PERSON_FIXED_INPUTS" --pose-engine-run "$PERSON_POSE_ENGINE_RUN"
  --trt-root "$PERSON_TRT_ROOT" --external-assets "$PERSON_EXTERNAL_ASSETS")

"${runner[@]}" inspect
for phase in detection pose body; do
  printf -v command '%q ' "${runner[@]}" suite "$phase"
  bash "$queue" add "$command" --name "person-speed-low-precision-$phase" --resource all \
    --issue 2496 --provider codex --session "$CODEX_THREAD_ID"
done
```

再計測では公開済みの固定入力を読み、manifestと計時protocolの期待SHA-256を照合する。追加planを作る場合は `--pose-engine-run` を省略する。pose suiteが条件の計時と別のsubprocessで追加planを構築する。

初回計測のcommitは `0ee6f8dde06c078b6ab40324f5416b68a9eaefc0`。共有queueの4ジョブの開始時に同じcommitと空の差分を確認した。`PERSON_TRT_ROOT` には `.worktree-outputs/parent-2278-issue-2363-primary/tensorrt` を使った。実行時の引数と環境変数はメインcheckoutからの `.training_queue/repro/<job ID>/run.json` にある。 [初回のqueue実行記録](queue-receipts.json)

## 固定入力と初回の成果物

- [固定入力manifest](inputs/manifest.json)が `meiji_4cam/person-model-speed-v1` の正本。全画面96枚、人物crop 384件、身体24系列、身体用224枚のID・パス・SHA-256をindexに保存した。manifestのSHA-256は `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62`。実データ728ファイルはメインcheckoutの `.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs/` にある。各indexの `path` はこの入力ディレクトリ相対。
- ViTPoseのB=1計時は固定順の先頭300 cropで、clip_003・057・004・013・040をカバーする。B=4/8の計時と初回の全ViTPose条件のparityには6 clipの384 cropを使った。候補比較には同じmanifestと [計時protocol](timing-protocol.json) を使う。protocolのSHA-256は `39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2`。共通の `PinnedBenchmarkInputs` にこの値とmanifestの期待SHA-256を渡す。
- [初回39条件の暫定値](cases.csv)に当時の判定、P50/P95、items/s、VRAM、load・cold startがある。当時parityが不成立だった行の速度欄は空欄で、実測した各反復は `measurements/<workload>-<backend>.csv` に保存した。同一反復中の複数forwardはCUDA event時間を合算し、同種stageのhost時刻も反復単位で合算した。0始まりの `batch_index` は、workloadに対応する入力indexをbatch sizeごとに先頭から区切った順序を指す。
- 環境、重みとplanのSHA-256、初回parity、元JSONとログのSHA-256は [初回の条件別記録](cases.json) にある。個別stage呼び出しを含む元JSONと予測は初回の永続出力 `outputs/person-model-speed-baseline-v1/results/<workload>/<backend>-b<batch>/` にある。追加planの初回buildは [receipt](vitpose-b1-build.json) に記録した。
- 初回の代表値と先行値の照合は [runノード](../../nodes/tennis_scene/000085-run-person-model-speed-baseline.md) にある。

再計測後は728ファイルの内容と各反復の入力ID・集計を照合し、新しい記録ディレクトリに22条件を出力する。`parity_status`、未判定の理由、速度を条件ごとに記録する。計時に失敗した条件も、理由と取得できた反復を記録する。

```bash
result_bundle=knowledge/runs/run-person-model-speed-low-precision
PYTHONPATH="$PWD" .venv/bin/python -B "$bundle/record.py" "$output" "$result_bundle" \
  --input-dir "$PERSON_FIXED_INPUTS"
```

このbundleのCPU検証は `PYTHONPATH="$PWD:$PWD/$bundle" .venv/bin/python -m pytest -n 0 "$bundle/test_speed_baseline.py"`。現行DINOと一致する検出復号、観測を使ったcrop選択とSHA-256、px単位のparityを確認する。
