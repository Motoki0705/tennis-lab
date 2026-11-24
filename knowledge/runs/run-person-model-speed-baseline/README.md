# Meiji 4camの現行人物モデル速度基準

[#2496](https://github.com/Motoki0705/tennis-lab/issues/2496) の再現用bundle。DINO 4-scale Swin-L、ViTPose-Hのflipありとなし、DINO＋ViTPose-H、HMR2特徴＋GVHMRを、RTX 5060 Ti 16 GB上でPT32・PT16・TRT16に揃えて測る。計時は[共通harness](../../../src/utils/inference_benchmark.md)を使う。

## 固定入力と条件

- `meiji_4cam/multiview-v1` の代表6 clipを使う。全4人のGVHMR実推論frameの共通集合から、各clipで20・40・60・80%の位置を選ぶ。各時点の4視点を含む96画像で、clip内の時刻と視点を揃え、DINOのbatch 1/2を全条件で比較する。
- 各画像のPT32 DINO検出boxを、同frameの `multiview_scene_inputs.people.raw` にある実観測のsquare boxとの最大IoU、次いで検出score順で並べ、最大4人をstandalone ViTPoseのcropに採る。共通crop集合はbatch 8の完全バッチを作れる件数に揃え、端数のIDもmanifestに記録する。DINO＋ViTPoseは全検出をその場でcropして、flipありの最大8 cropずつで推論する。
- `body_view_selection` から各clip・各選手に、全4人同時にGVHMRが実推論した連続16サンプルの窓を1本ずつ取る。24系列で、選択済みview・校正結果・観測keypoint・boxを使う。各系列内は2 source frame間隔。HMR2の16 crop batchで特徴を毎回作り、同長の1/2/4系列をGVHMR denoiserで処理して人物ごとに静止カメラ補正とIKを行う。
- 30回warm-up、100回×3ブロック、TF32無効。比較軸はDINOとDINO＋ViTPoseで画像batch 1/2、ViTPose単体でcrop batch 1/4/8、身体で系列batch 1/2/4。身体の平均単位は有効person-frame。各ケースは独立プロセスで動かす。素材は事前にBGRへdecodeし、計測の前処理でresize・normalize・人物cropを行う。
- PT16ではCUDA autocast FP16を使う。DINOのCUDA deformable attentionとGVHMRの復号・補正・IKはFP32。TRT16では既存のDINO GridSample・ViTPose・HMR2・GVHMR denoiserのplanを使い、GVHMRの復号・補正・IKは共通のPyTorch FP32を使う。ViTPoseのflipなしbatch 1は既存ONNXから内部batch 1/1/2の別planを構築し、buildのwall timeを記録する。他のViTPose条件は既存planを使う。
- PT32 batch 1の参照出力をID別に保存する。後続ケースのparityは同じID集合で、DINOのIoU・score・欠落/追加、ViTPoseのpx差とconfidence差、身体の特徴cosine/相対L2・SMPL-Xの回転角と関節距離を判定する。閾値は[先行プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md#精度の指標と許容差草案)を使う。

## 再現

worktreeのrootで実行する。`PERSON_TRT_ROOT` は #2363 の `tensorrt/`、`PERSON_EXTERNAL_ASSETS` は検証済み `DINO/` checkoutの親、`TENNIS_LAB_DINO_EXTENSION` は同じPyTorch/CUDAでビルド済みのCUDA拡張を指す。

```bash
bundle=knowledge/runs/run-person-model-speed-baseline
output=outputs/person-model-speed-baseline-v1
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"

runner=(env "PYTHONPATH=$PWD" "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  .venv/bin/python -B -u "$bundle/run.py" --output "$output"
  --trt-root "$PERSON_TRT_ROOT" --external-assets "$PERSON_EXTERNAL_ASSETS")

"${runner[@]}" inspect
for phase in prepare detection pose body; do
  if [[ $phase == prepare ]]; then
    printf -v command '%q ' "${runner[@]}" prepare
  else
    printf -v command '%q ' "${runner[@]}" suite "$phase"
  fi
  bash "$queue" add "$command" --name "person-speed-$phase" --resource all \
    --issue 2496 --provider codex --session "$CODEX_THREAD_ID"
done
```

計測commitは `0ee6f8dde06c078b6ab40324f5416b68a9eaefc0`。共有queueの4ジョブの開始時に同じcommitと空の差分を確認した。`PERSON_TRT_ROOT` には `.worktree-outputs/parent-2278-issue-2363-primary/tensorrt` を使った。実行時の引数と環境変数はメインcheckoutからの `.training_queue/repro/<job ID>/run.json` にある。 [queueの実行記録](queue-receipts.json)

## 記録された成果物

- [固定入力manifest](inputs/manifest.json)が `meiji_4cam/person-model-speed-v1` の正本。全画面96枚、人物crop 384件、身体24系列、身体用224枚のID・パス・SHA-256をindexに保存した。manifestのSHA-256は `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62`。実データ728ファイルはメインcheckoutの `.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs/` にある。各indexの `path` はこの入力ディレクトリ相対。
- ViTPoseのB=1計時は固定順の先頭300 cropで、clip_003・057・004・013・040をカバーした。B=4/8の計時と全ViTPose条件のparityは6 clipの384 cropをカバーした。候補比較では同じmanifestとbatchごとの巡回順を使う。[条件別計時対象](cases.json)
- [全39条件の表](cases.csv)には判定、P50/P95、items/s、VRAM、load・cold startがある。実行不能行の速度欄は空欄。同期済みの各反復は `measurements/<workload>-<backend>.csv` に保存した。同一反復中の複数forwardはCUDA event時間を合算し、同種stageのhost時刻も反復単位で合算する。0始まりの `batch_index` は、workloadに対応する入力indexをbatch sizeごとに先頭から区切った順序を指す。
- 環境、重みとplanのSHA-256、parity、各計測の元JSONとログのSHA-256は [条件別記録](cases.json) にある。絶対パスは `ckpt/` または再現用環境変数を起点に記した。`PERSON_RUN_OUTPUT` は再現コマンドの `output`。個別stage呼び出しを含む元JSONと予測は同じ永続出力の `results/<workload>/<backend>-b<batch>/` にある。追加planのbuildは [receipt](vitpose-b1-build.json) に記録した。
- 代表値、parityで実行不能になった理由、先行値との比較は [runノード](../../nodes/tennis_scene/000085-run-person-model-speed-baseline.md) にある。

728ファイルの内容と、39条件11,700反復の入力ID・集計を再照合し、公開用の計測記録を生成するCPUコマンド。

```bash
PYTHONPATH="$PWD" .venv/bin/python -B "$bundle/record.py" "$output" "$bundle"
```

このbundleのCPU検証は `PYTHONPATH="$PWD:$PWD/$bundle" .venv/bin/python -m pytest -n 0 "$bundle/test_speed_baseline.py"`。現行DINOと一致する検出復号、観測を使ったcrop選択とSHA-256、px単位のparityを確認する。
