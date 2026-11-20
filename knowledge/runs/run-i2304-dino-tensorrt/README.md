# DINO 4-scale Swin-L 固定入力の TensorRT 比較

[#2304](https://github.com/Motoki0705/tennis-lab/issues/2304) と [#2297 の比較プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md) に対応する。対象は `checkpoint0029_4scale_swin.pth` の人物クラス。計測区間はGPUに転送済みの画像から `pred_logits` と `pred_boxes` が得られるまで。現行decoderを元画像H/Wと confidence 0.3 で計測後に実行し、boxを条件0と照合する。

## 固定入力

| 用途 | clip | カメラ | 枚数 |
|---|---|---|---:|
| 評価 | `clip_000`、`clip_037` | cam0–cam3 | 64 |
| PTQ | `clip_027` | cam0–cam3 | 64 |

現行 `DinoPersonDetector._predict_batch_impl` で前処理し、`_predict_compatible_batch` のstackをCPUで受け取る。元画像は全128枚で `1080×1920`。正規化済みFP32 RGBのshape bucketは `(1, 3, 750, 1333)` の1つ。動画のSHA-256、収録元のカメラ別フレーム区間と、固定テンソル128枚の順序・shape・元画像H/Wを照合した。評価と校正で共通する動画ファイルと元フレーム区間は各0件。別出力先へのCPU再採取でtensor lockが一致した。TensorRTは各次元をビルド時に固定し、min、opt、maxを同じshapeとする。固定の代表入力は `clip_000/cam0/frame 178`。 [共通動画・フレームpin](../run-i2297-clipreid-tensorrt/inputs.json)・[DINO tensor lock](tensor-lock.json)

## 再現環境

[TensorRT専用環境](../run-i2297-clipreid-tensorrt/README.md#専用環境)に、DINOとその公開importで使うライブラリを [固定の追加requirements](requirements.txt) から導入する。DINO submoduleを初期化し、[#2238のrun](../../nodes/tennis_scene/000050-run-person-stage-precision-meiji-20261118.md) が指定するCUDA拡張の絶対パスを `TENNIS_LAB_DINO_EXTENSION` に指定する。計測入口で拡張のSHA-256を照合する。

```bash
bundle=knowledge/runs/run-i2304-dino-tensorrt
shared=knowledge/runs/run-i2297-clipreid-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/trt-dino-reproduction
git submodule update --init third_party/DINO
uv pip install --python "$trt_exp_env/bin/python" -r "$bundle/requirements.txt"
uv pip check --python "$trt_exp_env/bin/python"
export PYTHONPATH="$PWD:$PWD/$shared"
"$trt_exp_env/bin/python" -B "$bundle/fixed_inputs.py" --output "$out/inputs"
```

共有のtraining queueを利用する。`TRAINING_QUEUE_DIR` はメインworktreeにある `.training_queue` を指定する。推論・変換・PTQ・ビルドをそれぞれ別のプロセスにし、 `--resource all` で実行する。コマンドはworktreeルートから登録する。

```bash
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
registration=(--resource all --issue 2304)
if [[ -n "${CODEX_THREAD_ID:-}" ]]; then
    registration+=(--provider codex --session "$CODEX_THREAD_ID")
fi
enqueue_dino() {
    local job="$1" command
    shift
    printf -v command '%q ' env "PYTHONPATH=$PYTHONPATH" \
        "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION" \
        OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 "$trt_exp_env/bin/python" -B "$@"
    bash "$queue" add "$command" --name "$job" "${registration[@]}"
}

enqueue_dino i2304-msda-probe "$bundle/probe_msda.py" --output "$out/probe"
enqueue_dino i2304-dino-export-grid "$bundle/export_dino.py" --method grid \
    --bucket 750x1333 --inputs "$out/inputs" --output "$out/export-grid"
```

`comparison.json` は小規模のvalue projectionとサンプリングを比較する。部分TRT化ではvalue projectionをTensorRT FP16で実行し、同一CUDA streamで結果をFP32へcastしてPyTorchのCUDA拡張へ渡す。境界を含めて同期計測する。TensorRTプラグインは配布版の `MultiscaleDeformableAttnPlugin_TRT` v2を使う。 [NVIDIA v2実装](https://github.com/NVIDIA/TensorRT/blob/release/10.14/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.cpp)

inspectorの実行層とONNX演算は `Metadata` で対応付ける。DINOのncuはkernel replayを使い、上限を既定で1条件45分とする。再試行では `--ncu-timeout` で秒数を指定できる。nsysとの件数一致を収集完了条件にする。タイムアウト時はqueueのプロセスグループ内の監督プロセスがncuと残った子孫を停止し、子プロセスが残っていないことを確かめてから失敗を返す。 [試行記録と終了値](attempts.json)

## MSDAの方式比較

4レベル、900 query、8 head、各head・各レベル4サンプルの固定入力で計測した。FP32参照はvalue projectionとDINO CUDA拡張。速度は同じ範囲を30回warm-upした後の同期済み100回。誤差は出力全成分をFP32参照と比較した。 [実行層・入出力精度と生データ](comparison.json)

| 方式 | ONNX・TensorRT | 中央値 / P95 ms | 絶対誤差 P95 | 相対L2 | 実装・保守 |
|---|---|---:|---:|---:|---|
| 標準GridSample | 標準ONNXからビルド・実行 | 0.496 / 0.585 | 0.001471 | 0.005613 | 上流の参照計算を固定shapeでexportし、同じグラフをORT CPU校正に使う |
| TensorRT v2プラグイン | custom ONNX nodeからビルド・実行 | 0.091 / 0.144 | 0.002099 | 0.007902 | 配布済みプラグインを初期化し、位置・重み・INT32 shapeのABIを合わせる。planのビルド先と配布先でTensorRTとプラグインの版を管理する。ORT CPU校正には標準opのグラフも用意する |
| 部分TRT化 | FP16 projectionをビルド・実行、続けてPyTorch FP32 MSDAを実行 | 0.201 / 0.356 | 0.000172 | 0.000673 | TensorRT出力のcastとCUDA streamを管理する。本体では分割点ごとのバッファと実行順の管理が必要 |

PTQ条件は、同じ標準ONNXでModelOptのCPU校正とTensorRT実行ができるGridSampleで測った。プラグインはTensorRT配布版を利用した。対象環境に実装が含まれない場合はプラグインのビルドと配布が必要になる。標準グラフからプラグイングラフへ校正結果を移す手順は未検証。校正が不要な条件2は、正式値のGridSample版と比較用のプラグイン版を同じ固定画像で計測した。部分TRT化は小規模比較まで実施した。Phase 2の方式選定は #2278 で行う。 [モデル全体の結果と考察](../../nodes/tennis_scene/000078-run-i2304-dino-tensorrt.md)

## 条件別の精度

条件0はTF32を無効化したPyTorch FP32。条件1は #2238 と同じCUDA autocast FP16で、MSDAの各ブロック全体はFP32。条件2はTensorRTのstrongly typed FP16で、DINOの主要Conv・Gemm・MatMulのオペランドをinspectorで照合する。画像の転送とdecoderは計測区間の外に置く。

条件2の精度低下の原因は未確認。条件1から条件2へ比較を移すと、MSDAの方式とバックエンドに加え、Swinなどを含むONNX全体のFP16変換が加わる。固定入力でのモデル全体の差は観測できた。各変更の寄与を切り分けるには、同一精度でのONNX export前後の出力照合と、モデル全体でMSDAブロックだけをFP32に戻す対照が必要で、いずれも未実施。 [精度の集計と比較条件](metrics.json)

shifted-window attentionの融合層では、マスクの2つのviewを形状で確認した上でQ/K/Vのdtypeを照合する。最初のdecoder V射影は固定B=1で学習済みqueryと重みを使うため、inspectorでは実行時の入力が空になる。対象のONNX演算を特定し、タクティクス名の両オペランド精度を照合する。

INT8のGEMMと残差和が融合した層では、出力と同じshapeのFP16残差と、INT8の活性入力を区別する。実行タクティクスの両オペランドと、融合層に含まれるQ/DQ対も照合する。1×1畳み込みとして実行される線形層では `conv_fprop_smallk_i8i8_i32` をオペランド精度の証拠に使う。 [主要演算ごとの要求精度](precision-requirements.json)・[試行記録](attempts.json)

計測時点の #2297 の設定でModelOpt `max`、ORT CPU、明示Q/DQを適用した。校正は `clip_027` の固定64枚。主要演算のオペランドで、3aはINT8 292演算、FP16 24演算（box最終射影12とdecoder self-attentionの行列積12）を照合した。3bはFP8 E4M3 275演算、FP16 41演算（box最終射影12、Conv 5、Swin shifted-window attentionの行列積24）を照合した。

実測planに対応するONNXのハッシュを照合したところ、3a・3bとも指定外の数値入力にも量子化後の値が渡っていた。DequantizeLinearの出力を直接受けるAddが36件、LayerNormalizationが5件、Whereが1件ずつあり、最初のencoder残差Addの第1入力も含まれる。Shapeへの接続は3aが123件、3bが117件で、参照するのはテンソルの寸法。3a・3bの速度と精度はこのグラフでの実測値。 [演算種別・入力辺の集計と対象ハッシュ](metrics.json)

現行のprepare/buildは共通の `verify_qdq_scope` で指定演算・オペランド位置とDQの全接続先を照合し、旧グラフを非ゼロで拒否する。#2303（PR #2334）のマージ後は、ViT 3モデルの3aがattention出力投影だけのINT8 max、3bがQKV・attention出力・MLP前後の線形演算を対象にしたFP8 maxになった。新設定でのDINOは未計測で、再計測の採否は #2278 で判断する。 [現行のViT 3モデル向け設定](../run-i2297-clipreid-tensorrt/PROTOCOL.md#vit-3モデルのptq設定)

ModelOptはshape推論で入力ONNXを書き換えるため、校正は各条件のコピーで行う。decoderの共有重みを隠すONNX `Identity` は行列積から同じinitializerを直接参照する形に直す。別名のままでは、固定B=1がGEMVの判定に使われて18演算が校正対象から外れた。前処理にはTensorRTによるop検査があり、PTQジョブのCUDA可視性を保持する。 [試行記録](attempts.json)

## 精度別の再現コマンド

```bash
for condition in 0 1; do
    reference=()
    if [[ "$condition" == 1 ]]; then reference=(--reference "$out/0"); fi
    enqueue_dino "i2304-dino-$condition" "$bundle/benchmark_dino.py" \
        --condition "$condition" --bucket 750x1333 --inputs "$out/inputs" \
        --output "$out/$condition" --profiles nsys "${reference[@]}"
    enqueue_dino "i2304-dino-ncu-$condition" "$bundle/benchmark_dino.py" \
        --condition "$condition" --bucket 750x1333 --inputs "$out/inputs" \
        --output "$out/ncu-$condition" --profiles ncu --profile-from "$out/$condition"
done

enqueue_dino i2304-dino-prepare-2 "$bundle/convert_dino.py" prepare \
    --condition 2 --bucket 750x1333 --inputs "$out/inputs" \
    --source "$out/export-grid" --output "$out/typed-2"
enqueue_dino i2304-dino-build-2 "$bundle/convert_dino.py" build \
    --condition 2 --bucket 750x1333 --inputs "$out/inputs" \
    --source "$out/typed-2" --output "$out/engine-2"
```

ビルドが成功した条件で、`benchmark_dino.py` に `--engine-dir "$out/engine-$condition" --reference "$out/0"` を追加して測る。追測の `--profile-from` では同一 `--engine-dir` を指定し、`--reference` は省く。TensorRTの主要演算の条件別精度をONNX演算と実行層で照合し、検証に失敗したビルドは非ゼロで停止する。

プラグイン版の条件2では `--method plugin` でexportする。`prepare` の `--condition 2` にそのexportを渡し、`build` と `benchmark_dino.py` でも各成果物を指定する。GridSample版と共通の `--bucket 750x1333 --inputs "$out/inputs"` とFP32参照を用い、出力先には別ディレクトリを指定する。照合済みの変換記録から方式を計測プロセスへ渡し、プラグイン版を選んだ場合だけロード時に初期化する。変換時はプラグインの `custom_layer` と3個のFP16入力・1個のFP16出力を検査する。GridSampleは融合層と境界の型を検査する。

kernel replayの時間切れを切り分ける追測では、`--profiles ncu --profile-from` に `--ncu-replay application` を明示できる。同じplan・入力・指標でアプリケーションを再実行し、各パスでkernelの名前、グリッド、ブロック、context、streamを厳密に照合する。結果にはreplay方式を記録する。 [NVIDIAのreplay仕様](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay)

INT8の正式カウンタは未採取。v6はkernel replayを45分、同じplanを再利用したv7は12分で打ち切り、いずれも判定時点でrawのkernel行が0件だった。v6のCSVには終了判定より約98分後に585行が書き出された。v7のapplication replay診断は同じplan・入力で585カーネルを取得し、nsysと一致した。時間内に完了したkernel replayを正式値の条件とし、両方のCSVを正式集計から外した。v7の終了値3はkernel replayの失敗を示す。 [打ち切り・診断の記録](attempts.json)・[正式な条件別集計](metrics.json)

## CPU確認

```bash
vit=knowledge/runs/run-i2303-vit-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
PYTHONPATH=".:$shared:$bundle:$vit:$gvhmr" .venv/bin/python -B -m pytest -n0 \
    "$shared/test_protocol.py" "$bundle/test_dino_protocol.py" \
    "$vit/test_vit_benchmark.py" "$gvhmr/test_gvhmr.py" \
    tests/unit/submodules/models/dino tests/unit/tasks/player_association/test_appearance_sampling.py
```
