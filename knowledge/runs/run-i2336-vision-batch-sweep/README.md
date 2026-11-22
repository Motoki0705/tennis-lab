# ViTPose-H・HMR2・DINOのバッチサイズ掃引

[#2336](https://github.com/Motoki0705/tennis-lab/issues/2336) の再現手順。計時、Nsightの範囲、精度の指標は[共通プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md)に従う。[#2335の共通バッチ処理](../run-i2335-trt-batch-sweep/README.md#2336での再利用)を使い、同じ固定入力のPyTorch FP32 B=1を全ケースの精度比較の参照とする。

| モデル | 固定入力 | 条件 | バッチ | TensorRT min / opt / max | ncuの代表 |
|---|---|---|---|---|---|
| ViTPose-H | #2303の236 crop | 0 / 1 / 2 | 1, 4, 8, 16, 32 | 内部2 / 32 / 64 | 16 crop。現行パイプラインと同じ |
| HMR2 | #2303の40 crop | 0 / 1 / 2 / 3a / 3b | 1, 4, 8, 16, 32 | 1 / 16 / 32 | 16 crop。現行パイプラインと同じ |
| DINO 4-scale Swin-L | #2304の64画像、750×1333 | 0 / 1 / 2 GridSample / 2 plugin | 1, 2, 4 | 1 / 4 / 4 | 方式・条件ごとに収まる最大のバッチ |

ViTPose-Hは水平反転したcropをGPUで連結し、ヒートマップを反転復元して平均する。throughputは元のcrop数で数える。ViTPose-Hの端数はB=8で4件、B=16/32で12件。HMR2のB=16/32は8件。端数は実サイズで推論し、計時では別区分にする。全画像・cropをmanifest順に精度比較する。Nsightは先行runの代表を含む、そのサイズの完全バッチを計測する。

HMR2の3aは `attention-out-max`、3bは `linear-max`。`clip_027` の固定32 cropをB=1で校正する。DINOのONNX exportではSwinのwindow復元のバッチ次元を記号化し、#2304の各MSDA実装を使う。TensorRTではmin・opt・maxと実行shapeのinspectorを照合する。B=0と最大B+1の拒否をログで検査してから計測へ進む。

## 実行

[#2297のTensorRT専用環境](../run-i2297-clipreid-tensorrt/README.md#専用環境)に、[#2303](../run-i2303-vit-tensorrt/README.md#実行環境と固定入力)と[#2304](../run-i2304-dino-tensorrt/README.md#再現環境)の固定依存を使う。`vit_inputs` は展開済みのcrop 2モデルの親、`dino_inputs` はDINOの展開済み入力。`TENNIS_LAB_DINO_EXTENSION` は#2304が使った固定binaryの絶対パス。submoduleはworktree内で初期化する。全jobをworktreeのrootから共有queueに登録する。

```bash
bundle=knowledge/runs/run-i2336-vision-batch-sweep
core=knowledge/runs/run-i2297-clipreid-tensorrt
vit=knowledge/runs/run-i2303-vit-tensorrt
dino=knowledge/runs/run-i2304-dino-tensorrt
sweep=knowledge/runs/run-i2335-trt-batch-sweep
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/trt-batch-i2336
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"

enqueue() {
    local label=$1 model=$2 inputs=$3 limit=$4 destination=$5 command
    shift 5
    printf -v command '%q ' timeout --foreground --kill-after=30s "$limit" env \
        "PATH=/usr/local/cuda/bin:$PATH" "PYTHONPATH=$PWD:$core:$vit:$dino:$sweep" \
        "TENNIS_LAB_DINO_EXTENSION=$TENNIS_LAB_DINO_EXTENSION" \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "$trt_exp_env/bin/python" -B -X faulthandler "$bundle/sweep_vision.py" \
        --model "$model" --inputs "$inputs" --output "$destination" "$@"
    bash "$queue" add "$command" --name "i2336-$label" \
        --resource all --issue 2336 --provider codex --session "$CODEX_THREAD_ID"
}

enqueue vitpose-timing vitpose_h "$vit_inputs/vitpose_h" 45m "$out/vitpose_h"
enqueue hmr2-timing hmr2 "$vit_inputs/hmr2" 90m "$out/hmr2"
enqueue dino-timing dino_4scale_swin_l "$dino_inputs" 75m "$out/dino_4scale_swin_l"
enqueue vitpose-ncu-0 vitpose_h "$vit_inputs/vitpose_h" 42m "$out/vitpose-ncu-0" \
    --profile-from "$out/vitpose_h/c0-b16" --ncu-timeout 2400
```

`sweep.json` と `logs/` は変換、範囲外拒否、計測の終了値を記録する。各ケースの `result.json` は入力・参照・planの識別子、精度とnsys、`measurement.json` は速度とVRAM、`layers.json` は実行shapeのinspectorを保存する。`--profile-from` は同じplan・入力順のnsys結果とncuの件数を照合する。VRAMのガードは#2304と同じく起動時の空き12 GiBとPyTorch allocator上限60%を使う。このrunではTensorRT contextの確保前と計測時のNVMLサンプルで空き3 GiBも確認する。失敗したjobを再試行する場合は出力先を分け、使う条件・バッチ・変換成果物をCLIで指定する。

### 初回出力からの再開

ViTPose-Hは45分のジョブ上限にB=32のFP16計測中に達した。完了済みのFP32 B=32だけで約14分26秒を使ったため、残るPyTorch B=32とTensorRTを独立したジョブにする。HMR2のFP16とDINOのpluginはTensorRT inspectorまで進んだ。HMR2ではpacked K/Vを出したFP16 GEMMとの対応と、単一トークンのvalue積の軸を照合する。DINOでは同じマスクの2視点と、固定のウィンドウ形状を持つQ・K・Vを照合する。どちらも初回のtyped ONNXを使用する。

DINO GridSampleの初回はworkspace上限512 MiBで約948 MiBを要求したtacticが除外された。再試行は同じグラフ・profileで `--workspace-mib 1536` を明示する。HMR2の3aはModelOpt 0.48.0のFP16変換で、単一トークンを除くONNX `If` の分岐処理に失敗した。固定のdecoderではトークン軸が1なので、この出力を `Squeeze` に置換して同じ設定で校正する。3bは準備途中に中断した。 [初回のログと終了値](attempts.json)

再開時は `--condition` で条件を選び、初回の `c0-b1` を `--reference` に指定する。ViTPose-HのFP16は `--condition 1 --batch 32`、HMR2のFP16とDINO各方式は、保存済みtyped ONNXのディレクトリを `--build-from` で指定する。ncuは各条件のnsysが記録されたケースを `--profile-from` に指定する。

```bash
retry="$out/retry-2"
enqueue vit-b32 vitpose_h "$vit_inputs/vitpose_h" 14m "$retry/vit-1" \
    --condition 1 --batch 32 --reference "$out/vitpose_h/c0-b1"
enqueue vit-trt vitpose_h "$vit_inputs/vitpose_h" 18m "$retry/vit-2" \
    --condition 2 --reference "$out/vitpose_h/c0-b1"
enqueue hmr2-fp16 hmr2 "$vit_inputs/hmr2" 16m "$retry/hmr2-2" \
    --condition 2 --reference "$out/hmr2/c0-b1" --build-from "$out/hmr2/graph-2"
for condition in 3a 3b; do
    enqueue "hmr2-$condition" hmr2 "$vit_inputs/hmr2" 20m "$retry/hmr2-$condition" \
        --condition "$condition" --reference "$out/hmr2/c0-b1"
done
dino_initial="$out/dino_4scale_swin_l"
enqueue dino-grid dino_4scale_swin_l "$dino_inputs" 27m "$retry/dino-2" \
    --condition 2 --reference "$dino_initial/c0-b1" \
    --build-from "$dino_initial/graph-2" --workspace-mib 1536
enqueue dino-plugin dino_4scale_swin_l "$dino_inputs" 24m "$retry/dino-2-plugin" \
    --condition 2-plugin --reference "$dino_initial/c0-b1" \
    --build-from "$dino_initial/graph-2-plugin"
```

### ncuの時間切れからの再開

HMR2 INT8 B=16とDINO FP16両方式のB=4でkernel replayが時間切れになった。3件ともCSVにカーネル行が0件だったため、同じ計測ケースを `--ncu-replay application` で取り直した。パス間の照合には `all`・`strict` を使用し、元のnsysとの件数とグリッド・ブロック寸法の並びも一致した。B=1との比較には先行runのkernel replayを使用し、replay方式とTensorRTのprofileの違いを記録した。 [終了値とCSVの記録](attempts.json)・[Nsight Compute仕様](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

```bash
retry="$out/retry-2"
enqueue hmr2-ncu-3a-app hmr2 "$vit_inputs/hmr2" 22m "$out/retry-3/hmr2-ncu-3a" \
    --profile-from "$retry/hmr2-3a/c3a-b16" --ncu-replay application --ncu-timeout 1200
for condition in 2 2-plugin; do
    enqueue "dino-ncu-$condition-app" dino_4scale_swin_l "$dino_inputs" 22m "$out/retry-3/dino-ncu-$condition" \
        --profile-from "$retry/dino-$condition/c$condition-b4" --ncu-replay application --ncu-timeout 1200
done
```

## 集計の再生成

各ケースの元の `measurement.json`、`result.json`、予測、nsys SQLite、ncu CSVとエンジン記録から [metrics.json](metrics.json) を生成する。`artifact` は `outputs/trt-batch-i2336/` からの相対パス。計時の全標本とブロック別統計、精度のカメラ別集計、HMR2の次元別誤差、各実行shapeのinspectorは元の出力に残る。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$core:$vit:$dino:$sweep" \
    "$trt_exp_env/bin/python" -B "$bundle/summarize_vision.py" \
    --source "$out" --vit-inputs "$vit_inputs" --dino-inputs "$dino_inputs" \
    --output "$bundle/metrics.json"
```

## CPU検証

```bash
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$bundle:$core:$vit:$dino:$sweep:$gvhmr" \
    .venv/bin/python -B -m pytest -n0 --no-cov --durations=5 \
    "$core/test_protocol.py" "$core/test_batching.py" "$vit/test_vit_benchmark.py" \
    "$dino/test_dino_protocol.py" "$sweep/test_sweep.py" "$bundle/test_vision_sweep.py"
```
