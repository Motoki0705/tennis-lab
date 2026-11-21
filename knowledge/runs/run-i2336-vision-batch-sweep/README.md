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

## CPU検証

```bash
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$bundle:$core:$vit:$dino:$sweep:$gvhmr" \
    .venv/bin/python -B -m pytest -n0 --no-cov --durations=5 \
    "$core/test_protocol.py" "$core/test_batching.py" "$vit/test_vit_benchmark.py" \
    "$dino/test_dino_protocol.py" "$sweep/test_sweep.py" "$bundle/test_vision_sweep.py"
```
