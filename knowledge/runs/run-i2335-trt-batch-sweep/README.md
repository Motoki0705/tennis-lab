# CLIP-ReIDとGVHMRのバッチサイズ掃引

#2335 / #2278 Phase 1。条件0（PyTorch FP32）、条件1（PyTorch FP16）、条件2（TensorRT FP16）を[共通プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md)で測る。各モデル・系列長の同じ入力を処理したPyTorch FP32 B=1を全条件の単体精度の参照にする。

## バッチとプロファイル

| モデル | 固定入力と束ね方 | TensorRT min / opt / max | ncu代表バッチ |
|---|---|---|---|
| CLIP-ReID | #2297の236 cropをmanifest順に連続して束ねる。B=1, 4, 8, 16, 32 | B=1 / 16 / 32 | B=16。#2297の代表cropを含む先頭側の完全バッチ |
| GVHMR | #2305の同じ長さの4人物系列をmanifest順に束ねる。clip_037のL=145とclip_000のL=177を別々に測る。B=1, 2, 4 | (B,L)=(1,145) / (4,145) / (4,177) | B=4。各評価クリップの全4人物を束ねた上限バッチ |

CLIP-ReIDのB=1と4は全バッチが指定サイズ。B=8の最後の1バッチは4件、B=16と32では12件。端数をそのまま推論し、全236件を順序どおり精度照合に使う。速度とthroughputは完全バッチを循環して計測し、端数も固定shapeの区分として30 warm-upと300反復で別に計測する。VRAMは完全バッチと端数を処理する同じプロセスのピークを記録する。nsysは各指定Bの完全バッチを使う。

GVHMRは同じ長さの4本をBで等分する。全系列を保持し、padding・切り詰めを行わない。秒当たりの処理量は人物系列数とフレーム数を長さ区分ごとに記録する。従来のopt長145を引き継ぎ、評価系列の範囲145–177を使う。精度評価は出力を人物ごとに戻し、条件共通のFP32復号とCOCO-17再構成を適用する。

各モデルの条件2は全バッチで同じ単一profileのplanをロードする。min・opt・maxの実行を先に確認し、B=0と最大B+1がTensorRTから拒否されることをログに保存する。GVHMRではL=144と178も検査する。ビルド時の3端点と計測時の実shapeで主要演算のprecisionを照合する。ncuのkernel数は同じplan・入力で取得したnsysの件数と照合する。B=1のncu比較には#2297と#2305の記録を使う。条件2はprofileが異なるため、カーネルやタクティクスの差も含む比較になる。

## 再現

専用環境は[#2297のセットアップ](../run-i2297-clipreid-tensorrt/README.md#専用環境)と[#2305のGVHMR依存](../run-i2305-gvhmr-tensorrt/README.md#再現)を使う。`clip_inputs` は[CLIP-ReIDの展開済み入力](../run-i2297-clipreid-tensorrt/README.md#入力の展開)、`gvhmr_inputs` は[GVHMRの固定系列](../run-i2305-gvhmr-tensorrt/README.md#固定入力)のディレクトリ。入力と重みはそれぞれの既存tensor lockに照合する。

worktree rootで`clip_inputs` と `gvhmr_inputs` を設定し、共有training queueに2件登録する。各jobでビルド、速度・VRAM、精度比較、プロファイラを個別プロセスで実行する。子プロセスの終了値とログ、境界検査、計測データは `--output` に残る。`sweep.json` の `verified` は各コマンドの終了値と、範囲外shapeの場合はTensorRTの拒否出力から決める。

```bash
bundle=knowledge/runs/run-i2335-trt-batch-sweep
common=knowledge/runs/run-i2297-clipreid-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/trt-batch-i2335
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"

enqueue() {
    local model=$1 inputs=$2 limit=$3 command
    printf -v command '%q ' timeout --kill-after=30s "$limit" env \
        "PATH=/usr/local/cuda/bin:$PATH" "PYTHONPATH=$PWD:$PWD/$common:$PWD/$gvhmr" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "$trt_exp_env/bin/python" -B "$bundle/sweep.py" \
        --model "$model" --inputs "$inputs" --output "$out/$model"
    bash "$queue" add "$command" --name "i2335-$model" \
        --resource all --issue 2335 --provider codex --session "$CODEX_THREAD_ID"
}
enqueue clipreid "$clip_inputs" 4h
enqueue gvhmr "$gvhmr_inputs" 5h
```

## CPU検証

再現のコマンドと同じ `bundle`、`common`、`gvhmr` を使う。端数とthroughput、B=1の参照、GVHMRの系列束ね、planのprofile読戻し、変換エラー時の非ゼロ終了をCPUで検証する。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$PWD/$common:$PWD/$gvhmr:$PWD/$bundle" \
    .venv/bin/python -B -m pytest -n0 --no-cov \
    "$common/test_protocol.py" "$common/test_batching.py" \
    "$gvhmr/test_gvhmr.py" "$bundle/test_sweep.py"
```

## #2336での再利用

共通ディレクトリの `batching.batch_slices` でmanifest順と端数を保持し、`runtime.measure_latency` に固定shapeのバッチ列と `stats(batch=B)` を使う集計関数を渡す。TRTではモデルの実入力shapeのmin・opt・maxを `convert.build_engine(profile_shapes=...)` に渡し、`batching.require_engine_profile` でplanのprofileを照合する。`runtime.TensorRTContext` のshape設定とbinding、`profiles.capture` のNVTX範囲とkernel集計も共通。ViTPose-Hのflip testではエンジン入力軸は2Bで、throughputはB cropを数える。DINOはH/Wが揃う入力を同じサイズのバッチ列にまとめる。
