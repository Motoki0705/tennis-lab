# CLIP-ReIDとGVHMRのバッチサイズ掃引

#2335 / #2278 Phase 1。条件0（PyTorch FP32）、条件1（PyTorch FP16）、条件2（TensorRT FP16）を[共通プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md)で測る。各モデル・系列長の同じ入力を処理したPyTorch FP32 B=1を全条件の単体精度の参照にする。CLIP-ReIDは #2303 のPTQ設定で224 cropの許容差を満たした条件3a（`attention-out-max`）と3b（`linear-max`）も対象とする。校正は固定 `clip_027` の254 cropを1件ずつ使う。各エンジンが評価の全236 cropを処理し、単体精度の判定時に選定用12件をqueryとgalleryから外す。 [#2303の測定記録](../run-i2303-vit-tensorrt/metrics.json)

## バッチとプロファイル

| モデル | 固定入力と束ね方 | TensorRT min / opt / max | ncu代表バッチ |
|---|---|---|---|
| CLIP-ReID | #2297の236 cropをmanifest順に連続して束ねる。B=1, 4, 8, 16, 32 | B=1 / 16 / 32 | B=16。#2297の代表cropを含む先頭側の完全バッチ |
| GVHMR | #2305の同じ長さの4人物系列をmanifest順に束ねる。clip_037のL=145とclip_000のL=177を別々に測る。B=1, 2, 4 | L=145と177でそれぞれB=1 / 4 / 4を試行。B=2・4とLを固定したビルドも試行 | B=4。各評価クリップの全4人物を束ねた上限バッチ |

CLIP-ReIDのB=1と4は全バッチが指定サイズ。B=8の最後の1バッチは4件、B=16と32では12件。端数をそのまま推論し、全236件を順序どおり精度照合に使う。速度とthroughputは完全バッチを循環して計測し、端数も固定shapeの区分として30 warm-upと300反復で別に計測する。VRAMは完全バッチと端数を処理する同じプロセスのピークを記録する。nsysは各指定Bの完全バッチを使う。

GVHMRは同じ長さの4本をBで等分する。全フレームを保持したまま連結する。秒当たりの処理量は人物系列数とフレーム数を長さ区分ごとに記録する。精度評価は出力を人物ごとに戻し、条件共通のFP32復号とCOCO-17再構成を適用する。B=1固定のTensorRTは #2305 と今回で固定入力とFP32参照が一致したため、計測を再利用した。可変Bでは系列長も可変のprofileと固定長profileを試し、固定長profileで最適化レベル0も試した。固定B・固定Lでは全入力軸をONNXと照合して既定の最適化レベル3でビルドした。試した全設定でSIGSEGVとなり、FP16 B>1のplanは生成されなかった。 [ビルドの記録](validation.json)

可変バッチのTensorRT条件はモデル・系列長ごとに単一profileのplanをロードする。min・opt・maxの実行を先に確認し、B=0と最大B+1がTensorRTから拒否されることをログに保存する。GVHMRの可変profileでは範囲から外れるLも検査する。ビルド時の3端点と計測時の実shapeで主要演算のprecisionを照合する。ncuのkernel数は同じplan・入力で取得したnsysの件数と照合する。B=1のncu比較には#2297・#2303・#2305の記録を使う。TensorRTの比較は、profileの違いによるカーネルやタクティクスの差も含む。

## 再現

専用環境は[#2297のセットアップ](../run-i2297-clipreid-tensorrt/README.md#専用環境)と[#2305のGVHMR依存](../run-i2305-gvhmr-tensorrt/README.md#再現)を使う。`clip_inputs` は[CLIP-ReIDの展開済み入力](../run-i2297-clipreid-tensorrt/README.md#入力の展開)、`gvhmr_inputs` は[GVHMRの固定系列](../run-i2305-gvhmr-tensorrt/README.md#固定入力)のディレクトリ。入力と重みはそれぞれの既存tensor lockに照合する。

worktree rootで`clip_inputs` と `gvhmr_inputs` を設定し、共有training queueに速度・VRAM、精度、nsysを取得するjobを登録する。`sweep.json` の `verified` は各コマンドの終了値と、範囲外shapeの場合はTensorRTの拒否出力から決める。ncuはケースごとに独立したjobで、計測済みのケースを `--profile-from` に指定する。`timeout --foreground` で子プロセスをqueue管理下のプロセスグループに留める。再計測先は新しいディレクトリに置く。照合済みのCLIP FP32の速度・精度・nsysは `--reuse-case` を指定して再利用できる。 [初回の失敗と再計測対象](validation.json)

```bash
bundle=knowledge/runs/run-i2335-trt-batch-sweep
common=knowledge/runs/run-i2297-clipreid-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/trt-batch-i2335
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"

enqueue() {
    local label=$1 model=$2 inputs=$3 limit=$4 output=$5 command
    shift 5
    printf -v command '%q ' timeout --foreground --kill-after=30s "$limit" env \
        "PATH=/usr/local/cuda/bin:$PATH" "PYTHONPATH=$PWD:$PWD/$common:$PWD/$gvhmr" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "$trt_exp_env/bin/python" -B "$bundle/sweep.py" \
        --model "$model" --inputs "$inputs" --output "$output" "$@"
    bash "$queue" add "$command" --name "i2335-$label" \
        --resource all --issue 2335 --provider codex --session "$CODEX_THREAD_ID"
}
enqueue clipreid-timing clipreid "$clip_inputs" 45m "$out/clipreid"
enqueue gvhmr-torch gvhmr "$gvhmr_inputs" 40m "$out/gvhmr" --condition 0 --condition 1
for frames in 145 177; do
    enqueue "gvhmr-timing-l$frames" gvhmr "$gvhmr_inputs" 20m "$out/gvhmr-l$frames" \
        --condition 2 --fixed-length "$frames" --builder-optimization-level 0 \
        --reference-root "$out/gvhmr"
    for people in 2 4; do
        enqueue "gvhmr-static-b$people-l$frames" gvhmr "$gvhmr_inputs" 15m \
            "$out/gvhmr-b$people-l$frames" --fixed-length "$frames" --build-fixed-batch "$people"
    done
done
# ncuの登録例。各条件のB=16またはB=4の計測先を指定する。
enqueue clipreid-ncu-0 clipreid "$clip_inputs" 68m "$out/c0-b16-ncu" \
    --profile-from "$out/clipreid/c0-b16" --ncu-timeout 3900
```

CLIP-ReIDのタイムアウトからncuを再開するには、上の `--profile-from` に加えて、同一caseの時間切れ記録を `--ncu-prefix` に渡す。別々のreplayで得たprefixとsuffixが全launchを順番どおり覆うことをnsysで照合する。キャッシュとGPUクロックの設定は共通プロトコルを使う。

## 計測結果

[metrics.json](metrics.json) は速度・VRAM・nsys・単体精度と代表バッチの内部指標をまとめる。各ケースの `artifact` にraw計測の場所を、[validation.json](validation.json) にエンジンの拒否検査と失敗時の証拠を記録する。元のqueue出力をこのrunの構成で保持している場合は、次のコマンドで集計を再生成できる。

```bash
vit=knowledge/runs/run-i2303-vit-tensorrt
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$PWD/$common:$PWD/$gvhmr:$PWD/$vit" \
    .venv/bin/python -B "$bundle/aggregate.py" \
    --source outputs/trt-batch-i2335 --output "$bundle/metrics.json"
```

旧PTQのQ/DQ接続は、#2305の元の `outputs/trt-gvhmr` を指定してCPUで照合できる。保存された変換証跡とグラフのSHA-256が先行runの記録と一致することを確認してから検査する。 [旧PTQの検査結果](legacy-gvhmr-qdq.json)

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$PWD/$common:$PWD/$gvhmr" \
    "$trt_exp_env/bin/python" -B "$bundle/audit_gvhmr_qdq.py" \
    --artifact-root "$prior_gvhmr_outputs" --output "$bundle/legacy-gvhmr-qdq.json"
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

共通ディレクトリの `batching.batch_slices` でmanifest順と端数を保持し、`runtime.measure_latency` に固定shapeのバッチ列と `stats(batch=B)` を使う集計関数を渡す。TRTではモデルの実入力shapeのmin・opt・maxを `convert.build_engine(profile_shapes=...)` に渡し、`batching.require_engine_profile` でplanのprofileを照合する。`runtime.TensorRTContext` のshape設定とbinding、`profiles.capture` のNVTX範囲とkernel集計も共通。ViTPose-Hのflip testではエンジン入力軸は2Bで、throughputはB cropを数える。HMR2はcrop数をそのままBに使う。DINOは固定H/Wのbucketごとにバッチを作り、#2304の採用adapterを使う。許容したPTQ演算は変換後に `convert.verify_qdq_scope` で照合し、単体精度は同一入力のFP32 B=1と比較する。 [共通プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md)
