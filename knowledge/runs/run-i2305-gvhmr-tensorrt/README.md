# GVHMR SIGA24の可変長TensorRT比較

#2305 / #2278 Phase 1-D。共通の[単体推論プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md)で、GVHMRの時系列denoiserをPyTorch FP32・FP16とTensorRT FP16・INT8・FP8で比較する。

## 固定入力

先行の #1757 の `serial-fb6255a-first-1/attempts/1-first` に保存された同一実行の3クリップを使った。DINOとViTPoseはFP16、HMR2とGVHMRはFP32で実行されたstore。各clipの `body_view_selection` と `hmr2_features` を既存のartifact codecで開き、GVHMR入力規約でperson・camera・source動画・各frameを照合した。全4方向の動画digestは #2297 のmedia pinと一致した。評価・PTQのクリップ、動画、同一cameraの収録元frame区間の交差は0件だった。

| split | clip | 人物IDごとの採用camera（0, 1, 2, 3） | clip内frame範囲 | 1系列の長さ |
|---|---|---|---|---:|
| 評価 | clip_000 | cam0, cam0, cam0, cam2 | 2–354 | 177 |
| 評価 | clip_037 | cam2, cam0, cam2, cam2 | 2–290 | 145 |
| PTQ | clip_027 | cam0, cam2, cam2, cam1 | 2–234 | 117 |

各人物が1系列ずつ。範囲は0始まりの両端を含む。約59.94 fpsの各clipから本番と同じ約30 fpsの格子で連続区間を採取した。系列に含まれるすべてのframeに観測があった。固定画像frame 178・146も採用viewの系列に含まれる。各person・camera・frameの識別子と全テンソルのshape・dtype・digestを [tensor-lock.json](tensor-lock.json) に保存した。

`GvhmrMeshRecovery._predict_impl` から `GvhmrDemoModel.predict` に渡る2D関節とconfidence、box、intrinsics、カメラ角速度、HMR2特徴、length、およびdenoiserへ渡る正規化済み条件を採取した。静止カメラ指定、キーポイント可視mask、系列長mask、局所attention maskも固定した。配列はworktreeの永続outputsに置く。

## Shapeと計算精度

[shape-profile.json](shape-profile.json) のB=1、min/opt/maxは117/145/177。117はPTQでのみ使用し、profileの最短での実行確認にも使う。評価で同数だった2 shapeのうち短い145をoptとした。145と177を別々に計る。各shapeの固定代表は両方とも人物ID 0で、145は `clip_037/cam2`、177は `clip_000/cam0`。

PyTorchはSIGA24のdenoiserをそのまま呼び出す。ONNXでは系列長120を超えるときの局所attentionを同じ行別ウィンドウのテンソル演算で表し、2種のattention einsumは等価なMatMulに変換する。不可視keypointのゼロ埋めは、可視maskの座標軸に長さ1を加えた `Where` に変換する。clipとpersonをまたぐ連結、入力の切り詰め、paddingは行わない。profileの全端点で `length` と `mask` を確認する。116と178の入力shapeはTensorRTのcontextで拒否する。

計測区間は正規化済みのdenoiser入力から3つのheadと系列長・maskの出力まで。入力転送と、共通FP32の元のGVHMR `Pipeline` によるSMPL-X復号・静止カメラ補正・IK・COCO-17再構成は区間の外に置く。ネイティブFP32の `Pipeline` 出力との照合も行う。条件0はTF32を無効にし、条件1は重みと浮動小数点の入力をFP16へ明示castする。

条件2では全110行列積をFP16で検査する。条件3a・3bは12ブロック内の96行列積（72線形・24 attention）を指定精度、入力・出力headの14行列積をFP16として検査する。正規化・Softmax・reduction・castに高い精度を許容する。TensorRTは `STRONGLY_TYPED`、profile付きの各実行shapeでEngine Inspectorを照合する。主要演算の精度の証拠が欠ける条件は非ゼロで停止する。 [TensorRTの動的shape仕様](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html)・[Engine Inspector](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

TensorRT 10.14の融合attentionはQ・K・2つのBool mask・Vを入力として表示する。Q・K・Vのdtypeと、2つのattention MatMulへの対応を検査する。INT8の線形演算と残差加算が融合した層では、FP16残差とINT8活性値の入力、ONNXのQ/DQの対応、タクティクスの両オペランドがINT8であることを照合する。融合層の内部命令ごとの精度はEngine Inspectorの観測範囲外。Engine Inspectorで動的な内部次元は `-1` と表示されるため、contextに実際に設定された入力shapeを別に記録する。 [検証記録](validation.json)

#2303のPTQ設定がmainに入る前のため、#2297と同じModelOpt ONNXのmax、ORT CPU、Q/DQ外FP16を採用した。ModelOpt標準の入力providerに、キャリブレーションshapeを明示して `clip_027` の実系列4本をB=1で順次渡す。FP8はE4M3。適用演算を上記のGVHMRのブロックに固定し、Q/DQと実行precisionを照合する。 [PTQ設定](ptq.json)・[ModelOptのcalibration provider](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.quantization.calib_utils.html)

## 再現

worktree rootで実行する。ベースは [#2297 の専用環境](../run-i2297-clipreid-tensorrt/README.md)。GVHMRの依存は [追加の固定requirements](requirements.txt) から同じ専用環境へ追加する。

```bash
bundle=knowledge/runs/run-i2305-gvhmr-tensorrt
common=knowledge/runs/run-i2297-clipreid-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
uv pip install --no-deps --python "$trt_exp_env/bin/python" -r "$bundle/requirements.txt"
uv pip check --python "$trt_exp_env/bin/python"
export PYTHONPATH="$PWD:$PWD/$common"
out=outputs/trt-gvhmr-reproduction
CUDA_VISIBLE_DEVICES="" .venv/bin/python -B "$bundle/pin_segments.py" --output "$out/inputs"
cmp "$bundle/tensor-lock.json" "$out/inputs/tensor-lock.json"
```

`pin_segments.py` の `--archive` に #1757 の保存済みdataset rootを指定できる。必要なbody artifactが欠けていれば停止する。`--dataset` と `--checkpoints` でmedia・重みの別配置を指定できる。

変換・計測は固定済みテンソルを検証してから実行する。初回7件の起動エラーは、入力検証が採取専用のシーン処理をimportしてHydraを要求したために発生した。終了値と修正内容を記録した。 [validation.json](validation.json)

次の変換・計測を共有training queueに登録する。1条件・1評価shapeごとに、速度とVRAM、精度、nsys、ncuのCUDAプロセスをそれぞれ分けて起動する。最後の境界検査は子プロセスの終了値とエラーを保存する。

```bash
queue=.agents/skills/training-queue/scripts/training_queue.sh
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
registration=(--resource all --issue 2305)
if [[ -n "${CODEX_THREAD_ID:-}" ]]; then
    registration+=(--provider codex --session "$CODEX_THREAD_ID")
fi

enqueue() {
    local label=$1 command
    shift
    printf -v command '%q ' env "PYTHONPATH=$PYTHONPATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 "$trt_exp_env/bin/python" -B "$@"
    bash "$queue" add "$command" --name "i2305-gvhmr-$label" "${registration[@]}"
}

for condition in 2 3a 3b; do
    enqueue "build-$condition" "$bundle/convert_gvhmr.py" --condition "$condition" \
        --inputs "$out/inputs" --output "$out/build-$condition"
done

for condition in 0 1 2 3a 3b; do
    conversion=()
    case "$condition" in
        2|3a|3b) conversion=(--conversion "$out/build-$condition") ;;
    esac
    for length in 145 177; do
        reference=()
        if [[ "$condition" != 0 ]]; then reference=(--reference "$out/case-0-$length"); fi
        enqueue "$condition-l$length" "$bundle/experiment.py" --condition "$condition" --length "$length" \
            --inputs "$out/inputs" --output "$out/case-$condition-$length" "${conversion[@]}" "${reference[@]}"
    done
    if (( ${#conversion[@]} )); then
        enqueue "boundary-$condition" "$bundle/experiment.py" --condition "$condition" \
            --inputs "$out/inputs" --output "$out/boundary-$condition" --boundaries "${conversion[@]}"
    fi
done
```

速度・精度だけを先に記録する場合は `--profiles none`。後から同一shapeの `--profile-from <計測先>` に対して別の `--output` を指定して追測できる。nsysを取得済みの場合、`--profiles ncu` で保存済みのnsysを参照してncuだけを追測する。kernel replayの制限時間は `--ncu-timeout` で指定できる。nsys と ncu のkernel数に差がある場合、差を記録して終了値3を返す。エンジン検査、変換、子プロセスのログとrawデータは各出力ディレクトリに保存する。

## CPU検証

動的attentionの境界と、実際のモデルブロックでのpadding mask一致、tensor lockによる入力変更の拒否、SO(3)回転角の単位と周期性をCPUで検証する。既存の共通harnessとbody pipelineの契約テストも実行する。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PYTHONPATH:$PWD/$bundle" \
    .venv/bin/python -B -m pytest -n0 --no-cov \
    "$bundle/test_gvhmr.py" "$common/test_protocol.py" \
    tests/unit/submodules/models/gvhmr/test_mesh_recovery.py \
    tests/unit/submodules/vendor/gvhmr/network/test_relative_transformer.py \
    tests/unit/tennis_scene/pipeline/components/test_gvhmr.py \
    tests/integration/tennis_scene/test_gvhmr_input_from_stored_selection.py
```
