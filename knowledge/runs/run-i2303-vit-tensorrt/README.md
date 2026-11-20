# ViTPose-H・HMR2 の TensorRT 比較

#2303 の再現手順。速度、メモリ、GPUカウンタ、精度の定義は [比較プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md) を参照。作業はリポジトリの worktree root から実行する。

## 実行環境と固定入力

[#2297 の専用環境](../run-i2297-clipreid-tensorrt/README.md#専用環境)を作成し、今回のモデルロードに必要な版を追加する。

```bash
shared=knowledge/runs/run-i2297-clipreid-tensorrt
bundle=knowledge/runs/run-i2303-vit-tensorrt
trt_exp_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/i2303
export PYTHONPATH="$PWD:$shared"
uv pip install --python "$trt_exp_env/bin/python" --no-deps -r "$bundle/requirements.txt"
uv pip check --python "$trt_exp_env/bin/python"
"$trt_exp_env/bin/python" -B "$bundle/fixed_crops.py" --output "$out/inputs"
```

ViTPose-H は [#2297 の検出行](../run-i2297-clipreid-tensorrt/inputs.json)に `pose_boxes`（拡張率1.2）を適用する。標準の `iter_frame_person_crops` で正規化した後、幅 `32:224` を切る。HMR2 は同じクリップ・候補フレームと保存済みの身体ビュー選択を照合し、実際のカメラ・box・フレームで `Hmr2FeatureExtractor` の `img` を再展開する。固定フレームのうち身体ビュー選択の間引き後に存在する行が対象で、元の body/segment のアドレスと関節の有効性を [body-boxes.json](body-boxes.json) に保存した。

| モデル | 評価 `clip_000/037` | 校正 `clip_027` | 推論入力 |
|---|---:|---:|---|
| ViTPose-H | 236 crop | 254 crop | 実crop B=1、反転を加えたモデル入力 B=2 × 3 × 256 × 192 |
| HMR2 | 40 crop | 32 crop | B=1 × 3 × 256 × 256、モデル内で幅を切る |

身体ビューの正本は `clip_000` が #1752 ダブルス比較の `comparison/main/store`、`clip_037` が #1753 向け HMR2 parity の `comparison/clip_037/separated_store`、`clip_027` が共有FP16比較の `comparison/runtime/store`。この順に `fixed_crops.py --pin-body-from STORE STORE STORE --output FILE` を実行すると body-boxes の再照合ができる。展開時に全カメラの媒体digestと非重複を検査し、[tensor-lock.json](tensor-lock.json) と一致した receipt を出す。精度とプロファイルの代表は両モデルとも `clip_000/cam0/frame 178/person 0` を使う。

## 計測

[共有キュー](../../../.agents/skills/training-queue/SKILL.md)の `--resource all` で登録する。下記の例は ViTPose-H の基準。HMR2 はモデルを `hmr2`、重みを `ckpt/hmr2/epoch=10-step=25000.ckpt` にし、出力と入力をモデル名に合わせる。実行種別は `0`、`1`、`2`、`3a`、`3b`。条件0を先に登録し、以降は `--reference "$out/$model/0"` を付ける。量子化条件は [プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md#vit-3モデルのptq設定) のとおり `--ptq-recipe` を指定する。

```bash
export TRAINING_QUEUE_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
queue=.agents/skills/training-queue/scripts/training_queue.sh
model=vitpose_h
printf -v command '%q ' env "PYTHONPATH=$PYTHONPATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$trt_exp_env/bin/python" -B "$shared/benchmark.py" \
  --adapter "$bundle/vit_models.py" --model "$model" --condition 0 \
  --checkpoint ckpt/vitpose/vitpose-h-multi-coco.pth \
  --inputs "$out/inputs/$model" --output "$out/$model/0"
bash "$queue" add "$command" --name i2303-vitpose-0 \
  --provider codex --session "$CODEX_THREAD_ID" --issue 2303 --resource all
```

ViTPose-H の計測区間は GPU上でのcrop反転・連結、モデルのforward、ヒートマップの左右入れ替えと反転復元、平均、同期まで。UDPの元画像座標復元は精度比較に使用する。HMR2 は SMPL head が出す1024次元の特徴までを計測する。HMR2 の FP16 ONNX export では、SMPL head が作成する定数ゼロtokenを特徴と同じ型に揃える。実行時の演算型と TRT の ONNX 演算ごとの精度照合を `result.json` に記録する。

## CLIP-ReID の切り分け

[#2297 の出力](../run-i2297-clipreid-tensorrt/README.md#照合できる計測記録)を展開済みの `--source` に指定する。固定評価順の20件ごとの12 crop を比較し、校正条件比較は `clip_027` の8件ごとの32 crop で行う。`--only ablation` は旧Q/DQグラフで活性化と定数の入力辺を識別し、種類・ブロックごとの寄与を調べる。`--only calibration` は量子化対象と校正方法を比較し、`--only sensitivity` は max 校正を固定して QKV・attention出力・MLP入力／出力への適用範囲を比較する。training queue に以下のコマンドを登録する。

```bash
env "PYTHONPATH=$PYTHONPATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$trt_exp_env/bin/python" -B "$bundle/clip_ptq.py" \
  --source outputs/trt-phase1a --output "$out/clip-ptq"
```

CLIP-ReID の追測は `benchmark.py --adapter "$bundle/vit_models.py" --model clipreid_vitb16_market1501` を使い、#2297 の `inputs` と基準FP32の `0` を参照する。

32 crop の校正比較では、デフォルト対象の max で cosine P05 が 0.526、MatMul/Gemm のみで 0.850、重み付き線形層のみの max で 0.849、entropy で 0.856 だった。旧グラフで活性化181辺だけをFP16へ戻すと 0.515 から 0.99974 に回復し、定数49辺だけを戻したときは 0.486 だった。LayerNorm入力だけを戻すと 0.749、LayerNorm・QuickGELUの積・残差Addの入力をまとめて戻すと 0.848 だった。重み付き線形層のうち、MLP出力投影だけを量子化した条件の0.860に対し、attention出力投影だけの条件は0.99971だった。3aには同じ12 cropで最も高い値を得た attention出力投影を使う。範囲の選定に使った12 cropは全236 cropでの比較にも含まれる。[校正・層別比較](int8-diagnosis.json)

[条件別の計測結果](metrics.json) と [HMR2の次元別絶対差](hmr2-feature-errors.json) を参照。実行精度検査、入力照合、プロファイラの失敗と追測は [validation.json](validation.json) に記録した。HMR2のINT8は元のplanと結果を `--profiles ncu --profile-from "$out/round3/hmr2/3a" --ncu-timeout 1800` で参照すると、同じ代表入力でGPUカウンタだけを追測できる。

## CPU検証

```bash
PYTHONPATH=".:$shared:$bundle" .venv/bin/python -B -m pytest -n0 \
  "$shared/test_protocol.py" "$bundle/test_vit_benchmark.py" \
  tests/unit/submodules/models/test_indexed_crops.py \
  tests/unit/submodules/models/vitpose/test_pose2d.py \
  tests/unit/submodules/vendor/gvhmr/vitpose/test_kp2d_utils.py
```
