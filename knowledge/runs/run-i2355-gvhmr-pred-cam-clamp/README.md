# GVHMR TensorRT FP16 B>1ビルド異常終了の切り分け

#2355 / #2278 Phase 1。#2335で、RTX 5060 Ti（SM120）とTensorRT 10.14.1.48.post1上の
GVHMR FP16 B>1ビルドが異常終了した（終了値 -11）。その最小範囲を切り分け、回避策を
試し、回避できた構成をB=1/2/4 × L=145/177で計測する。結果と考察はknowledge node
`run-i2355-gvhmr-pred-cam-clamp` に書く。この文書は再現手順とファイルの役割を示す。

## ファイル

| ファイル | 役割 |
|---|---|
| `rewrite_pred_cam.py` | `pred_cam` clampの `ScatterND` を `Slice` / `Clip` / `Concat` に等価書換えする |
| `rewrite_attention_mask.py` | exportされた `masked_fill`（`Where(Cast(Expand(mask)), -inf, scores)`）を等価に書き換える。`additive` は `scores + Where(mask, -inf, 0)` |
| `pin_precision.py` | 指定したattention演算を12 blockすべてでFP32計算に固定する（診断用。丸めが変わる） |
| `graph_slices.py` | 境界tensorを入出力にして部分グラフを切り出す |
| `variant_build.py` | 静的shapeの部分グラフを、指定のbuilder設定で1回ビルドする子プロセス |
| `diagnose.py` / `diagnose_block.py` | 実モデルの静的B=2グラフを、mask・embedding・block prefix、最初のblockの演算単位へ切り分ける |
| `diagnose_attention.py` | mask書換え2種と、非等価なmask除去3種を、attention境界で切ってビルドする |
| `diagnose_precision.py` | attention演算のFP32固定4種を同じ境界でビルドする |
| `campaign.py` | ScatterND最小グラフの比較、`pred_cam` 書換えのFP32照合、書換え後の動的planの作成 |
| `campaign_additive.py` | 加算maskのFP32照合とFP16ビット一致、動的plan、境界の拒否、#2335の計測 |
| `additive_precision.py` | 加算mask attentionとK=2線形層を含むplanの層精度検査 |
| `check_variant_accuracy.py` | 静的B=2の変種をONNX RuntimeでFP32グラフと比べる（CPU） |
| `summarize.py` | 診断出力の終了値を `diagnosis-summary.json` に集約する |
| `diagnosis-summary.json` / `ort-variant-accuracy.json` / `campaign-additive.json` | 記録した結果 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | TensorRT一次資料との照合 |

## 環境

- TensorRT・ONNX・ONNX Runtimeは専用環境 `$HOME/.venvs/tennis-lab-trt` を使う。共有
  `.venv` には入れない。
- GPUを使う処理（engine buildを含む）は、共有training queueの `--resource all` で実行する。
  `require_queue()` がqueue外の実行を止める。
- `gvhmr_inputs` は #2335 で固定した12系列。`fp32_reference_root` は同じGPUで測った
  #2335 の `c0-b1-l145` / `c0-b1-l177` を含むディレクトリである。

```bash
bundle=knowledge/runs/run-i2355-gvhmr-pred-cam-clamp
common=knowledge/runs/run-i2297-clipreid-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
trt_env="$HOME/.venvs/tennis-lab-trt"
export PYTHONPATH="$PWD:$PWD/$bundle:$PWD/$common:$PWD/$gvhmr"
export PATH="/usr/local/cuda/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

## CPUテスト

```bash
PYTHONPATH="$trt_env/lib/python3.11/site-packages:$PWD/$bundle:$PWD/$common" \
  .venv/bin/python -B -m pytest -n0 --no-cov $bundle/test_*.py
```

書換えの等価性（ONNX Runtimeでビット一致）、部分グラフの切り出し、builder設定の検査、
FP32固定の型、層精度検査の受理・拒否を確かめる。

## 再現手順

各コマンドはtraining queueのジョブとして実行する。静的グラフはCPUで作る。
`pred_cam` 書換え後の `concat.onnx` を診断に渡す。

```bash
CUDA_VISIBLE_DEVICES= "$trt_env/bin/python" -B $bundle/export_static.py \
  --inputs "$gvhmr_inputs" --batch 2 --length 145 --dtype fp16 \
  --output outputs/trt-gvhmr-i2355/exports/fp16-b2-l145
```

FP32 B=2とFP16 B=1も `--dtype` と `--batch` を変えて同じように作る。

1. ScatterND最小グラフと、書換え後の実モデル動的planを比べる。

   ```bash
   "$trt_env/bin/python" -B -X faulthandler $bundle/campaign.py \
     --inputs "$gvhmr_inputs" --reference-root "$fp32_reference_root" \
     --output outputs/trt-gvhmr-i2355/campaign-1
   ```

2. 実モデルの静的B=2グラフを、mask・embedding・block prefixで切る。

   ```bash
   "$trt_env/bin/python" -B -X faulthandler $bundle/diagnose.py \
     --source "$static_b2_l145_concat_onnx" --output outputs/trt-gvhmr-i2355/diagnosis-1
   ```

3. 最初のblockを演算単位で切り、FP16 B=2、FP32 B=2、FP16 B=1と、builder設定10種を比べる。

   ```bash
   "$trt_env/bin/python" -B -X faulthandler $bundle/diagnose_block.py \
     --source fp16-b2="$static_b2_l145_concat_onnx" \
     --source fp32-b2="$static_fp32_b2_l145_concat_onnx" \
     --source fp16-b1="$static_b1_l145_concat_onnx" \
     --probe-source fp16-b2 --output outputs/trt-gvhmr-i2355/diagnosis-block0-1
   ```

4. mask書換えとFP32固定を比べる。

   ```bash
   "$trt_env/bin/python" -B -X faulthandler $bundle/diagnose_attention.py \
     --source "$static_b2_l145_concat_onnx" --output outputs/trt-gvhmr-i2355/diagnosis-attention-1
   "$trt_env/bin/python" -B -X faulthandler $bundle/diagnose_precision.py \
     --source "$static_b2_l145_concat_onnx" --output outputs/trt-gvhmr-i2355/diagnosis-precision-1
   ```

5. 加算maskの動的planを検証して計測する。

   ```bash
   "$trt_env/bin/python" -B -X faulthandler $bundle/campaign_additive.py \
     --inputs "$gvhmr_inputs" --reference-root "$fp32_reference_root" \
     --output outputs/trt-gvhmr-i2355/campaign-additive-2
   ```

   段階ごとに非ゼロで止まる。FP32静的B=2で加算mask版と元グラフのONNX Runtime出力が
   ビット一致し、PyTorch FP32とrtol/atol 5e-4で一致すること。FP16動的グラフで
   B=1/2/4 × L=145/177のONNX Runtime出力が書換え前とビット一致すること。planの
   min/opt/maxで110個の主要演算がFP16で計算されること。B=0/5、L=144/178を
   TensorRTが拒否すること。

各ビルドは別の子プロセスで行い、異常終了（終了値 -11）後も次の比較を続けて記録する。
