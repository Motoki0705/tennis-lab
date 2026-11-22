# GVHMR `pred_cam` clamp のTensorRTバッチビルド調査

#2355 / #2278 Phase 1。PyTorch legacy ONNX exporterが
`torch.clamp_min_(pred_cam[..., 0], 0.25)` を `ScatterND` に下げる最小グラフと、
同じ計算を `Slice` / `Clip` / `Concat` で表すグラフを比較する。GPU結果と結論は
queue campaign完了後に追記する。

TensorRT 10.14.1のrelease notesと公開Issueには同一条件の既知不具合は見つからず、
ScatterNDを含む過去のMyelin failureは条件が異なった。比較の根拠は
[一次資料との照合](KNOWN_ISSUES.md)に残す。

## CPU検証

```bash
trt_env="$HOME/.venvs/tennis-lab-trt"
bundle=knowledge/runs/run-i2355-gvhmr-pred-cam-clamp
trt_packages="$trt_env/lib/python3.11/site-packages"
PYTHONPATH="$trt_packages:$PWD/$bundle" \
  .venv/bin/python -B -m pytest -n0 --no-cov "$bundle/test_rewrite_pred_cam.py"
```

## GPU campaign

`gvhmr_inputs` は #2335 で固定した12系列、`fp32_reference_root` は同じGPUで
測定した #2335 の `c0-b1-l145` / `c0-b1-l177` を含むディレクトリである。
全GPU処理は共有training queueの `--resource all` で実行する。

```bash
bundle=knowledge/runs/run-i2355-gvhmr-pred-cam-clamp
common=knowledge/runs/run-i2297-clipreid-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
trt_env="$HOME/.venvs/tennis-lab-trt"
out=outputs/trt-gvhmr-i2355/campaign-1
PYTHONPATH="$PWD:$PWD/$bundle:$PWD/$common:$PWD/$gvhmr" \
  PATH="/usr/local/cuda/bin:$PATH" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$trt_env/bin/python" -B -X faulthandler "$bundle/campaign.py" \
  --inputs "$gvhmr_inputs" --reference-root "$fp32_reference_root" --output "$out"
```

campaignは次を別プロセスで記録する。

- 46ノードの最小ONNXについてB=1/2、FP16/FP32、builder最適化レベル0/3、
  ScatterND版とConcat版の終了値を比較する。
- 実GVHMRのFP32 B=2, L=145で、PyTorch・元ONNX Runtime・書換えONNX
  Runtimeの全5出力を照合する。
- B=1/4、L=145/177のdynamic profileを持つFP16 planを作り、engine inspectorで
  110個の主要演算のFP16をmin/opt/maxごとに検査する。
- B=0/5、L=144/178をTensorRT自身が非ゼロで拒否することを確認する。
- B=1/2/4 × L=145/177で速度、throughput、Peak VRAM、nsys kernel数、
  length・mask・復号後COCO-17関節差を #2335 と同じharnessで測る。

## 原因切り分けcampaign

ScatterND単体とScatterND除去後の実モデルを比べた後、実モデルの静的
B=2/L=145グラフをmask、入力embedding、`pred_cam` head、Transformer block prefixへ
切り分ける。各engine buildを別プロセスにし、SIGSEGV後も次の比較を記録する。

```bash
bundle=knowledge/runs/run-i2355-gvhmr-pred-cam-clamp
common=knowledge/runs/run-i2297-clipreid-tensorrt
gvhmr=knowledge/runs/run-i2305-gvhmr-tensorrt
trt_env="$HOME/.venvs/tennis-lab-trt"
PYTHONPATH="$PWD:$PWD/$bundle:$PWD/$common:$PWD/$gvhmr" \
  PATH="/usr/local/cuda/bin:$PATH" \
  "$trt_env/bin/python" -B -X faulthandler "$bundle/diagnose.py" \
  --source "$static_b2_l145_concat_onnx" \
  --output outputs/trt-gvhmr-i2355/diagnosis-1
```
