# TensorRT一次資料との照合

2026-11-22に、TensorRT 10.14.1のrelease notesとNVIDIA/TensorRTの公開Issueを
`ScatterND`、`Myelin`、`buildSerializedNetwork`、`segmentation fault`で照合した。

- [TensorRT 10.14.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.14.1.html)
  のFixed Issues / Known Issuesには、ScatterNDまたは同じSIGSEGVの記載がない。
  Blackwell向けにはSlice-Fill-Convの`MyelinCheckException`修正があるが、今回の
  Gather / Clip / ScatterND融合とは演算も停止形態も異なる。
- [NVIDIA/TensorRT #3527](https://github.com/NVIDIA/TensorRT/issues/3527) は
  TensorRT 8.5.1でScatterNDのforeign-node最適化とdata-dependent shapeが両立しない
  internal errorを報告している。モデルが提供されず、NVIDIA担当者はScatterND自体の
  制約とは断定せず、inactiveでcloseされた。今回とMyelin・ScatterNDは共通するが、
  TensorRT版、GPU、エラー、shape条件が異なるため、同じ不具合とは確認できない。
- [NVIDIA/TensorRT #4840](https://github.com/NVIDIA/TensorRT/issues/4840) は
  Myelin compilation中のSIGSEGVを報告しているが、TensorRT 10.3 / Jetson Orin上の
  PointCNN（MatMul / Transpose / depthwise Conv）であり、今回の最小候補とは異なる。

したがって、公開一次資料に同じTensorRT 10.14.1 / SM120 / FP16 / B>1の
ScatterNDビルドSIGSEGVは見つからなかった。GPU campaignで小さいONNXへ切り出した
結果が同じ終了値を示す場合、そのONNX、builder設定、verboseログを新規報告へ渡せる。

## 切り分け後の範囲（2026-11-22追記）

診断の結果、異常終了（終了値 -11）は `ScatterND` ではなく、最初のblockのattentionで起きていた。
2つのBool maskとFP16 Softmaxを含むfused attentionで、B≥2のときに限られる
（[node](../../nodes/tennis_scene/000082-run-i2355-gvhmr-pred-cam-clamp.md)）。
この範囲では、release notesと公開Issueをまだ照合していない。新規報告に使えるのは
`diagnosis-attention-1/graphs/source-pv-from-qkv.onnx`（32ノード、静的B=2）と、その
builder設定・verboseログである。
