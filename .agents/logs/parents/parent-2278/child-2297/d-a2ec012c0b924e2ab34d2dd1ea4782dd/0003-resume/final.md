正式レビューのmajor 4件を修正し、同じブランチ `codex/trt-env-protocol` にpushしました。PR #2298のHEAD `1f780773c23931b5128a89b33eb4925dc33c1fde` で `Python` と `Knowledge library` が成功しました。 [PR #2298](https://github.com/Motoki0705/tennis-lab/pull/2298)・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44656794748/job/136030999129)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44656794789)

1. **速度計測**では実行contextを `nvtx_verbosity=NONE` に設定し、読み戻した値を保存しました。training queueで同じ3 planを再計測し、各条件で236 cropの出力ハッシュが初回と一致しました。 [計測・照合記録](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json#L242)
2. **実行精度**は主要ONNX演算を実行層に対応付けて検査し、保存済みplanでFP16 74演算、INT8 73演算、FP8 72演算を確認しました。PTQでFP16を使う演算をプロトコルに明記し、低精度castとFP32 GEMMの組み合わせは `test_engine_inspection_rejects_fp32_gemm_with_low_precision_cast` で3条件とも拒否しました。融合attentionはQ/K/Vのdtypeを確認し、内部の命令精度は未確認です。 [プロトコル](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md#L12)・[回帰テスト](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/test_protocol.py#L92)
3. **ncu追測**では保存済みnsysと照合します。`test_ncu_followup_fails_when_reused_nsys_kernel_count_differs` で1件対114件なら終了値3、条件2の実機追測では114件対114件で終了値0を確認しました。 [回帰テスト](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/test_protocol.py#L187)
4. **DINOのbox照合**はIoU 0.50以上の対応だけで誤差を集計します。`test_dino_matching_counts_same_score_box_replacements_as_missing_and_extra` では、同スコアの別boxへの置換2件を欠落2件・追加2件と数えました。閾値はPhase 2で見直せる暫定値です。 [回帰テスト](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/test_protocol.py#L260)

| 条件 | 中央値 / P95 ms | 初回との差 ms | images/s | NVML peak GiB（idle差） |
|---|---:|---:|---:|---:|
| 0 PyTorch FP32 | 5.606 / 5.828 | — | 171.8 | 1.532（0.567） |
| 1 PyTorch FP16 | 4.792 / 15.748 | — | 167.7 | 1.339（0.374） |
| 2 TensorRT FP16 | 1.684 / 4.278 | +0.320 / +0.264 | 516.9 | 0.598（0.343） |
| 3a TensorRT INT8 | 1.394 / 2.170 | +0.059 / −0.211 | 663.3 | 0.483（0.228） |
| 3b TensorRT FP8 | 1.689 / 2.592 | +0.204 / +0.181 | 526.8 | 0.479（0.224） |

PyTorchは初回値、TensorRTは再計測値です。TensorRTのNVML絶対peakはidleと同じ0.710 GiB分だけ低下し、idleからの増分とTorch allocatorのpeakは各条件で初回と一致しました。latencyの差は計測時点の異なる観測値です。 [実測データ](https://github.com/Motoki0705/tennis-lab/blob/1f780773c23931b5128a89b33eb4925dc33c1fde/knowledge/runs/run-i2297-clipreid-tensorrt/metrics.json)

ローカルpytestは21件成功（11件増）。Ruffとknowledge検証も通過しました。pre-commitのruff・mypyは対象パス規定でskipされました。