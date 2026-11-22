[ドラフトPR #2298](https://github.com/Motoki0705/tennis-lab/pull/2298) を作成しました。ブランチは `codex/trt-env-protocol`、HEADは `4c27391e4` です。対象HEADの [Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44644572466/job/135992711106) と [Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44644572470) は成功しました。

CLIP-ReIDの5条件で速度・VRAM・カーネル数・精度を実測しました。**受入は6項目中4項目を達成**し、残る2項目は `ncu` のカウンタ権限待ちです。

| 条件 | 中央値 / P95 ms | images/s | Peak VRAM NVML GiB / Torch MiB | DRAM R/W bytes | カーネル数 | Cosine P05 | FP32とのID一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 PyTorch FP32 | 5.606 / 5.828 | 171.8 | 1.532 / 369.75 | — | 188 | 1.000000 | 236/236 |
| 1 PyTorch FP16 | 4.792 / 15.748 | 167.7 | 1.339 / 206.02 | — | 166 | 0.999997 | 236/236 |
| 2 TensorRT FP16 | 1.364 / 4.014 | 538.5 | 1.308 / 0.19 | — | 114 | 0.999987 | 236/236 |
| 3a TensorRT INT8 | 1.336 / 2.381 | 526.4 | 1.193 / 0.38 | — | 117 | 0.489896 | 96/236 |
| 3b TensorRT FP8 | 1.485 / 2.411 | 535.7 | 1.189 / 0.38 | — | 139 | 0.988983 | 236/236 |

RTX 5060 Ti、B=1、各300回。NVMLはデバイス全体の同期点での最大観測値、Torchはallocatorの最大割当、カーネル数は代表1推論です。TRT FP16の中央値はFP32の約4.1倍速く、ID変更は0件でした。INT8の精度低下はTensorRTビルド前のQ/DQ ONNXでも再現しました。FP8はID一致を維持し、cosine P05は許容差草案の0.995を下回りました。 [実測と考察](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/nodes/tennis_scene/000075-run-i2297-clipreid-tensorrt.md)

| 受入項目 | 結果と証拠 |
|---|---|
| 専用環境 | **達成。** 新規venvで73パッケージを再導入。CC 12.0でINT8・FP8のQ/DQ行列積が期待値に一致し、`trtexec` でFP8 planを読み込めました。 [再現手順と環境出力](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/README.md) |
| `ncu` / `nsys` 疎通 | **未達。** `nsys` 2026.5.2で全5条件のGPUカーネルを取得。`ncu` は全条件で `ERR_NVGPUCTRPERM` を返し、DRAM・stall・Tensor Core指標は欠測です。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json) |
| 固定入力と分離 | **達成。** 評価 `clip_000/037` の236 cropとPTQ `clip_027` の254 cropを固定。動画と収録元フレーム区間の重複は0件で、再展開したテンソルのdigestも一致しました。 [入力照合](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/tensor-lock.json) |
| 計測プロトコル | **達成。** 5モデルの入力採取位置、shape、計測範囲、精度指標、許容差草案を記録しました。 [プロトコル](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md) |
| CLIP-ReIDの全指標 | **未達。** 5条件の推論と精度比較は完了し、残る計測は `ncu` 指標です。 [条件別データ](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/metrics.json) |
| 失敗時の明示停止 | **達成。** 無効精度は終了値2、無効ONNXは終了値1で停止しました。TRT FP16の初回はinspectorの `Half` 表記を拾えず停止したため、判定を修正して再計測しました。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/4c27391e4a95b84c6e1662d63032af54900f6e25/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json) |

ローカルpytestは新規6件と既存3件が成功しました。pre-commitは対象パスの規定でskipとなり、実験コードにはRuffを別途実行しました。

**親への依頼**は、Windows側のNVIDIA Control PanelでGPU性能カウンタへのアクセスを許可する管理者操作の手配です。許可後は、検証済みの `--profile-from` で既存planを再利用して追測できます。 [NVIDIAの手順](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

後続では `clip_027` をPTQ用に保ち、ViTPose-Hはflip込みのshape、HMR2は前処理後のcrop、DINOは独自opの実行経路、GVHMRは連続segmentとdynamic profileを固定してください。