#2297は**受入6項目すべて達成**しました。`codex/trt-env-protocol` の HEAD `e78be9632` で `Python` と Knowledge CI が成功し、PR #2298のドラフトを解除しました。 [PR #2298](https://github.com/Motoki0705/tennis-lab/pull/2298)・[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44650496373/job/136014758806)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44650496017)

| 条件 | DRAM read B | DRAM write B | Long scoreboard % | Short scoreboard % | LG throttle % | MIO throttle % | Tensor Core active % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 PyTorch FP32 | 483,787,520 | 0 | 4.65 | 7.89 | 2.44 | 31.20 | 1.39 |
| 1 PyTorch FP16 | 191,208,960 | 0 | 11.84 | 3.94 | 1.36 | 1.62 | 19.17 |
| 2 TensorRT FP16 | 161,052,160 | 0 | 9.47 | 3.67 | 0.42 | 1.59 | 29.88 |
| 3a TensorRT INT8 | 85,307,392 | 0 | 12.29 | 6.56 | 0.46 | 3.25 | 22.98 |
| 3b TensorRT FP8 | 76,959,232 | 0 | 12.10 | 6.21 | 1.23 | 1.34 | 18.99 |

代表1推論の値です。stall率とTensor Core activeはカーネル時間で加重平均しました。`ncu` と `nsys` のカーネル数は全条件で一致しました。DRAM writeは全条件で実測0 Bで、別の256 MiB CUDA fillでは258,687,232 Bを取得し、書き込みカウンタの動作を確認しました。 [実測データ](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/metrics.json)・[照合記録](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json)

**律速の解釈。** PyTorch FP32では、計測したstall理由のうちMIOの命令待ちが最も大きく、31.20%でした。TensorRT FP16ではFP32比でDRAM readが66.7%減り、カーネル数は188から114、Tensor Core activeは1.39%から29.88%に変化しました。INT8・FP8はFP16から読込量をさらに47.0%・52.2%減らしましたが、long scoreboardが12%台に上がり、カーネル数も増え、同期込み中央値の差は小さくなりました。DRAM帯域の飽和を判定するには帯域利用率の確認が要ります。 [結果ノード](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/nodes/tennis_scene/000075-run-i2297-clipreid-tensorrt.md)

| 受入項目 | 達成の証拠 |
|---|---|
| 専用環境 | 固定73パッケージを新規venvに再導入し、`trtexec` とCC 12.0のINT8・FP8実行を確認。 [再現手順](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/README.md) |
| `ncu` / `nsys` 疎通 | Windows側の設定後、同じWSLで全指標を取得。カーネル数は188・166・114・117・139で一致。 [照合記録](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json) |
| 固定入力と分離 | 評価 `clip_000/037` の236 cropとPTQ `clip_027` の254 cropで、動画と収録元フレーム区間の重複0件。 [入力照合](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/tensor-lock.json) |
| 計測プロトコル | 5モデルの採取位置、計測範囲、精度指標、許容差草案を記録。 [プロトコル](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md) |
| CLIP-ReIDの全指標 | 速度・VRAM・GPU内部指標・単体精度を全5条件で記録。 [結果ノード](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/nodes/tennis_scene/000075-run-i2297-clipreid-tensorrt.md) |
| 失敗時の明示停止 | 無効精度は終了値2、無効ONNXは終了値1。列ずれを拒否するテストを追加。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/e78be9632b7e6a449bec1b46f866588ca7a6f443/knowledge/runs/run-i2297-clipreid-tensorrt/validation.json) |

追測の初回はTensorRTの詳細NVTXラベル内のJSONで `ncu` のCSV列がずれました。`ncu` 実行時だけ詳細レイヤーラベルを抑え、同じplan・固定入力・NVTX範囲で3条件を再収集しました。ローカルpytestは追加1件を含む10件が成功し、Ruffとknowledge検証も通過しました。 [TensorRTのNVTX設定](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Core/ExecutionContext.html)