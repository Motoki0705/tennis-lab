**C1は未完です。** recorder実装とCourt停止例の実測を[draft PR #2154](https://github.com/Motoki0705/tennis-lab/pull/2154)へ保存しました。指定3clipと追加8候補がCourtで停止し、全モデル基準値・全cache hitは未達です。

branch: `codex/video-batch-baseline`  
HEAD: `643e1115c`  
基準main: `e561377ce`[local path]`572e36a03`

以下は**Court停止まで**の実測です。時間は秒、±は総wallの標本SDです。

| 条件 | 総wall | Court load/unload時間 | 回数 |
|---|---:|---:|---:|
| 初回cold | 337.038 | 49.306 / 2.814 | 7 / 7 |
| process-warm cold平均（n=2） | 254.274 ± 2.038 | 44.264 / 2.733 | 7 / 7 |
| 完成prefix再利用平均（n=3） | 103.705 ± 3.390 | 17.276 / 1.087 | 3 / 3 |

初回のdecodeは1.417秒、推論4.413秒、後処理185.749秒、保存0.006秒、公開0.546秒でした。最大値はPyTorch allocated **1,028.90 MiB**、reserved **1,176 MiB**、sampled RAM **3,562.11 MiB**、登録一時directory **11,940 bytes**です。

[実測・全内訳・再現手順](https://github.com/Motoki0705/tennis-lab/blob/643e1115ced5e36e8f2d009bf5d7c91d8506a340/knowledge/nodes/tennis_scene/000042-run-video-batch-baseline-20261116.md)と[固定入力manifest](https://github.com/Motoki0705/tennis-lab/blob/643e1115ced5e36e8f2d009bf5d7c91d8506a340/knowledge/runs/run-video-batch-baseline-20261116/input-manifest.json)を保存済みです。入力manifest SHA-256は `4d0013ff12b47f3a274d9501e5bb65476ae12755f1f176fb49dcb16f43668009`。

関連382テスト、pre-commit、最終HEADの必須`Python`・knowledge CIは成功しました。独立validatorは未指定、試行／完了0／0です。

**親判断が必要です：Court側の対処後に再測定するか、比較入力を見直すか。** 他モデルの実GPU経路と空検出例は未検証です。