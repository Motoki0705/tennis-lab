#2336 の52ケースを計測し、PR #2373 の対象HEAD `7aa9afeff42d59e86a7f74a11fa3ddbe896c0f62` でPython CIの成功を確認しました。照会時点でPRはすでにドラフト解除済みで、mainのマージコミット `a99c7f038a2b2cc9bf9700a0d39d42e6305209fc` が記録されていました。`gh pr ready 2373` はPRがclosedのため終了値1でした。 [PR #2373](https://github.com/Motoki0705/tennis-lab/pull/2373) · [Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45013294698/job/137185842155)

### #2363 の初回設定に使う推奨バッチ

| モデル | 精度 | B | 中央値 / P95 (ms) | 件[local path] | Peak VRAM allocator / NVML増分 (MiB) | nsys kernel | FP32 B=1との精度差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ViTPose-H | PyTorch FP32 | **4** | 202.39 / 202.78 | 19.77 | 2,617 / 2,909 | 650 | xy P95 0.000031 px |
| ViTPose-H | TRT FP16 | **8** | 56.21 / 56.40 | 142.43 | 11 / 1,699 | 254 | xy P95 0.02469 px、最大0.303 px |
| HMR2 | PyTorch FP32 | **16** | 410.37 / 413.24 | 38.96 | 2,839 / 3,221 | 781 | 相対L2 P95 0.000001 |
| HMR2 | TRT FP16 | **16** | 55.07 / 55.35 | 290.65 | 9 / 1,597 | 325 | cosine P05 0.999994、相対L2 P95 0.003534 |
| DINO | PyTorch FP32 | **1** | 267.02 / 272.17 | 3.74 | 1,780 / 2,311 | 1,862 | IoU P05 1.000 |
| DINO | TRT FP16 GridSample | **1（E2E比較用）** | 74.98 / 75.30 | 13.33 | 13 / 1,627 | 574 | IoU P05 0.929859、score差 P95 0.017161 |

- **ViTPose-H** はFP32のB=4とTRTのB=8がそれぞれ計測中の最高throughputでした。現行と同じB=16に揃えて比較する場合も、19.63 / 141.19 crop/sと差は小さいです。Bは反転前のcrop数で、TRTへの入力は2Bです。
- **HMR2** は両精度でB=16が最高throughputで、B=32よりVRAMを抑えられました。
- **DINO** のFP32はB=4にしても3.82画像[local path]、NVML増分は5,771 MiBに上がります。GridSample版はB=4で16.39画像[local path]、全バッチでIoUとscore差の単体基準が未達でした。#2363では現行と同じB=1を明示したTRT比較で下流への影響を調べ、採否は全クリップの結果で判断します。 [全52ケースの表](https://github.com/Motoki0705/tennis-lab/blob/a99c7f038a2b2cc9bf9700a0d39d42e6305209fc/knowledge/nodes/tennis_scene/000080-run-i2336-vision-batch-sweep.md)

**設定への転用**は、FP32の `4 / 16 / 1` を初回値に使えます。横断バッチを有効にする場合は `batching.sizes` に指定します。TRTのViTPose-H B=8とHMR2 B=16は、#2362の検証済みエンジンが最大B=2なので再ビルドが必要です。測定時のmin/opt/maxはViTPose-Hが内部 `2/32/64`、HMR2が `1/16/32` です。DINO B=1は、既存の750×1333エンジンに入力shapeと識別情報が一致すれば使えます。 [#2362の設定・ビルド仕様](https://github.com/Motoki0705/tennis-lab/pull/2379) · [#2363](https://github.com/Motoki0705/tennis-lab/issues/2363)

**受入の証拠**として、全52ケースの計時・VRAM・kernel数・単体精度、代表12ケースのncuとnsysの件数・起動形状の一致、6エンジンのprofile・層precision、範囲外12試行の終了値1を保存しました。TRT FP16のTensor Core activeはB=1から代表Bで、ViTPose-Hが53.72→63.56%、HMR2が31.05→67.65%、DINO GridSampleが26.20→26.02%でした。初回の変換・workspace不足は終了値とともに記録し、明示した条件で再実行しました。 [metrics.json](https://github.com/Motoki0705/tennis-lab/blob/a99c7f038a2b2cc9bf9700a0d39d42e6305209fc/knowledge/runs/run-i2336-vision-batch-sweep/metrics.json) · [試行記録](https://github.com/Motoki0705/tennis-lab/blob/a99c7f038a2b2cc9bf9700a0d39d42e6305209fc/knowledge/runs/run-i2336-vision-batch-sweep/attempts.json)

**未確認**は、複数モデルを使う全クリップ実行でのVRAM 16 GB・WSLメモリ40 GBへの収まり、総処理時間、追跡ID・人物対応・3D関節・身体配置・maskへの影響です。これらを #2363 で確認して設定値を確定します。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}