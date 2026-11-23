#2336の全52ケースの速度・VRAM・nsys・精度を再計算し、TensorRT 6エンジンの層精度と範囲外入力の拒否12件を照合しました。全ケースをknowledgeに記録しました。 [runノード](https://github.com/Motoki0705/tennis-lab/blob/87893cc290baa8a1718bd34c9b0f2fdb1239598f/knowledge/nodes/tennis_scene/000080-run-i2336-vision-batch-sweep.md)・[metrics.json](https://github.com/Motoki0705/tennis-lab/blob/87893cc290baa8a1718bd34c9b0f2fdb1239598f/knowledge/runs/run-i2336-vision-batch-sweep/metrics.json)

- ViTPose-HのB=16はTensorRT FP16で中央値815.24→113.38 ms、座標誤差P95は0.0247 pxでした。
- HMR2のB=16はTensorRT FP8が316.49 crop/s、FP16が290.65 crop/sで、両方とも単体精度の草案を満たしました。
- DINOはPyTorch FP16の全バッチが精度草案に適合しました。TensorRTの両方式は全バッチでIoUとscore差の基準を外れました。

代表バッチのncuは9件でnsysとカーネル数・グリッド・ブロック寸法の並びが一致しました。時間切れの3件を同じplan・入力のapplication replayで再登録しました。CPUテスト86件とknowledge検証が通過しています。コミットは `87893cc290baa8a1718bd34c9b0f2fdb1239598f` です。追測結果の照合後に集計を確定し、PRを作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204664054813652569_2026-11-22T08-25-45-186347430Z_2321832_i2336-r3-hmr2-ncu-3a-app.job","0-8204664054536945634_2026-11-22T08-25-45-463054365Z_2321888_i2336-r3-dino-ncu-2-app.job","0-8204664054236205593_2026-11-22T08-25-45-763794406Z_2321944_i2336-r3-dino-ncu-2-plugin-app.job"]}}