[#2336のドラフトPR #2373](https://github.com/Motoki0705/tennis-lab/pull/2373)を作成しました。計測と集計は完了し、対象HEAD `c07f0ac7a5805a10e034b0fff406869c10b3f57d` のPython CI待ちです。

- 全52ケースの速度・throughput・Peak VRAM・nsys kernel数・単体精度を記録しました。TensorRT 6エンジンでprofileと層精度を確認し、範囲外12試行はすべて終了値1でした。 [metrics.json](https://github.com/Motoki0705/tennis-lab/blob/c07f0ac7a5805a10e034b0fff406869c10b3f57d/knowledge/runs/run-i2336-vision-batch-sweep/metrics.json)
- 代表バッチ12件のncuは、nsysとkernel数・グリッド・ブロック寸法の並びが一致しました。時間切れの3件は同じplan・入力のapplication replayで取得しました。先行DINOプラグイン版のB=1はnsysの記録があります。 [全ケースの表とB=1比較](https://github.com/Motoki0705/tennis-lab/blob/c07f0ac7a5805a10e034b0fff406869c10b3f57d/knowledge/nodes/tennis_scene/000080-run-i2336-vision-batch-sweep.md)
- 初回の変換・校正失敗と非ゼロ終了、GridSampleのworkspace変更を記録しました。CPUテスト86件、knowledgeテスト187件、Ruffとknowledge検証が通過しました。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/c07f0ac7a5805a10e034b0fff406869c10b3f57d/knowledge/runs/run-i2336-vision-batch-sweep/attempts.json)

E2EにはViTPose-HのTensorRT FP16 B=16と、HMR2のTensorRT FP16・FP8 B=16を候補として引き継ぎます。DINOのTensorRT両方式は全バッチで精度草案を超過し、E2Eの比較条件は現行のPyTorch FP16 B=1です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2373,"head":"c07f0ac7a5805a10e034b0fff406869c10b3f57d","resume_on_success":true,"next_step":"対象HEADのPython CI成功を確認し、PR #2373のドラフトを解除して最終報告する。"}}