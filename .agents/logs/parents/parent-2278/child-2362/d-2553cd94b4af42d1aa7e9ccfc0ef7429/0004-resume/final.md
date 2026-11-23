`Closes #2362` を記載した日本語のドラフトPRを作成しました。ブランチは `codex/parent-2278-issue-2362-primary`、HEADは `a9d254566` です。 [PR #2379](https://github.com/Motoki0705/tennis-lab/pull/2379)

DINO GridSampleは、今回のIoU P05が0.93056、score差 P95が0.01724で、単体基準の0.98以上、0.01以下をともに満たしませんでした。Phase 1でも同じ2項目が基準外だったことをREADMEとPRに記載しました。既定はFP32とし、TensorRT FP16は明示選択で残します。採用は #2363 の全クリップE2Eで判断します。再検証ジョブの終了値1と、GVHMRをPhase 1と同じカメラ座標の基準で再判定した経緯も記録しました。 [README](https://github.com/Motoki0705/tennis-lab/blob/a9d2545663ea5e412d1c8b41b5f4caf75f0c2f65/src/tennis_scene/pipeline/README.md#dino-gridsampleの単体精度)

関連CPUテスト57件が11.34秒で成功し、pre-commitとknowledge検証も通過しました。既知障害の修正 #2371 はmainにマージ済みです。対象HEADのPython CI成功後にドラフトを解除します。 [修正PR #2371](https://github.com/Motoki0705/tennis-lab/pull/2371)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2379,"head":"a9d2545663ea5e412d1c8b41b5f4caf75f0c2f65","resume_on_success":true,"next_step":"ドラフト解除と最終報告"}}