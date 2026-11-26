PMPose-L/H × PT32・PT16・TRT16の全6条件で、固定26,276 BBoxの推論が完了しました。各条件でCOCO17の446,692関節がすべて有限で、23関節分のpresence・visibility・期待OKSも保存されています。TensorRTのビルドはLが85.09秒、Hが109.28秒でした。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/3101844995475b77eb9f7136ab197a400732b19a/knowledge/runs/run-meiji-topdown-pose-comparison/PMPose.md)

ブランチ `codex/parent-2492-issue-2490-pmpose`、HEAD `3101844995475b77eb9f7136ab197a400732b19a`。採点器とscene cacheの回帰を含む56テストが成功しました（10.99秒）。#2498の正準照合で再採点するCPUジョブを起動しました。OKS・PCKの変更有無は結果待ちです。速度・VRAMは#2497の修正版公開後に計測します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T235729Z_c92fccebb632_i2490-pmpose-rescore"]}}