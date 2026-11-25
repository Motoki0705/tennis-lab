RTMPose・PMPoseの速度6ジョブを照合し、候補24条件とViTPose-H基準6条件の比較表・Pareto図を [draft PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) に反映しました。branchは `codex/parent-2492-issue-2490-primary`、commitは `67dcbc42bb7e83f55298152ee7c3d7ac11c0760f` です。

**速度とVRAMを重視する場合の下流検証候補はRTMPose-X TRT16です。** Batch 1で現行ViTPose-H flipあり TRT16比の処理時間はP50で28.7%、P95で10.1%、ピークVRAMは60.2%減りました。疑似GTの平均OKSは0.77777から0.76826へ0.95ポイント低下しました。RTMPose-M TRT16はbatch 8のスループットが最高で、PMPose-H TRT16は平均OKSが最高でした。 [比較表](https://github.com/Motoki0705/tennis-lab/blob/67dcbc42bb7e83f55298152ee7c3d7ac11c0760f/knowledge/runs/run-meiji-topdown-pose-comparison/comparison.csv) · [Pareto図](https://github.com/Motoki0705/tennis-lab/blob/67dcbc42bb7e83f55298152ee7c3d7ac11c0760f/knowledge/runs/run-meiji-topdown-pose-comparison/pareto-batch1.png)

7,200反復を再集計し、比較表と図のバイト一致を確認しました。対象pytest 106件が13.00秒で成功し、pre-commitとknowledge検査も通過しました。新規計測のparityは「PT32未計測」です。実DINO BBoxでの精度、欠測処理、3D coverage・再投影・選手IDへの影響は採用判断に向けた未検証事項です。CIは起動済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"67dcbc42bb7e83f55298152ee7c3d7ac11c0760f","resume_on_success":false}}