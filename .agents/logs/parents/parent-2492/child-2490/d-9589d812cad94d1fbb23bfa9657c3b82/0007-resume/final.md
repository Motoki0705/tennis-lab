PMPose-L/HのPT16・TRT16は、batch 1・4・8の全12条件で計測できました。3,600反復の入力順と集計値を照合し、重みとTensorRT engineのSHA-256が精度計測時と一致することを確認しました。parityは「PT32未計測」です。[計測表と再現手順](https://github.com/Motoki0705/tennis-lab/blob/67c3a65df8e987c71d4cdb18d5db96919d5b7b05/knowledge/runs/run-meiji-topdown-pose-comparison/PMPose.md%23synchronized-speed-results) · [結果と成果物のSHA-256](https://github.com/Motoki0705/tennis-lab/blob/67c3a65df8e987c71d4cdb18d5db96919d5b7b05/knowledge/runs/run-meiji-topdown-pose-comparison/pmpose_speed.json)

batch 1のL TRT16は、GPU P50が9.77 ms/人、前後処理込みのP50が30.98 ms/人でした。ViTPose-HのflipありTRT16はそれぞれ9.28、12.64 ms/人でした。batch 8のTRT16では、Lが45.37人/秒、Hが39.05人/秒でした。

結果を `codex/parent-2492-issue-2490-pmpose` の `67c3a65df8e987c71d4cdb18d5db96919d5b7b05` にコミットしてpushしました。64テストが13.36秒で通過し、knowledge検査はエラー・警告ともに0件でした。統合branchとのマージ競合もありませんでした。push後のCIは起動済みで、結果は未確認です。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45615597800)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}