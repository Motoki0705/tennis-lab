共有training queueへ、固定入力生成と現行モデル39条件の計測を4ジョブに分けて登録しました。全ジョブがGPUを排他的に使用します。

6 clipのsceneと既存TensorRTの定義を照合し、全画面96枚と身体24系列を選定しました。人物cropはPT32 DINOの検出boxをscene内の実観測と照合して生成します。ViTPoseのflipなしbatch 1には、保存済みONNXから専用planを構築します。CPUテスト17件が通過しました。

実行スクリプトのローカルHEADは `0ee6f8dde06c078b6ab40324f5416b68a9eaefc0` です。[PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497) のharnessはPython CIを通過済みです。固定入力のSHA-256、速度・VRAM・parityの実測値、既存値との照合はジョブ結果の受領後に確定します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204477064386375676_2026-11-24T12-22-15-613624323Z_1489359_person-speed-prepare.job","0-8204477064089669440_2026-11-24T12-22-15-910330559Z_1489427_person-speed-detection.job","0-8204477063810218746_2026-11-24T12-22-16-189781253Z_1489498_person-speed-pose.job","0-8204477063492839690_2026-11-24T12-22-16-507160309Z_1489567_person-speed-body.job"]}}