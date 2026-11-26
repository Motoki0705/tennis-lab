RTX 5060 Tiで39条件・11,700反復を測定しました。24条件がparityを通過し、15条件は許容差超過で実行不能と記録しました。DINO PT32のbatch 1のforward中央値は268.15 msで、先行計測と一致しました。ViTPose-Hのflipありは3 backendの全batchでparityを通過しました。 [速度・VRAM・実行可否の表](https://github.com/Motoki0705/tennis-lab/blob/14ca88f8a5616087435b411f79d2b63214d887e2/knowledge/nodes/tennis_scene/000085-run-person-model-speed-baseline.md)

固定入力 `meiji_4cam/person-model-speed-v1` は全画面96枚、人物crop 384件、身体24系列です。728ファイルのSHA-256を再照合しました。ViTPoseのbatch 1は固定順の先頭300 crop（5 clip）を計時し、parityは6 clipの384 cropで確認しました。 [入力manifest](https://github.com/Motoki0705/tennis-lab/blob/14ca88f8a5616087435b411f79d2b63214d887e2/knowledge/runs/run-person-model-speed-baseline/inputs/manifest.json)

manifestのSHA-256は `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62` です。生計測値、環境、重みとengineのハッシュ、再現手順をknowledgeに保存しました。CPUテスト53件とknowledge検証が通過しました。

`codex/parent-2492-issue-2496-primary` の完全HEAD `14ca88f8a5616087435b411f79d2b63214d887e2` をpushしました。新HEADのCI結果を待っています。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2497,"head":"14ca88f8a5616087435b411f79d2b63214d887e2","resume_on_success":false}}