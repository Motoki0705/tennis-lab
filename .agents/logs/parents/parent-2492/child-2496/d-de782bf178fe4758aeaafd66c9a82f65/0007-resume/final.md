PT16・TRT16の全22条件で計時が完了しました。6,600反復の入力ID順と重み・TensorRT planのSHA-256は初回と一致し、全条件のparityは未判定です。DINOのbatch 1のwall P50はPT16で165.69 ms、TRT16で94.79 msでした。 [速度・VRAM・初回との差](https://github.com/Motoki0705/tennis-lab/blob/30f3fcddf5fbdec85908fc5fcf65d7276729a354/knowledge/nodes/tennis_scene/000086-run-person-model-speed-low-precision.md)

固定入力728ファイルを再検証し、公開表と生データCSV 8件の再生成結果がバイト単位で一致しました。CPUテスト84件、pre-commit、knowledge検査も通過しました。PT32とDINO＋ViTPose-H合計は初回の暫定記録を保全しました。

`codex/parent-2492-issue-2496-primary` の `30f3fcddf5fbdec85908fc5fcf65d7276729a354` をpushし、実行証拠を更新しました。このHEADのCIは結果待ちです。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2497,"head":"30f3fcddf5fbdec85908fc5fcf65d7276729a354","resume_on_success":false}}