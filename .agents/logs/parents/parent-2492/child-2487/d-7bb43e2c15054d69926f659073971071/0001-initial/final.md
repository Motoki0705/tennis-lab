RT-DETRv4-L/X の検出アダプターを `codex/parent-2492-issue-2487-rtdetrv4` の `a65f910e4ac9109edeba35b7edc18bee89fa5c79` にコミットしました。共通 JSONL、疑似GT採点器、速度計測ハーネスへ接続し、公式ソースの固定版と重みの利用条件を記録しました。[実装と取得手順](https://github.com/Motoki0705/tennis-lab/blob/a65f910e4ac9109edeba35b7edc18bee89fa5c79/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.md)

新規4件を含む26件のテストが通り、pre-commit の Ruff・mypy も成功しました。公式重みの取得、実フレームでの CPU 推論、ONNX 変換を L/X 別のローカルジョブに登録済みです。完了後に重みの SHA-256 と変換結果を確認します。GPU での精度・速度・VRAM と PT32 との差分は未計測で、速度計測は #2496 の固定入力の公開待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T113216Z_8364d66c2532_rtdetrv4-l-prepare","20261124T113216Z_ef583d5731c4_rtdetrv4-x-prepare"]}}