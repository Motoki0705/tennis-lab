RF-DETRの人物BBox adapterと評価CLIを `codex/parent-2492-issue-2487-rfdetr` の `64fde00dca465a7c2bb27881805f971d0bbaaee4` にコミットしました。公式重みの版・SHA-256・利用条件・再現手順は[実験記録](https://github.com/Motoki0705/tennis-lab/blob/64fde00dca465a7c2bb27881805f971d0bbaaee4/knowledge/runs/run-meiji-player-model-comparison/rfdetr.md)にあります。

- **RF-DETR-L**は実映像1枚で、10件の人物BBoxとscoreが公式APIと完全に一致しました。batch 1のFP16 ONNX生成、対象テスト23件、pre-commitのRuff・mypyが成功しました。
- 固定6,844画像のPT32・PT16採点と、TRT16のビルド・採点を共有queueに登録しました。GPU実測は結果待ちです。速度・VRAMの計測には #2496 の固定速度manifestが必要です。
- **RF-DETR-2XL**の3条件は、ユーザーによるPML 1.0への同意、ローカル評価を許すRoboflowアカウント・プランの確認、公式 `rf-detr-xxlarge.pth` のローカルパスとSHA-256が揃い次第進められます。[PML 1.0](https://github.com/roboflow/rf-detr_plus/blob/31474876e519dacef3ba320bc8977f1483d69b34/LICENSE)・[Roboflowアカウント](https://app.roboflow.com/)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204479781438562013_2026-11-24T11-36-58-561437986Z_1136476_i2487-rfdetr-l-pt32-accuracy.job","0-8204479781165493285_2026-11-24T11-36-58-834506714Z_1136517_i2487-rfdetr-l-pt16-accuracy.job","0-8204479780885618164_2026-11-24T11-36-59-114381835Z_1136558_i2487-rfdetr-l-trt16-accuracy.job"]}}