PMPose-L/Hの公式重みを取得し、両方のSHA-256とチェックポイント内のCOCO17順序を確認しました。MMPretrainが必要とするTransformers旧APIの互換版をPMPose専用環境に固定し、更新された共通速度harnessにも接続しました。 [再現手順](https://github.com/Motoki0705/tennis-lab/blob/0860ce5e12c953dc5520f02a502cc50cc2b3381b/knowledge/runs/run-meiji-topdown-pose-comparison/PMPose.md)

- ブランチ `codex/parent-2492-issue-2490-pmpose`、HEAD `0860ce5e12c953dc5520f02a502cc50cc2b3381b`。#2495・#2496の最新HEADを含むことを確認済みです。
- 更新後の共通harnessやscene cacheの回帰を含む37テストが成功しました（15.06秒）。Ruff、mypy、knowledge検査も通過しました。
- 依存環境の再検証、L/Hの公式APIとの実重みparity、ONNX出力を行うCPUジョブを起動しました。GPUでの精度・速度・VRAMは未計測です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T123234Z_03845a48192c_i2490-pmpose-parity-export"]}}