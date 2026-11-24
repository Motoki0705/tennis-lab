Sapiens2 の評価利用と ONNX／TensorRT 変換への同意を確認し、本番採用前の3つの確認事項とともに記録しました。`codex/parent-2492-issue-2490-sapiens2` の `13f9ca058eeb4fb8879f2a5ac29ad0d33db61786` にコミット済みです。#2495・#2496 の最新 HEAD を含むことを確認し、CPU テスト43件が通過しました。 [ライセンス記録](https://github.com/Motoki0705/tennis-lab/blob/13f9ca058eeb4fb8879f2a5ac29ad0d33db61786/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2.md)

公式重み1B／5Bの取得と SHA-256 照合、Python 3.12 分離環境の構築と公式前処理の確認を、計3件のローカルジョブとして開始しました。完了後に、共有 GPU キューで5B FP32のCUDAロード試行と精度評価へ進みます。

速度計測には、#2497 の `d64962a28` 時点で未公開の固定入力 manifest、SHA-256、人物 crop の行インデックスが必要です。 [#2497 進捗](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T122655Z_02cca93c039e_sapiens2-runtime","20261124T122655Z_ee8741870b59_sapiens2-1b-weight","20261124T122655Z_c5756afdd2ff_sapiens2-5b-weight"]}}