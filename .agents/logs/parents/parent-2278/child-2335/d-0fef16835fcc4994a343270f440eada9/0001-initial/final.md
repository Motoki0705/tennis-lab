#2335の掃引harnessを `codex/parent-2278-issue-2335-primary` の `a12644c29c52dd9e56f7d8c59ed2028ef4037a1a` にコミットし、共有GPU queueに2件登録しました。

- CLIP-ReIDは固定236 cropを保持し、端数を実サイズで別に計測します。TRT profileはB=1/16/32、ncuはB=16です。
- GVHMRは145・177フレームを分け、同じ長さの人物系列を束ねます。TRT profileはB=1/4/4、ncuは両系列長のB=4です。全条件の精度を同一入力のFP32 B=1に照合します。 [再現手順と#2336向けの引き継ぎ](https://github.com/Motoki0705/tennis-lab/blob/a12644c29c52dd9e56f7d8c59ed2028ef4037a1a/knowledge/runs/run-i2335-trt-batch-sweep/README.md)

CPUテスト85件（追加7件、23.52秒）、Ruff、knowledge検証が通過しました。pre-commitのフックは対象パス外でスキップされました。#2303の現行ブランチとの仮統合では、共有harnessの `benchmark.py`、`convert.py`、`runtime.py` に競合を確認しました。

GPU実測、実機での範囲外拒否と層精度の照合、`metrics.json`・知見ノードの作成、PR作成はqueue結果の受け渡し後に行います。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204817422666730453_2026-11-20T13-49-37-333269546Z_1038617_i2335-clipreid.job","0-8204817422399039221_2026-11-20T13-49-37-600960778Z_1038674_i2335-gvhmr.job"]}}