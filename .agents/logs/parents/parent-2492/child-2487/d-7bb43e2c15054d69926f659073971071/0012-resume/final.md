RT-DETRv4 L/X の PT16 再計測は全4条件で完了しました。DINO と同じ RTX 5060 Ti、PyTorch 4・OpenCV 1スレッドで、固定96画像の入力順と各300反復の集計を照合しました。parity は「PT32未計測」です。

| モデル | Batch | Stage wall P50 / P95（ms/画像） | NVML ピーク増分（MiB） |
|---|---:|---:|---:|
| L | 1 | 53.67 / 60.03 | 338 |
| L | 2 | 38.55 / 42.08 | 468 |
| X | 1 | 58.16 / 73.34 | 570 |
| X | 2 | 44.80 / 62.33 | 696 |

比較用の `conditions[].speed` を更新し、旧2スレッドの値は `initial_speed` に保存しました。統合側の batch 1 読み込みも両モデルで確認済みです。新規2件を含むログ10件を保管し、SHA-256 を照合しました。TensorRT の変換不具合は原因未特定で、Pareto 計算の対象外です。 [計測レシートとログ](https://github.com/Motoki0705/tennis-lab/blob/d2706513d58b342444240d7a136455133c8fff25/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json) · [#2487 の検証記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7166696706)

`codex/parent-2492-issue-2487-rtdetrv4` の `d2706513d58b342444240d7a136455133c8fff25` を push しました。pytest 91件（7.96秒）、pre-commit、knowledge 検証が通過しました。CI は起動済みで、結果は未確認です。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45644062272)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}