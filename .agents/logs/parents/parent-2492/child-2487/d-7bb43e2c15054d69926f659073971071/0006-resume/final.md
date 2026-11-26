RT-DETRv4-L/X の4条件を固定6,844画像で再採点し、保存済みの結果と一致しました。

| モデル | Backend | Recall@0.50 | Recall@0.75 | バッチ1の全処理 P50 |
|---|---|---:|---:|---:|
| L | PT16 | 98.12% | 45.57% | 52.52 ms/画像 |
| L | TRT16 | 0.00% | 0.00% | 31.48 ms/画像 |
| X | PT16 | 97.97% | 44.66% | 56.13 ms/画像 |
| X | TRT16 | 0.03% | 0.00% | 32.48 ms/画像 |

速度8件は main と同一の計時コードで取得され、固定入力の順序・集計・VRAMを照合しました。TRT16 は両方とも実行できましたが、精度が大幅に低下したため比較対象外として記録しました。原因は未特定です。同一テンソルを用いる L/X の切り分け診断を GPU キューに登録しました。[条件別記録](https://github.com/Motoki0705/tennis-lab/blob/f409f60fab9681c92ee1ad2dc7883ec7dda1ff84/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json)

82件のテストと pre-commit が通過し、`f409f60fab9681c92ee1ad2dc7883ec7dda1ff84` を push しました。[CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45589965494)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204399464052865069_2026-11-25T09-55-35-947134930Z_2581290_i2487-rtdetrv4-l-trt16-diagnosis-v1.job","0-8204399463726207696_2026-11-25T09-55-36-273792303Z_2581382_i2487-rtdetrv4-x-trt16-diagnosis-v1.job"]}}