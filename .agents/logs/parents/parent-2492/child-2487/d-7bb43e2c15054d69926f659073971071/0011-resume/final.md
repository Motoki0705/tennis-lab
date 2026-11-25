DINO と DINO＋ViTPose の計測記録から、**PyTorch 4・OpenCV 1スレッド**を確認しました。同じ設定と同期込みの main harness を使い、RT-DETRv4 L/X の PT16 を固定96画像、batch 1・2、各300反復で測るジョブを2件投入しました。parity は「PT32未計測」です。 [再現手順](https://github.com/Motoki0705/tennis-lab/blob/a70a7fa24c23b3a68d108cfc1e483767b6d56e55/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.md#pt16-speed-with-matched-host-threads)

旧速度値は PyTorch 2スレッドの履歴として記録しました。既存の queue ログ8件は作業出力へコピーし、SHA-256 を照合しました。両 TensorRT 条件は「変換の不具合、原因未特定」として Pareto 計算から除外しています。 [計測・ログの記録](https://github.com/Motoki0705/tennis-lab/blob/a70a7fa24c23b3a68d108cfc1e483767b6d56e55/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4-results.json)

`codex/parent-2492-issue-2487-rtdetrv4` の `a70a7fa24c23b3a68d108cfc1e483767b6d56e55` を push しました。pytest 91件（追加1件、7.67秒）、pre-commit、knowledge 検証が通過し、CI を起動しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45642386776)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204380472820503651_2026-11-25T15-12-07-179496348Z_1537562_i2487-rtdetrv4-l-pt16-speed-threads4-1.job","0-8204380472575366699_2026-11-25T15-12-07-424633300Z_1537589_i2487-rtdetrv4-x-pt16-speed-threads4-1.job"]}}