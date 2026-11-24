[#2499](https://github.com/Motoki0705/tennis-lab/pull/2499) の `codex/parent-2492-issue-2487-primary` に `913f2c27dae88e57343988aa2523e7aa86a08f60` をpushしました。現行DINOとDINO＋ViTPose-Hを固定6,844画像で採点するPT32・PT16の2ジョブを共有queueに登録しました。

- #2496の現行速度6条件を照合しました。DINO＋ViTPose-HのPT32はバッチ1でP50が590.33 ms/画像、NVMLピーク増分が4,730 MiBでした。統合PT16・TRT16はPT32との出力一致基準を外れました。全コホート精度と有効な速度が揃った条件は現在0件です。[比較表](https://github.com/Motoki0705/tennis-lab/blob/913f2c27dae88e57343988aa2523e7aa86a08f60/knowledge/runs/run-meiji-player-model-comparison/conditions.csv)
- 代表6 clipの同一フレームでECPose-LのCPU PT32を現行経路と並べました。score 0.30以上の検出は各画像1〜2件で、疑似GTの採点対象は各4件でした。ECPoseの再変換後の実機結果と同期動画は未確認です。[clip_003の比較画像](https://github.com/Motoki0705/tennis-lab/blob/913f2c27dae88e57343988aa2523e7aa86a08f60/outputs/player-model-comparison/representative-cam0-ecpose-preview/clip_003-cam0.png)
- 対象pytest 43件、pre-commit、knowledge検証が通過しました。

#1739の下流はDINO専用の依存関係を検査するため、ECPoseを受け付けるには入力検査の拡張が必要です。依存PRの統合後に #2499 で対応するか、#1739側で先行対応するかを親に判断してほしいです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204468076511125596_2026-11-24T14-52-03-488874403Z_2687268_i2487-current-dino-vitpose-pt32-cohort.job","0-8204468076215471927_2026-11-24T14-52-03-784528072Z_2687337_i2487-current-dino-vitpose-pt16-cohort.job"]}}