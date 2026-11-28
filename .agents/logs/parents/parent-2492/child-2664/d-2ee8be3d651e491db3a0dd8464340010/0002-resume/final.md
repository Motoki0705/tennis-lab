RF-DETR-L と RTMPose-X を、シーンパイプラインの検出・pose として独立に選べるようにしました。RF-DETR-L の初期閾値は `0.30`、RTMPose-X の confidence は生の SimCC スコアです。モデル・backend・重み・engine を成果識別に含め、既定の DINO＋ViTPose は旧キャッシュを再利用できることを確認しました。

- 公式 MMPose との CPU 照合は実画像2 box・34関節で通過しました。座標の最大差は `0.0000324 px`、confidence の最大差は `0.000000954` です。
- 選択した pytest 238件、pre-commit の ruff・mypy、`uv lock --check` が通過しました。全体の設定一覧は CI で確認します。 [検証記録](https://github.com/Motoki0705/tennis-lab/issues/2664#issuecomment-7205899214)
- #2470 を含む main を取り込みました。GPU の PT16・TRT16 照合と `video_000/clip_003` の4カメラ標準経路を、共有 queue の `all` 予約に登録しました。#2470 の3D追跡から公開までの通し確認は #2487 に残ります。

ブランチは `codex/parent-2492-issue-2664-primary`、実行 HEAD は `ae8e0daa5131ce77c67a10e8b768a3816fbbb69b` です。GPU 結果を受け取った次のターンで照合し、PR に実行証拠を記載します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204263006209488759_2026-11-26T23-49-53-790511240Z_3990778_i2664-people-models-qualification.job"]}}