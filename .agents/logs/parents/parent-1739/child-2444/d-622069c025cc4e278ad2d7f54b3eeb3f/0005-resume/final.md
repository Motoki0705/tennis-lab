予約未見3 clip・9カメラの38,421検出行を、開発用と同じViTPose設定で処理する計画に固定しました。CPUで実映像1フレームの3行を推論し、模擬出力では中断後に全9カメラを読み戻せました。関連pytest 23件、ruff、mypy、knowledge検査が通りました。人物ラベルの採点は0回です。

`video_001/clip_003` のcam0は、同じ映像の保存Court投影点から復元できました。投影点へのRMSEは3.13 pxで、校正上限11.01 px以内です。同じclipのボール注釈1,249フレームで、half-turnは `[false, false, true]`、次点との差は0.475でした。 [復元の証拠](https://github.com/Motoki0705/tennis-lab/blob/41b8e0ad7c0fad6e5eb2664b1b6305588ee76739/knowledge/runs/run-multiview-pose-link-recall-20261124/holdout-camera-recovery.json)

親には、この補完を採用して3 clipを一括採点するかの判断と、共有GPUロックの復旧をお願いします。workerは稼働中ですが、`[local path]` にアクセスできず、直近のジョブも `lock-namespace` で失敗しています。本推論は投入前です。

準備コードは `codex/parent-1739-issue-2444-primary` のローカルコミット `41b8e0ad7c0fad6e5eb2664b1b6305588ee76739` に保存しました。公開HEAD `5abec5a715d485ebd13c53112a9d764ba3eef08d` のCI成功を確認し、PR本文に今回の証拠を追記しました。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}