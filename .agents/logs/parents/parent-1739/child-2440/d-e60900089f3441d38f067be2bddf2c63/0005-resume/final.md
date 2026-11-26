`origin/main` を取り込み、`knowledge/summary.md` の競合を解消しました。main側の人物モデル比較と #2440 のダブルス幅の採用根拠を残しました。PR本文を更新してDraftを解除し、GitHub上で競合が解消されたことを確認しました。[PR #2450](https://github.com/Motoki0705/tennis-lab/pull/2450)

CPUで回帰テスト104件が通過しました（28.06秒）。選手選別、人物対応、FP32キャッシュ、新レシピとの接続を確認しました。対象ファイルのpre-commitも通過し、knowledge検査は751ノードでエラー・警告ともに0件でした。

ブランチ `codex/parent-1739-issue-2440-primary` の新HEADは `63e9730fa23a57d46e7df4d9d4fed9e9204e9f94` です。CIは実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2450,"head":"63e9730fa23a57d46e7df4d9d4fed9e9204e9f94","resume_on_success":false}}