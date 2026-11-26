#2469 の修正版と `main` を取り込み、PR本文を更新してpushしました。#2469 のマージ後は #2461 固有の13ファイルだけが差分に残ることを確認しました。[PR #2461](https://github.com/Motoki0705/tennis-lab/pull/2461)

ブランチ `codex/parent-1739-issue-2459-primary`  
HEAD `2426d1f9242a51157259a16b51eb8c4667d15a48`

`source_identity.py` の競合は、比較側、追跡側の順に互換処理を適用して解消しました。`inventory.py` は両CLIの登録を保持し、FP32・FastHMRの回帰テストも通過しました。#2469 を含む想定mainの実ソースとキャッシュ識別が全16設定で一致し、対象ファイルを個別に編集した256条件ではすべて互換が失効しました。

関連pytest 66件、pre-commit、knowledge検査が成功しました。短時間と見込んだpytestは実測104.87秒でした。knowledgeの連番は104です。[新HEADのCIは実行中です。](https://github.com/Motoki0705/tennis-lab/actions/runs/45822671012)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2461,"head":"2426d1f9242a51157259a16b51eb8c4667d15a48","resume_on_success":false}}