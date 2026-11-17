# #2204 旧レビュー前提の共有memoryの整理

担当Issue #2204 の本文（目的・変更内容・対象外・Acceptance checklist）を正本として作業してください。親は #2157 です。

背景: PR #2184（#2176）で、レビュー〜mergeは `review-request` 1回からCLIが自動進行する流れへ切り替わり、旧レビュー経路（`parent_review.py run/resolve/recover/resume-publication/close-interrupted/complete-evidence`、`review-state`、digest照合、旧merge gateの全体収集）は削除されました。現行の仕様は `.agents/ops/review/README.md` と `.agents/ops/parent/README.md`（レビュー要求と自動進行）です。

要件:
- 共有memoryの変更は `.agents/skills/memory/SKILL.md` の更新・削除の手順に従ってください（削除はエントリと index.md の行を同じcommitで消し、理由をcommitメッセージに書く）。
- 各エントリについて、現行コードの該当箇所（または該当箇所が無いこと）を確かめてから、削除・書き直し・据え置きを決めてください。推測で書き換えないでください。確かめた部分だけ `last_verified` を進めます。
- `wsl2-review-isolated-cpu-probe-fails.md` は、隔離CPUテストのprobe（`lib/review_tools`）が今も同じ失敗をするか、新しいレビューで「使えない資料」がどう扱われるか（approve / major、実行失敗は承認へ変換しない）を確かめて判断してください。実際にprobeを実行して確かめられるなら実行してください。
- 変更後、memoryの形式検査（`tests/e2e/agents_ops/test_agents_docs.py` など）を実行してください。

完了したらPRを作成してください（本文は日本語、`Closes #2204`、`References #2157`。エントリごとの判断と根拠を本文に載せる）。PR作成後はCIを待たずに報告して終了してください。レビュー・CI待機・mergeは親が新しい自動進行で行います。CIが失敗した場合は、自動進行があなたのsessionを修正ターンで再開します。
