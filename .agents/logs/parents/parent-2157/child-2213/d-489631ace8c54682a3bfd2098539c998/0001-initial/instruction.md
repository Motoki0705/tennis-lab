# #2213 取り出し親の完了条件を完了判定で読む

担当Issue #2213 の本文（目的・変更内容・対象外・Acceptance checklist）を正本として実装してください。親は #2157 です。

実例: #2157 の本文「全体の完了条件」は `### I-1878-merge-gate-api-scope` の下に12項目のチェックリストを持ちます（`improvement_proposals._extraction_body` が作る形式）。この本文で `parent status` の `completion.requirements` が `available=false`（`acceptance snapshot is not an explicit checklist`）になります。

要件:
- 取り出し形式の候補見出しだけを受け付け、それ以外の見出し・自由文は従来どおり拒否する（静かなフォールバックを作らない）。
- 通常形式の判定・IDの振り方は変えない。取り出し形式でのIDの振り方は、候補をまたぐ同じ文言を区別でき、合意snapshotとの変更検出が働く方式にし、READMEの該当箇所（`.agents/ops/parent/improvements.md` の「今回の受入と候補整理」）に必要なら1〜2行で追記する（二重管理しない）。
- 既存の取り出し親の本文と合意snapshotは書き換えない。
- テストは tests/README.md の方針で設計してから書く。`improvement-extract` が実際に作る本文を使うe2eで、available=true と finish の受理を確認し、修正を外すと失敗することを確かめる。

完了したらPRを作成してください（本文は日本語、`Closes #2213`、`References #2157`）。PR作成後はCIを待たずに報告して終了してください。レビュー・CI待機・mergeは親が新しい自動進行で行い、CI失敗時は自動進行があなたのsessionを修正ターンで再開します。
