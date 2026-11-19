# #2157 設計調査（コード変更なし・報告のみ）

#2157（レビュー〜mergeの簡素化と自動進行）の実装に入る前に、現状のコードを調べて設計案を報告してください。今回のターンではコードを変更・commitしないでください。

## 要件の正本

#2157 本文の「全体のスコープ」「全体方針・合意事項」(a)〜(h)・「全体の完了条件」12項目。人間との合意済みで、範囲・対象外は変更しない。要点:

- 厳密な一致確認（diff SHA-256の固定・postcheck・モデル一致検証・公開順序とdigest・GitHubコメントとローカル記録の照合・resolveの4項目証拠と確認パス完全一致・complete-evidence・base取込みのファイル単位継承・merge前後の全体取り直し）を削除する。
- 大局レビュー1回（approve か major指摘一覧）。再レビューは親AIが決め、既定なし。CI修正・競合解消では再レビューしない。
- 親AIのレビュー依頼1回で、CLIが 大局レビュー → CI待機 → 軽いgate → merge → merge後確認 を切り離したプロセスで進める。状態はprivate stateに保存し、中断後は同じ段階から再開する。
- CI失敗・gate停止は、PRを作成したworker（Codex）のsessionを再開して直させる。workerはpush後にターンを終え、CIはCLIが待つ。自動修正はCI修正とgate修正の合算で最大3回。needs-human と AGENT_AUTO_MERGE=0 は親AIへ、依存未mergeはCLIが待つ。gate停止時に自分で直すか親AIを呼ぶかはworkerが決める。
- 軽いgateは 手元確認 → CI → 競合・draft・head → needs-human → 依存 → `gh pr merge --match-head-commit` の順で、最初の不合格で止める。
- 親AIへの通知は、手動の親は `wait-worker` と同じ背景待機の形、dispatcher起動の親は既存の待機・再開。
- GitHub GETは対象家族・PR数に比例させ、repo全体のIssue一覧を取らない。403のrate limitはリセットまで待つ。

## 調べること

1. `.agents/ops/review/`（約6100行）と、`.agents/ops/parent/` のレビュー・merge関連入口（review-prepare / review-request / review-state / record-result 周辺）、`merge_gate.py`。何を削除し、何を残し、何を作り直すかをファイル・関数単位で示す。
2. 削除する厳密照合に依存している他の箇所（dispatcher・supervisor・development workflow・decision_logs・worker_logs・review-inspect・ops-report など）。削除で壊れるもの、互換読取が必要な既存記録。
3. 自動進行CLIの状態機械の案（段階・遷移・保存する状態・再開時の照合・二重merge/二重worker起動の防止）と、既存の `run-worker --resume`・`wait-worker`・共通ランナーをどう再利用するか。
4. CI待機の実装方法（問い合わせ間隔・ETagの条件付きGET・タイムアウト）と、rate limit待機。
5. workerへ渡す失敗情報（失敗job・ログ）と、workerが「自分で直す／親AIを呼ぶ」を返す方法（最終報告の形式）。
6. 子Issueへの分け方の提案。独立にmergeできる単位・依存順・各子の完了条件案。分けないほうがよい場合はその理由。
7. 合意内容と既存の設計原則（静かなフォールバック禁止、ドキュメントの二重管理禁止など）がぶつかる点、人間の判断が必要な未決事項。

## 報告

自由な書式で、上の1〜7に答えてください。根拠にしたファイル・関数・行を添えてください。推測と確認済みの事実を区別してください。
