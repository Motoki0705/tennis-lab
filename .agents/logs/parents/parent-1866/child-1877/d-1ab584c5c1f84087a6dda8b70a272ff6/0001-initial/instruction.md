## 今回のターン: 調査と設計案の報告（実装はまだ始めない）

担当issue #1877（親 #1866）の全Acceptanceを満たす実装に先立ち、最新mainのコードを読んで設計案を報告してほしい。親が設計を確認してから実装を指示する。

### 調べること
- 現行の改善候補記録（`parent improvement --child`、子テンプレートの「次の改善候補」）、手動終了（`release --ended`）、自律完了報告（`workflow-report`・development_workflow.md の終了規約）、合意・実行根拠（`agree`・`execution_basis`・`authorization`）、委任ゲート（`delegate`・`run-worker`・`start-worker`）、dispatcher の `human_hold` 判定、issues共通API（render・`ProposalSource`・`index_proposals`・`parent-task:v1` マーカーによる冪等起票）、本文補足確定待ち。
- #1796 の到達点（PR #1865・#1871 merged、PR #1876 open）の判断ログjournal・結果捕捉の契約と、本機能がそこへ何を書けば二重入力にならないか。
- #1804（親だけでの実行）の現状と、本機能が依存・共有する部分。

### 報告してほしいこと（日本語）
1. 候補の正本の保存先とschema案（安定ID、出典、根拠、親の判断、目的、次の作業の境界、判断・実行状態）。既存 `improvement --child` との関係。
2. CLI入口案（終了時の候補整理・候補なし記録、新親起票、承認・保留・見送り・撤回の記録）とコマンド名。手動・自律の両入口での接続点。
3. 新親の作成POST時点で `issue:parent` / `needs-human` を付ける方法と、出典マーカー（`ai-proposed` / `ops-report-proposal` を流用しない）。冪等・並行再送・POST後応答消失・既存open/closed/見送りへの集約の設計。
4. 承認ゲート: 未承認の新親で register/agree/delegate/run-worker/dispatcher/自律workerが起動しない仕組みと、明示承認の記録形式（人間とagentが同じGitHub loginを使う前提で、何を承認の正本とするか）。承認後に既存の合意・担当・実行経路へどう渡すか。2世代目が承認を引き継がない仕組み。
5. 受入未達の条件を候補へ移してcompletedにできない検査の方法。
6. #1796 判断ログへの接続点。
7. テスト計画（Acceptance各項目ごと、fake GitHubでの検証方法、既存テストの再利用）。
8. 独立して受け入れられる成果に分割する実益があるか。あるなら分割案、なければ1 PR（または順序付きの少数PR）の計画。
9. 人間の判断が必要な未決事項（無ければ無いと書く）。既存仕様と矛盾する点・リスク。

### 制約
- このターンではコードを変更しない（調査用の一時スクリプトはscratchに置き、コミットしない）。
- 作業時間の目安は最大90分。上記9項目を報告し終えたら終了してよい。それより前に止めてよいのは、設計を確定できない重大な前提の欠落を見つけた場合だけで、その場合は欠落の具体的内容と、親に判断してほしい選択肢を報告する。
