# #1987 実装依頼（担当Issue自身。子issueは作らない）

## 成果
正式レビューのモデル選択を通常設定で有効にする。1つのPR（`Closes #1987`、本文は日本語）にまとめる。

## 前提（既に実装済みの基盤）
- `.agents/ops/lib/agents.conf` の `REVIEW_SELECTION_MODE="fixed"`、許可組 `gpt-6.1-sol:xhigh,gpt-6-astra:xhigh`、固定値 `REVIEW_CODEX_MODEL="gpt-6-astra"`。
- 親CLIの `review-prepare` / `review-request --prs` は既に `--review-model` `--review-effort` `--model-reason-file` `--review-model-source human_instruction` `--human-model-reference` `--handoff-from-run` `--handoff-scope-file` を持つ（`.agents/ops/parent/README.md` の「レビュー要求契約」節、`.agents/ops/lib/README.md` の「レビュー専用の選択入力」節、`.agents/ops/review/README.md` の引継ぎ契約）。
- 比較根拠: `knowledge/reports/review-model-comparison-20261111/README.md`（当時の記録として残し、書き換えない）。

## やること
1. `agents.conf` の通常設定を `REVIEW_SELECTION_MODE="select"` にする。固定値 `REVIEW_CODEX_MODEL` の扱い（select時の意味、Astra固定へ戻す手順＝`fixed` に戻す）を確認し、整合させる。
2. selectで `--review-model` / `--model-reason-file` が必須になることで、手動・自律の親の通常手順（parent-session skill、`.agents/ops/parent/README.md`、`.agents/ops/parent/development_workflow.md`、`.agents/ops/review/README.md`、decision_logs README の3言語版など、`review-prepare`[local path]`review-request --prs` の例を載せている箇所）がモデル指定漏れで止まらないよう、例・手順を選択引数付きに揃える。自律親の呼出元（スクリプト・プロンプト・dispatcherが組み立てる指示など）があれば同様に揃える。指定漏れは黙って既定へ振替えず、明確なエラーで何を渡せばよいか分かるようにする（既存の挙動がそうなら確認だけでよい）。
3. 運用手順（正本は1か所。二重管理しない。parent-session skill からはリンク程度）に以下を記載する:
   - Sol / xhigh を選ぶ具体例（影響が局所的で期待挙動・確認方法が明確。例: 表示文言の修正）と Astra / xhigh を選ぶ具体例（公開範囲・権限・永続状態・並行処理・merge gate等に影響、または影響範囲不明。例: 1行の条件変更でも復旧後にどの結果を採用するかが変わる）。理由は1文。
   - 人間の明示指定は既存契約（human_instruction）に従うこと。
   - Solで判断しきれない重大な論点が残ったら、既存の `--handoff-from-run` / `--handoff-scope-file` でAstraの追加レビューへ渡す手順。
   - 利用状況の確認方法: 既存の記録（run の `contract.review_selection` / `review-selection.json` / runner の `runtime.effective`、トークン使用量・実行時間など、実在するフィールドを確認して書く）からモデル別の起動件数・トークン・実行時間を後から確認する方法。既存のCLI/readerで足りなければ、最小限の集計手段を追加してよい（追加する場合はテスト必須）。
   - Astra固定へ戻す設定（`REVIEW_SELECTION_MODE="fixed"`）。
4. テスト: 既定がselectになることで、既定fixedを前提にしている既存テスト（`tests/e2e/agents_ops/test_review_model_selection.py`、`test_review_choice_handoff.py`、`tests/unit/agents_ops/test_review_settings.py`、`.agents/ops/review/evaluation/execution.py` の置換など）を調べ、意味を保って直す。少なくとも次をテストで確認する（既存テストで足りる部分は再利用）:
   - 既定設定(select)で、簡易例はSol、高リスク/影響不明例はAstraを親が選んだとき、request・runner契約・保存選択に選択理由・要求モデル・実効モデル[local path]。
   - select で指定漏れは明確なエラー（黙って振替えない）。
   - human_instruction 指定、既存レビューの再利用（`review-request --run-id` の保存選択）、coverage・merge gate が回帰しない。
   - fixed に戻した設定で従来どおり Astra 固定になる。
5. pytest は関連ファイル（`tests/unit/agents_ops`、`tests/e2e/agents_ops` のうちreview/parent関連）を実行。`tests/e2e/agents_ops` には変更と無関係な既存失敗がありうるので、失敗は origin/main で同じテストを比べて切り分け、報告する。pre-commit（ruff/mypy）を通す。

## 制約
- 既存の行数・パスからモデルを自動決定する仕組みは作らない（意味判断は親）。
- 静かなフォールバック禁止。
- 過去の比較レポート・保存済みrun/manifestを書き換えない。
- push・PR作成まで行う。mergeはしない（親が正式レビューとmerge gateの後に行う）。

## 最終報告
PR番号、変更点の要約、受入条件3項目それぞれの証拠（テスト名・実行結果）、未解決点・判断してほしい点。
