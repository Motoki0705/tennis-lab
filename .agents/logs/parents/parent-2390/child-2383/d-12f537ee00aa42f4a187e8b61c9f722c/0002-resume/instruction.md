# PR #2395 正式レビュー major 指摘の修正指示

正式レビュー（run `parent-2390-b265c0e197bc48bb9575b95e43d37fa2`）で以下の major 指摘（1件）があり、親として採用します。同じブランチ（`codex/parent-2390-issue-2383-primary`）で修正・回帰テスト・文書更新・PR本文更新・commit・push まで行い、新HEADで CI 待機を宣言して終了してください（`resume_on_success: false`）。

## 採用した major 指摘
- **タイトル**: 軽量CIの結果をテスト用マージSHAの一致まで確認せず受理する
- **詳細**:
  - 現行の `verify_result`（`.agents/ops/lib/knowledge_ci.py`）は artifact の `merge_sha` が完全SHAの形式であるかだけを確認しており、他の項目が一致していれば `merge_sha` が別のコミットを指していても受理してしまう。
  - これでは下流 #2386 が、`classify_pr` に渡したテスト用マージコミット（`merge_sha`）と、軽量CIが検査したコミットが同一であることを公開APIで確認できない。

## 修正要求
1. `.agents/ops/lib/knowledge_ci.py` の `verify_result` にキーワード専用引数 `merge_sha: str` を追加し、`_sha(merge_sha, "expected test merge")` で検証のうえ、artifact JSON 内の `merge_sha` との完全一致を要求する（他の識別フィールド `pr_number`, `head_sha`, `base_sha`, `run_id`, `run_attempt` と同様に不一致時は理由付き `ValueError`）。
2. `tests/unit/agents_ops/test_knowledge_ci.py`（および `verify_result` を呼ぶ他のテストがあればすべて）を更新し、**異なる有効形式の40桁SHAの `merge_sha` を拒否する回帰テスト**を追加する。
3. `.agents/ops/lib/README.md` の `verify_result` シグネチャと照合項目の記述、および PR #2395 本文の「#2386 が使う契約」を更新する。
4. 既存の実イベント証拠（検証用ドラフト #2396 の実artifact `knowledge-records-45024770222-1`）の組（`merge_sha = cfebfbf0c641256c9cd43a4860e67589770a7cf0`）に対しても一致受理・別SHA拒否が成り立つことをローカル検証で確認できるなら追記し、pytest・pre-commit（ruff/mypy）を通して commit・push すること。
5. 最終応答では修正commit SHA、追加・実行した回帰テストの結果、更新後の `verify_result` シグネチャを報告し、新HEADで CI 待機（`resume_on_success: false`）を宣言すること。
