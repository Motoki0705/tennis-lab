# PR #2407（#2386）の最新 `main` への載せ替え指示

先行PR #2395（#2383）が `main` へマージされ（merge commit `785db4ac7d53cbf0825e389f122effc0a26da79a`、#2383 は `closed/completed`）、その後にも他PRが `main` へマージされています。`workers.md` のスタックPR手順に従い、PR #2407 を最新 `origin/main` 上へ載せ替えて `main` 向けPRとして完了させてください。

## 作業手順
1. このworktree（`.claude/worktrees/parent-2390-issue-2386-primary`）で `git fetch origin main` する。
2. 先行PRとの境界コミット `6d60b29184cd626d47957c4448f170d1e2b81127` より後の #2386 の変更（`ad44503d6feb08389b2a9a927d56e860553b95a1`）だけを最新 `origin/main` 上へ載せ替える（`git rebase --onto origin/main 6d60b29184cd626d47957c4448f170d1e2b81127`）。
   - `origin/main` と重複するファイル（`.agents/ops/parent/worker_monitor.py`, `.agents/ops/review/README.md`, `.agents/ops/review/merge_gate_impl.py`, `.agents/ops/review/progression.py`, `.agents/ops/review/progression_backend.py` 等）に競合や意味的な相互作用がある場合は、`origin/main` 側の最新変更を壊さずに #2386 の変更を正しく統合すること。
3. 載せ替え後のHEADで、対象テスト（新規32件を含む関連テストおよび競合・重複ファイルの関連テスト）と `pre-commit`（ruff / mypy / diff check）および文書リンクテスト（`tests/e2e/agents_ops/test_agents_docs.py`）を実行して通過を確認する。
4. `git push --force-with-lease origin codex/parent-2390-issue-2386-primary` で push し、`gh pr edit 2407 --base main` で PR #2407 の比較元を `main` に切り替える。
5. PR #2407 の差分が #2386 の変更のみであること・競合がないことを確認し、PR本文の比較元・最新検証結果を更新する（冒頭の「先行PR #2395 のブランチを比較元にしている」旨を `main` ベースへ更新）。
6. 新HEADでの CI 起動を確認し、新HEADの完全SHAで CI 待機（`resume_on_success: false`）を宣言して終了する。
