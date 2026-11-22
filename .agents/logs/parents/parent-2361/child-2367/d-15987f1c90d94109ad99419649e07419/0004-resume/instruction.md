# PR #2372 をmainへ載せ直す

#2371 がmainにマージされた（merge commit 8541470e737928c97a26345ac3159b52e712536b）。そのため、このPRはmainと競合している（GitHubの判定はCONFLICTING）。PRが競合しているので、HEAD 4739c50aa のPython CIは始まっていない。

1. `origin/main` を取り込んでbranchを載せ直す。rebaseでもmergeでもよいが、履歴を壊さない方法を選ぶ。
2. 親の判断（PRコメント https://github.com/Motoki0705/tennis-lab/pull/2372#issuecomment-7076828050 ）に従う。`tests/e2e/agents_ops/test_manual_worker_runs.py::test_from_commit_prepares_and_starts_separate_manual_workers` のE:出力fixtureの修正は、mainに入った #2371 の版を使う。このPRの同じ修正（コミット 7fce1cbce「手動workerの検証に一時ミラーを用意する」、`MANUAL_WORKER_STORAGE_CONFIG` の注入）は取り下げる。ただし、同じファイルにこのPRの容量保護の変更（容量拒否後の再試行など）があるなら、それは残す。
3. 載せ直した後、容量保護のfixtureがmainの版のテストと噛み合うか確かめる（#2371 の版は `.agents/ops` を一時repoへ複製して起動する。容量測定の差し替えがその複製先でも効くか）。そのテスト、対象の容量保護テスト、回帰テストを実行する。
4. 検証してpushし、PR番号と完全なHEADを宣言して終了する。CIの結果は後続のターンで受け取る。
