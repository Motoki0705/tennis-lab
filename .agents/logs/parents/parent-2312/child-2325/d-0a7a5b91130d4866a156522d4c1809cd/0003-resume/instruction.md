PR #2340（HEAD `4d1e5d580c2f10994690d44d77d8b2d1b309b873`）の正式レビュー（`parent-2312-439588a37d3243729512b17af09a48aa`）で、以下の major 指摘が5件出ました。親は**5件すべて採用して直す**と判断しました。再レビューは行いません。

同じブランチ・同じPR（#2340）で5件すべてを修正し、それぞれに対する回帰テストを追加し、全検証・pre-commitを通して commit・push したうえで、`PARENT_RESULT` の `wait.kind="ci"`（`resume_on_success: false`）で新HEADを宣言して終了してください（CI待機はこのターンで行わない）。

## 対応する major 指摘（全5件）

### 1. 最終照合後に進んだ `main` へ未検証の状態をマージする
- **指摘**: `publish.py:131-152` は `main` を GET で照合し、`github.py:195-201` は HEAD SHA だけを条件に `PUT [local path]` する。最後の GET と PUT の間に別PRが引用先の見出しを削除すると、Gitの内容競合は発生せず、引用が壊れた状態をGitHubがマージできる。405/409 が返らないため載せ直しも起動しない。
- **対応**: マージ時に**対象 `base` と `head` の両方の一致をサーバー側で原子的に保証**すること。
  - なお、対象リポジトリの `main` は現在ブランチ保護なし（`protected: false`, ruleset 空）であり、専用PRを作成・差分全ページ照合したうえで、`head`（唯一の親が `base`）を **`git push --force-with-lease=refs[local path]<base> <remote> <head>:refs/heads/main`** で原子的に fast-forward 反映する方式であれば、Git サーバー側で `refs/heads/main == <base>` を CAS 保証しつつ GitHub 上のオープンPRも `MERGED` に遷移する（1コミットでも `main` が進めば内容競合の有無に関係なくサーバー側が stale info で確実に拒否し、載せ直し・再検査へ回せる）。反映後に PR 状態がマージ済みであることも照合し、不整合や拒否は静かにフォールバックせず明示エラーまたは載せ直しとする（より堅牢なサーバー側 base+head 原子的保証がある場合は根拠を示して採用可）。
  - 最終照合直後に別PRが引用先見出しを削除して `main` を進めたケースで、未検証状態がマージされずサーバー側で拒否され、載せ直し後の再検査で引用切れを検出して中止する回帰テストを追加すること。

### 2. AutoDream の古い削除計画が再検証済みのエントリを消す
- **指摘**: 期限切れを理由とする同一の削除操作を、元のエントリと、別セッションが本文と `last_verified` を更新したエントリに適用したところ、どちらも削除として通った。`operations.py:122-146` は削除対象の存在だけを確認し、`publish.py:61-68` は初回も競合時も最新のツリーへ操作を適用する。
- **対応**: `#2326`（AutoDream）や JIT 更新が判断に使った版を `update` / `delete` / `batch` / `publish_operations` に渡せるようにすること（例: 各操作の任意フィールド `expected_sha256`［対象エントリ内容のSHA-256］および必要ならバッチ全体の参照版）。指定された場合、初回適用時および競合載せ直し時の両方で現在の対象エントリのハッシュを照合し、変更を検出したら自動適用せず「対象が変更されたため再判定が必要」という明示エラーで中止すること。再検証済みエントリへの古い削除・更新計画が初回・競合載せ直し時の双方で拒否される回帰テストを追加すること。

### 3. 専用CLIが拒否するファイル名まで `Python` CI から除外する
- **指摘**: 両トリガーの `paths-ignore`（`.agents/memory/<layer>[local path]`）と `schema.py:197-207` の判定を比較すると、`.agents/memory/workflow/Draft.md` は `*.md` に一致して CI 除外となる一方、専用CLIでは拒否される。`test_ci_test_lanes.py:92-149` はこの食い違いを検出しない。
- **対応**: GitHub Actions の `paths-ignore` フィルタ仕様（公式の glob / negation `!` / 文字クラス構文）に基づき、`ci.yml` の除外パターンと専用CLIの許可ファイル名境界（`SUMMARY.md`、各層 `index.md`、各層直下の kebab-case `[a-z0-9]+(-[a-z0-9]+)*\.md`）を正確に揃えること（例: `.agents/memory/<layer>[local path]` に続けて negation `!` で大文字・アンダースコア・空白・複数ドット・先頭/末尾ハイフン・連続ハイフン等の不正ファイル名を除外対象から戻すなど、GitHub Actions の仕様で確実に評価される表現を確認して適用する）。
  - `test_ci_test_lanes.py` を強化し、GitHub Actions のパターン評価順（negation 含む）に沿って、有効な共有memoryファイルだけでなく、不正な直下名（`Draft.md`・`bad_name.md`・`-bad.md`・`bad--name.md`・`a.b.md` 等）、サブディレクトリ配下、非Markdown、`ops-report` がすべて CI 実行対象になることを検証すること。

### 4. 生成文書だけがずれた場合に専用経路で復旧できない
- **指摘**: 有効なエントリを保ったまま生成文書（`SUMMARY.md`・`index.md`）だけを古い内容にすると、`plan_sync` は問題なしで修復差分を返したが、`operations.py:99-101` は空の操作配列を拒否した。`sync-index` の書き込み先は呼び出し元のcheckoutであり、#2326 の Prune & Index が専用経路で生成文書だけ修復できない。
- **対応**: エントリ操作が0件でも、`render_documents` による生成文書（`SUMMARY.md`・4層の `index.md`）に実差分が存在するときだけ専用経路で機械検査・PR作成・即マージにより復旧できる明示的な経路（CLI サブコマンド／フラグ例: `sync-index --publish --reason <TEXT>` または `repair-index`、および Python API の `publish_index_repair(...)` / `publish_operations([], allow_index_repair=True, ...)` 等）を用意すること。生成文書がすでに最新で差分0件（完全な無変更）の場合は従来どおり明示拒否すること。両ケースの回帰テストを追加し、`README.md` と `SKILL.md` にも反映すること。

### 5. SIGTERM で一時 worktree と Git の登録が残る
- **指摘**: `git.py:101-113` は `finally` だけで一時 worktree の登録を解除する。登録後の fetch 中に SIGTERM を受けると、Python の既定動作ではその処理を通らず終了する。一時ディレクトリは `.claude/worktrees/` の外にあるため、AutoDream の実行を停止するたびに残り得る。
- **対応**: 一時 worktree 登録が SIGTERM で残存しない仕組みへ改修すること。
  - **推奨**: 一時 worktree を作らずに専用の一時 `GIT_INDEX_FILE`＋`git fetch --no-write-fetch-head`＋Git plumbing（`read-tree` / `update-index` / `write-tree` / `commit-tree` / `push`）だけで完結させるか（この場合 Git の worktree 登録自体が発生しない！）、または一時 worktree を使う場合でも (a) SIGTERM を捕捉してクリーンアップを確実に実行し、(b) 管理専用の親ディレクトリ／プレフィックスの残存 worktree 登録を次回起動時に自動回収する仕組みを入れること。
  - 登録後（または一時資源作成後）に実プロセスへ SIGTERM を送って中断する回帰テストを追加し、Git の worktree 登録および一時ファイルが確実に回収されること（または次回起動で自動回収されること）を検証すること。

## 検証・完了手順
1. `.agents/ops/shared_memory/README.md` や `.agents/skills/memory/SKILL.md` の記述を上記5点の改善（サーバー側 base+head 原子的固定、`expected_sha256` による楽観ロック、生成文書修復の専用公開経路、`paths-ignore` 境界等）に合わせて更新すること。
2. 以下すべてを実行し全通過を確認すること：
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit`
   - `.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_interactive_checkout.py tests/e2e/development/test_ci_test_lanes.py tests/e2e/agents_ops/test_ops_report_memory.py -n 2 -q`
   - `.venv/bin/python -m ruff check .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/development/test_ci_test_lanes.py`
   - `.venv/bin/python -m ruff format --check .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/development/test_ci_test_lanes.py`
   - `.venv/bin/python -m mypy --follow-imports=skip .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/development/test_ci_test_lanes.py`
   - `.agents/ops/review/` の差分が0件であることを確認。
3. コミット（pre-commit通過）して同じブランチへ push し、PR #2340 に5件それぞれの修正内容と回帰テスト結果をコメントすること。
4. 最終報告で5件それぞれの修正コミット・回帰テスト結果・新HEADを報告し、末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2340,"head":"<新40桁HEAD>","resume_on_success":false}}` を出力して終了すること。
