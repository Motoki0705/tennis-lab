PR #1953 の統合レビュー（run parent-1893-de0fed66393b44de94f34aa665c6ba17）の指摘を修正してほしい。runは横断指摘の検証エラーで記録上は incomplete になったが、reviewerの内容は request_changes で、以下の major 4件はすべて有効と親が判断した。

なお、親がこのbranchを #1918 merge後の main へ rebase して force push 済み（現在のHEAD `ff1804f1e`、子のcommit 4件）。作業前に `git fetch origin && git status` で、ローカルが `origin/codex/worktree-expiry-cleanup` と一致することを確認する。PR #1953 の base は main に変更済み。

## PR内の指摘（すべて修正する）

### .agents/ops/worktree_cleanup/cleanup.py:205（major）
静的推論（Git 2.43.0の実装も照合、テスト再実行なし）です。199行の存在確認後、state.save()からgit呼出しまでに別プロセスがgit worktree add --forceで同じパスを再作成すると、Gitは欠落した旧登録を置き換えられます。その後のremove --force --forceはパスから新登録を選び、新worktreeの実体まで削除します。事後のcurrent(row)は旧個体が消えたことしか確認しないため成功扱いです。登録解除を旧管理個体に固定し、この直前の再作成を保持する競合テストを追加してください。既存テストは登録解除後の再作成だけです。

### .agents/ops/worktree_cleanup/cleanup.py:53（major）
静的推論（テスト再実行なし）です。birth timeを取得できない環境で、初観測後に外部のgit worktree removeで削除すると、次のscanで登録消失を確認しても旧記録はobservedのまま残ります。同じ管理パスでdevice/inodeが再利用されると、この検索が旧記録を選び、新しいdirectory_identityへ更新しながら古いcreated_at_nsを維持するため、再作成直後でも旧期限で削除されます。消失を観測した記録は再利用対象から外してください。既存のinode再利用テストは本ツールでphase=deletedになった場合だけです。

### .agents/ops/parent/session.py:211（major）
静的推論（テスト再実行なし）です。owner_runtimeはregisterだけで更新され、既存のhandoff --acceptはsessionとgenerationを変更しても旧値を保持します。旧親のworktreeが期限到達すると、Stopper.registry()はその古いcwdだけで現在の親を対象と判断し、移管先の担当をreleasedにして、別worktreeの子まで停止します。旧PIDが既に終了していても担当解除は起きます。担当移管時に来歴を更新／失効し、停止側でも現在の担当との一致を確認してください。登録フックの新規2テストには移管ケースがありません。

### .agents/ops/worktree_cleanup/processes.py:144（major）
静的推論（テスト再実行なし）です。stop_tree()は一度取得したプロセス一覧だけを停止・待機します。共通ランナー経由でないnative親やその子が一覧取得後からTERM処理中に新しい子を生成すると、その子はpinnedに入らず、親終了後も残ります。後段のagents()はcodex／claudeだけを探すため、残ったPythonやshellを検出せず、停止成功としてworktreeを削除できます。停止中の新規生成・親変更を含めて子孫の消滅を保証し、一覧取得後に子が生まれるケースを検証してください。既存テストは固定された2プロセスの停止だけです。

## 横断指摘（今回はまだ実装しない）

.agents/ops/worktree_cleanup/systemd/tennis-lab-worktree-cleanup.service:9（major）: 静的推論（テスト再実行なし）です。兄弟PR #1952のHEAD db30761e63d876e0640420df939234e3b93234feでは、Maintenance.finite_idle()がTARGETS内の有限serviceについてcheckout_job.py経由のExecStartを要求します。このserviceは直接モジュールを起動するため、両PRを導入すると、停止中でもgit-syncがunguarded_job_unitsで毎回延期されます。手動spin入口にも共通ロックがありません。serviceと手動入口を共通launcherへ接続し、両PRを含む構成で排他を検証してください。

共通launcher（`checkout_job.py` など）は兄弟PR #1952（#1916 git-sync、branch `codex/main-git-sync`）にあり、まだmergeされていない。親は #1952 を先に統合レビュー・mergeし、その後このbranchを main へ載せ替えてから、worktree-cleanup の service と手動 spin 入口を共通launcherへ接続する作業を、次のターンで依頼する。今回のターンでは、#1952 のlauncherを前提にしたコードを書かない。接続を見越して入口を一か所にまとめておくのはよい。

## やること
1. 上の4件を修正し、それぞれ指摘が述べる競合・ケースを再現する回帰テストを追加する（直前の再作成、birth time 不明時の外部削除とinode再利用、handoff 後の古い担当来歴、停止中の子プロセス生成）。修正を外すと新しいテストが失敗することを一時的に確認する。
2. `.agents/ops/parent/session.py` の担当移管（`handoff --accept`）で、来歴を更新または失効させる。停止側でも現在の担当との一致を確認する。parent CLI の既存テスト（`tests/e2e/agents_ops/test_parent_runtime.py` など関係するもの）も通す。
3. 関連テストと pre-commit を通し、commit して通常の push をする（force push しない）。PR #1953 の本文に修正内容とテストの増分を追記し、対象HEADのCI結果を確認する。pytest の並列は `-n 4` 以下。
4. ホストの systemd 操作、実repoでの削除、メインcheckout・他worktreeの変更はしない。
作業時間の目安は最大4時間。止めてよいのは、4件を修正してpushしCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
