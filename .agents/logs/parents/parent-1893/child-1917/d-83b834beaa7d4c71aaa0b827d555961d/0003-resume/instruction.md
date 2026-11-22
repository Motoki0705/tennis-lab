PR #1952（#1916 git-sync と共通launcher）が main にmergeされた（merge commit `e1016bb8e`）。このターンでは、PR #1953 を main へ載せ替え、前回の統合レビューの横断指摘を解消してほしい。

## 1. main への rebase（あなたが行う）

- 現在のHEADは `f1a03dcc9`（origin と一致、子のcommit 5件）。`git fetch origin && git rebase origin/main` を行う。親が試したところ、最初のcommit「Expire registered worktrees after 336 hours…」で `.agents/ops/ops_control/README.md`・`.agents/ops/ops_control/targets.py`・`tests/e2e/agents_ops/test_ops_control.py` が競合した（git-sync の対象登録と並ぶ追加）。両方の対象を残す形で解決する。
- rebase後は `git push --force-with-lease=codex[local path]` で push する（この1回だけ force push を許可する）。

## 2. 横断指摘の解消（前回レビュー run parent-1893-de0fed66393b44de94f34aa665c6ba17 の横断 major）

> `.agents/ops/worktree_cleanup/systemd/tennis-lab-worktree-cleanup.service:9`: Maintenance.finite_idle() が TARGETS 内の有限serviceについて checkout_job.py 経由の ExecStart を要求する。このserviceは直接モジュールを起動するため、両PRを導入すると、停止中でもgit-syncが unguarded_job_units で毎回延期される。手動spin入口にも共通ロックがない。serviceと手動入口を共通launcherへ接続し、両PRを含む構成で排他を検証すること。

- `.agents/ops/lib/README.md` の後続job接続手順に従い、worktree-cleanup の service（ExecStart）と、削除を実行する手動入口（`spin worktrees cleanup`、`spin ops run worktree-cleanup`、モジュールの直接実行）を共通launcher（`checkout_job.py`）経由にするか、launcher外からの直接実行を `require_checkout_job` で拒否する。読み取りだけの `spin worktrees scan` と `cleanup --dry-run` を launcher 必須にするかは、排他の必要性から判断し、PR本文に理由を書く。
- worktree-cleanup が git-sync と排他されること（git-sync 実行中は削除しない／削除中は git-sync が延期する）と、worktree-cleanup の unit がある構成で git-sync が `unguarded_job_units` で延期されないことを、両方を含む構成のテストで検証する。
- worktree-cleanup が停止連携で dispatcher や他の運用ジョブを止める処理と、git-sync の保守停止・再開（resume 記録・手動OFFの保持）が衝突しないか確認し、必要なら調整する。

## 3. 仕上げ

- 関連テスト（worktree_cleanup・git_sync・checkout_job・ops_control・cleanup・parent_runtime など）と pre-commit を通す。pytest の並列は `-n 4` 以下。
- PR #1953 の本文を、rebase・launcher接続・テスト増分で更新し、対象HEADのCI結果を確認する。
- ホストの systemd 操作、実repoでの削除、メインcheckout・他worktreeの変更はしない。
- 作業時間の目安は最大4時間。止めてよいのは、rebase・接続・テストをpushしCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
