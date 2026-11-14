担当は #2051（親 #1893 の実ホスト検証で、worktree-cleanup の停止確認が無関係なプロセスを「不明」に含め、期限到達6件を削除できない問題）。issue本文の受入条件をすべて満たす。

## 実ホストの証拠（読み取りに使ってよい）

- PR #2034 merge後の単発実行（2026-11-14 23:47〜11-15 00:05 JST）で94件中88件を削除し、6件（ball-transfer-feasibility、ci-contract-tests-before、ci-speed-parent、main-check-1329、scene-ball-loess、scheduled-parent-dispatch）が失敗した。直後の再実行（00:1x JST）でも6件とも同じ理由で失敗した。最新の結果は `[local path]`、2回目実行時点の写しは `[local path]`、journal は `journalctl --user -u tennis-lab-worktree-cleanup.service --since "2026-11-14 23:40"`。
- 不明（uncertain）に含まれた主なプロセス（親が確認した時点の情報）:
  - 1463143: 他の Claude Code セッションの bash（cwd=メインcheckout、親は Claude 本体）。1482639: メインcheckoutの `.agents/ops/lib/agent_runtime.py`（他の親の共通ランナー）。いずれも対象worktreeと無関係に稼働中。
  - 1489942: root の `[local path]`（親PID 1、cwd 読取不可）。
  - 1425096・1425100: 別worktree `parent-1900-cli` の runner（pid 1267147）の zombie 子。
  - そのほか、snapshot の後に消えた短命PIDが多数。
- 6件の worktree の cwd を持つプロセスは実行前に0件だった。

## 実ホストで許すこと・禁止すること

- 許す: 読み取りだけの `spin worktrees scan`・`spin worktrees cleanup --dry-run`、`[local path]`・journal の読み取り、読み取りだけの診断スクリプト（scratchは `[local path]`）。
- 禁止: 実repoでの削除の実行、systemd の unit 操作、worktree-cleanup の状態ファイル・parent CLI の registry・dispatcher・training queue の書き換え、プロセスへのシグナル送信。merge後の実ホストでの単発実行は親が行う。

## 進め方

- まず、上の各プロセスが不明に入った経路を、実ホストの読み取りだけの診断とコードで特定する（観測順・親PIDの連鎖・reparent・uid・Z state など）。原因と根拠を PR 本文に、分類ごとに書く。
- 「対象の子孫かもしれない」の判定を、対象worktreeの cwd を観測したプロセス（または消滅前に対象に属すると観測した親）からの連鎖に限定し、対象との関係が観測されていない独立の系統（メインcheckoutや別worktreeのジョブ、root のサービス、PID 1 配下で対象と無関係に起動したもの）を含めないようにする。対象の子孫の追跡（pr-2034-1 の修正）は弱めない。どちらにも決められないケースが残るなら、その条件を明示して不明のまま残し、PR本文で親へ説明する。
- 修正を外すと新しいテストが失敗することを確かめる。関連テスト（worktree_cleanup・worktree_generation_races・ops_control・cleanup・git_sync・checkout_job）と pre-commit を通し、PRを作って対象HEADのCIを確認する。pytest の並列は `-n 4` 以下。
- PR本文に、実ホストでの dry-run（stop_preflight を含む）の結果を載せる。可能なら、6件それぞれについて、修正後の判定で不明が空になることを読み取りだけの診断で示す。
- 作業時間の目安は最大4時間。止めてよいのは、受入条件を満たしPRのCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
