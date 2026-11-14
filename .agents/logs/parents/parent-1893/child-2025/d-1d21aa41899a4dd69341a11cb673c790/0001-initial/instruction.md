担当は #2025（親 #1893 の実ホスト検証で見つかった worktree-cleanup の停止要因の修正）。issue本文の受入条件をすべて満たす。

## 実ホストの証拠（読み取りに使ってよい）

- 2026-11-14 17:26〜18:17 JST の単発実行（`spin ops run worktree-cleanup`）の結果: `[local path]`・`[local path]`・`[local path]`。94件すべて `deleted: false`。エラーの内訳は「duplicate keys in worktree state」94件、「native descendants survived or new orphan provenance is unconfirmed」56件、「delegated worker has no runner termination evidence」12件。
- journal: `journalctl --user -u tennis-lab-worktree-cleanup.service --since "2026-11-14 17:26" --until "2026-11-14 18:18"`。git-sync の延期は `journalctl --user -u tennis-lab-git-sync.service --since "2026-11-14 17:59" --until "2026-11-14 18:01"`。
- 「duplicate keys」の原因として親が特定したもの: `lifecycle.py` が `[local path]` 配下を再帰走査して全 `result.json`（5,793件）を厳密に読む。壊れたものは13件で、すべてpytestの一時ディレクトリに残った故意の破損fixture（`parent-1900-tmp/…`、`campaign-1329/pt-1331*[local path]`、`worker-session-replacement/tmp/pytest-of-kamimura/…`）。これらのファイルは他の親の作業物なので、削除・移動しない。
- 実行前の確認では、期限到達94件のどれにも cwd を持つプロセスはなかった。parent CLI の registry・dispatcher・training queue には今回の実行による変更はなかった（94件の記録は worktree-cleanup の状態で intent/stopping のまま）。

## 実ホストで許すこと・禁止すること

- 許す: 読み取りだけの `spin worktrees scan`・`spin worktrees cleanup --dry-run`、`[local path]` や journal の読み取り、原因特定のための読み取りだけの診断スクリプト（scratchは worktree 内の ignore される場所か `[local path]` に置く）。
- 禁止: 実repoでの削除の実行（`spin ops run worktree-cleanup`、`spin worktrees cleanup` の非dry-run）、systemd の unit 操作（on/off/start/stop/daemon-reload/install）、worktree-cleanup の状態ファイル・parent CLI の registry・dispatcher・training queue の書き換え、プロセスへのシグナル送信。merge後の実ホストでの単発実行は親が行う。

## 進め方

- 「native descendants survived…」は、実ホストでどのPIDが `uncertain` や `live` に残ったのかを、読み取りだけの診断で特定してから直す（WSL2 の [local path] cwd など）。原因と根拠をPR本文に書く。
- 「記録なし委任」は、worktree内に稼働プロセスがないことを確認したうえで根拠付きで終了扱いにするか、人間判断を要求する明示状態にするか、合意（使用中でも免除しない・停止失敗を成功扱いにしない・削除後に稼働中と誤表示しない）に照らして選び、理由を書く。判断に迷う場合は、独立した部分を仕上げてから最終報告で親に選択肢と影響を返す。
- 修正を外すと新しいテストが失敗することを確かめる。関連テスト（worktree_cleanup・git_sync・checkout_job・ops_control・cleanup・parent_runtime）と pre-commit を通し、PRを作って対象HEADのCIを確認する。pytest の並列は `-n 4` 以下。
- PR本文に、実ホストでの dry-run の結果（対象件数、失敗理由が消えたか）を載せる。
- 作業時間の目安は最大5時間。止めてよいのは、受入条件を満たしPRのCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
