担当は #1916（メインcheckoutの main を30分周期で origin/main へ安全に fast-forward する git-sync）。issue本文の範囲・対象外・受入条件をすべて満たす。

## 重点

- 更新先のコミットは fetch 後に固定した SHA とし、`merge --ff-only` 相当で更新する。分岐・追跡ファイルの変更・進行中のGit操作（rebase/merge/cherry-pick/bisect等）・未追跡ファイル（data/ 等）と更新先の衝突は、それぞれ区別できる理由を状態とログに記録し、非0または明示的なskip状態にする。自動 stash/reset/rebase はしない。静かに成功扱いにしない。
- メインcheckoutが main 以外のbranchをcheckoutしている場合の扱いも明示する（更新しない理由として記録する）。
- 作業用worktree・共有 .venv・submodule checkout は変更しない。`uv.lock` 等の依存定義や submodule 固定版が更新で変わった場合は、状態・ログに表示する。
- 運用ジョブとの排他: 更新と、メインのコードを使う運用ジョブ（cleanup・report・triage・dispatch判定など）の起動を共通ロックで排他する。有限ジョブが実行中なら延期する。常駐 supervisor（dispatcher）は管理下の子の状況を確認してから停止・更新・再開し、手動OFFになっているものを再開しない。既存のロック・dispatcher CLI があれば再利用する。
- `spin ops on/off/run/status/logs git-sync` で30分周期の timer を扱う。
- テスト: bare origin + clone の一時repoで各条件、fake のプロセス・ロックで並行・延期・手動OFF保持を検証する。READMEと unit 導入手順を実装と一致させる。
## 積み上げの前提

- このworktreeは PR #1918（#1915: spinの運用CLIと systemd 制御の土台、branch `codex/spin-ops-control`、head `34db761045e42cd6cd3a6496e749392f6fa1e89b`）の上に積んでいる。#1918 は統合レビュー中でまだmergeされていない。
- 先に `.agents/ops/ops_control/README.md` と `targets.py` を読み、#1918 の拡張点（`TARGETS` に `timer_target(name, stem)` を追加し unit を置く）に載せる。土台に足りない点があっても、このbranchで土台を書き換えずに、必要な変更と理由を最終報告で返す（土台の修正は #1918 側で行う）。小さな追加フックが必須なら、その変更を独立したcommitに分けて説明する。
- PRは base を `codex/spin-ops-control` にして作り、子の差分だけを表示する。本文に「Closes #担当issue」と「親: #1893」、積み上げ先が #1918 であることを書く。#1918 のmerge後は親がmainへ載せ替える。
- 兄弟の子（#1916 git-sync と #1917 worktree-cleanup）が別worktreeで並行して同じ土台に積んでいる。`targets.py`・`.spin/README.md`・`install_units.sh` などの共通ファイルへの追加は、相手との競合が最小になるよう、既存の並びの末尾に1件追加する形にする。

## 制約

- ホストの systemd unit の導入・有効化・停止、`systemctl --user` の状態変更、`loginctl` の操作をしない。実ホストでの検証は全子のmerge後に親が行う。
- メインcheckout（`[local path]`）と他のworktreeを変更・削除しない。テストは一時repo・一時worktree・fake の systemctl / journalctl / 時計 / プロセスで行う。
- 兄弟workerとpytestが並行するので、pytestの並列数は `-n 4` 以下にする。
- 作業時間の目安は最大6時間。実装・テスト・pre-commit・PR作成・対象HEADのCI結果確認まで進める。止めてよいのは、(1) 受入条件をすべて満たしPRのCIが成功した、(2) 人間か親の判断なしには進めない矛盾を見つけた、(3) 土台 #1918 の変更なしには実装できないと判明した、のいずれかの場合だけ。(2)(3) では独立に進められる部分を仕上げてから報告する。
