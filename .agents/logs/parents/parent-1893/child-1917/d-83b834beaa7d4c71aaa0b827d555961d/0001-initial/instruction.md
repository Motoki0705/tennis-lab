担当は #1917（メインcheckout配下 `.claude/worktrees/` の登録worktreeを作成から14日＝336時間で日次に強制削除する worktree-cleanup）。issue本文の範囲・対象外・受入条件をすべて満たす。

## 重点

- 作成時刻の根拠は `.git/worktrees/<id>` 管理ディレクトリの birth time（statx）。取得できない場合だけ、削除ツールの初観測時刻を根拠 `first_observed` 付きで記録し、そこから14日で削除する（人間の判断、2026-11-12）。mtime/ctime・最古commit・reflogを作成時刻に使わない。UTCで `${XDG_STATE_HOME:-[local path]}[local path]` 配下に永続記録し、同じパスの再作成は管理ディレクトリの同一性（inode等）で検出して新しい作成時刻を記録する。
- 境界: 336時間未満は残し、336時間ちょうど以上は削除。dirty・未追跡[local path] lock を免除理由にしない。
- 使用中の扱い: parent CLI の担当・委任、共通ランナーのjob、training queue など管理可能なものは停止と状態更新を連携させ、削除後に稼働中と誤表示しない。停止・削除の失敗は明示し成功扱いにしない。連携に既存CLIで足りない部分があれば、最小の追加か最終報告での判断依頼にする。
- 安全: Git の登録解除と実体削除を整合させる（`git worktree remove --force --force` 等と prune の順序、中断後の再実行）。symlink の参照先（共有 .venv 等）・repo外・メインcheckout を絶対に削除しない。削除後のmerge済みbranch整理は既存 cleanup を再利用する。
- `spin worktrees scan`・`spin worktrees cleanup [--dry-run]` と、毎日04:30 Asia/Tokyo の timer を持つ spin 対象 `worktree-cleanup`（on/off/run/status/logs）を追加する。scan/dry-run は作成時刻・根拠・期限・対象・破棄される変更の有無を、実行結果は path・HEAD・判定時刻・削除成否を残す。
- cleanup README の「worktree は承認制で clean な場合のみ削除」という方針を今回の合意に合わせて更新する。
- テスト: fake 時計と一時repo・一時worktreeで、境界（直前・ちょうど・超過）、移行、同パス再作成、birth time 不明時の初観測、並行起動、削除の中断・失敗からの再実行、symlink先・.venv・repo外・メインcheckout を消さないことを検証する。
- 実repoでは読み取りだけの `spin worktrees scan` と `cleanup --dry-run` を実行してよく、その出力の要約（件数・最古・根拠の内訳）をPRに載せる。実repoで削除を実行しない。
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
