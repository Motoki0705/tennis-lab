担当は #2189（親 #1893）。旧形式で隔離済みだった worktree が、Git登録の消失を理由に識別子移行で閉じられ、隔離実体が放置される問題を直す。issue本文の受入条件をすべて満たす。

## 実ホストの証拠（読み取りに使ってよい）

- 2026-11-16 15:09 JST の単発実行（`spin ops run worktree-cleanup`）。report は `[local path]`、state は `[local path]`、直前の dry-run の写しは `[local path]`。journal は `journalctl --user -u tennis-lab-worktree-cleanup.service --since "2026-11-16 15:05"`。
- 閉じられた3件の記録（state の records）: id 6a2fb73e…（ball-transfer-feasibility）・062ad49c…（ci-contract-tests-before）・23ab7153…（main-check-1329）。いずれも phase=superseded、filesystem_migration.result=retired、reason=legacy Git registration disappeared、previous_phase=quarantined、quarantine に `.claude/worktrees/.expired-<id>` が保存されている。実体は残っている（各約500MB、.git ファイルを含む）。
- これらは旧コード（PR #2034 時代）で隔離後に停止確認の失敗で中断し続けていた記録で、その後のどこかで Git の worktree 登録が prune された。

## 進め方

- まず、登録が消えた経緯（prune の主体と時刻、隔離実体の .git ファイルの中身、`.git/worktrees/<id>` の有無）を読み取りで調べ、PR本文に書く。
- 移行で閉じてはいけない未完了段階（quarantined・unregistered・stopping など）を列挙し、隔離実体が残る限り義務を保持する。照合できる場合は参照ゲートを通して削除を完了し、照合できない場合は明示的な失敗として残す。既に superseded で閉じてしまった実ホストの3件を、再実行で回復できるようにする（移行の再解釈を許す条件を明確にし、冪等にする）。
- 修正を外すと新しいテストが失敗することを確かめる。関連テスト（worktree_cleanup・identity_migration・reference_gate・references・generation_races・checkout_job）と pre-commit を通し、PR（本文は日本語、「Closes #2189」「親: #1893」）を作って対象HEADのCIを確認する。pytest の並列は `-n 4` 以下。GitHub API は共有上限が逼迫しているので CI の確認は必要最小限にする。
- PR本文に、本番stateのコピーでの dry-run で3件が削除対象として選ばれ、参照0であることを示す。

## 実ホストで許すこと・禁止すること

- 許す: 読み取りだけの診断、`[local path]`・journal・state の読み取り、本番stateのコピーでの dry-run（scratch は `[local path]`）。
- 禁止: 本番の削除、隔離実体の変更、unit操作、worktree-cleanup の本番 state の書き換え、シグナル送信。メインcheckoutと他のworktreeの変更。
- 作業時間の目安は最大4時間。止めてよいのは、受入条件を満たしPRのCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
