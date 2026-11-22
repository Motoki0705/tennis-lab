# #2031 の実装（PR作成まで）

担当Issue #2031 の本文が要件と受入条件の正本です。全数調査: https://github.com/Motoki0705/tennis-lab/issues/2013#issuecomment-6875909177 の表の dispatcher（#2〜#25）・triage（#26）・parent/development.py（#29〜#31）・worker_logs/decision_logs（P表）の行と、「storeを渡したときの副作用と競合契約」の dispatcher に関する指摘（Store の同一性、複数親の focus、schedule が unlocked_read の比較対象でないこと、history_read_scope と GET cache、Supervisor/service が別の DispatchStore を作ること）を前提にしてください。

#2029（PR #2038）で、transaction用client（`GitHub(repo, store=store)`）とロック外client（`GitHub(repo)` / `detached()`）の分離、cache の transaction epoch、`fresh_reads`、`_require_scope` が main に入っています。この契約の上に乗せ、契約自体を変える必要があると判断したら実装を止めて報告してください。

## 進め方

- runner の status/stop・終了待ち、整理runner は「intent を保存→ロック外で実行→新しいtransactionで同じ job/run を照合」に変える。競合や不明な終了を成功扱いしない。
- ロック外I/Oの間の schedule・capacity・担当の変更を再照合し、競合後に古い schedule を保存しない。
- 並行して #2040（gh サブプロセス通信の置き換え）が `issues/cli.py` と `parent/github.py` の通信層を、#2032 が board・Project を変更します。それらのファイルは必要最小限にし、衝突したら後からmergeする側がrebaseします。

## 完了の条件

- 受入条件のテストを含むPRを `Closes #2031` 付きで作成し、関連テストと ruff/mypy（pre-commit）が通り、PRの対象HEADでCI（Python）が成功していること。
- 時間の上限は8時間。止めてよいのは、(1) 既存契約の大きな変更が必要で親の判断が要る、(2) 8時間を超えた、のどちらかだけ。止める場合は、済んだ範囲・未了の受入条件・次の一手を報告する。
