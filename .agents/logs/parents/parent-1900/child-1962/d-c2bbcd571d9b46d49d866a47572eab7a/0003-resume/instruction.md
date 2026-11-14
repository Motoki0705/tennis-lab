統合レビュー（run parent-1900-8ac8121cbf4b4348b9819cb7785723f2、HEAD 0c240f59、判定 request_changes）の指摘を直してください。PR #1973 のその他の確認項目（51／4／47、4状態と重複排除、公開snapshotのみの生成、PRとmainの分離、v1原文の維持、新しい観測値への安全検査）は問題なしと判定されています。

## 指摘 cross-1（major、.agents/ops/decision_logs/trace_schema.py 67行付近）
兄弟の PR #1984（HEAD 9926146e、子 #1961）の `remaining_publications()` は、捕捉が不完全なとき `error=input_capture_failed`、`unpublished_count=None` を返す。しかし本schemaは件数を非負整数に限り、`remaining_status=unknown` も残件配列が空の場合しか許さない。このため、判明しているworker残件と、件数未確認のdecision残件を同時に保持できない。0への置換や残件の削除は、不明値の誤表示になる。

## 対応
1. termination観測のschemaを、#1984 の実際の契約に合わせる。#1984 のbranch（origin/codex/log-publication-completion）の `.agents/ops/worker_logs/reconciliation.py`、`worker_logs/lifecycle.py` の `record_exit()`、worker_logs README の「続行・一括回収・終了checkpoint」節を正本として読む。
   - `RemainingPublication`: kind、parent、issue、public_key、sequence、state、stage、error、pr_number、pr_head、unpublished_count、batch_key。欠測はNone。
   - 終了記録: repo、parent、generation、event、observed_at、main_revision、remaining、publication_actor、resume_command に加え、remaining_complete、error、decision_resume_command。
2. 件数未確認（None）や `remaining_complete=false` を、既知の残件と同時に保持し、READMEで「未確認」と表示できるようにする。0や空一覧へ置き換えない。
3. fixtureでの仮定（`d-` 付きの公開キー、`unpublished_count`・`batch_key`、担当の文字列表現）を、上の契約と一致させる。`trace_capture.capture_termination` のadapterは、#1984 の型をそのまま受け取れる形にする。
4. 回帰テスト: #1984 の不完全捕捉の出力（worker残件1件＋件数未確認のdecision残件、`remaining_complete=false`）を入力にしたケース。
5. release / handoff / supervisor への配線は、まだ行わない（#1984 のmerge後に依頼する）。

## 検証と終了条件
- 関連テストと tests/e2e/agents_ops の decision_logs 関連を実行し、pre-commit を通して push する。pytest の並列は `-n 4` 以下。ホストのディスクとメモリに余裕がないため、一時ファイルは [local path] ではなく worktree 外の state 配下に置く。
- 作業時間の上限は2時間。修正をpushし、最新HEADのCIを確認したら終了する。
- 最終報告には、最終HEAD、変更点、追加したテスト名、CIの結果、#1984 の契約との対応表を書く。
