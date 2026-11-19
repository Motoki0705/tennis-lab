PR #2052 の再レビュー（run parent-1893-402cdd10a85349ad9d8d115b4ca813f8、HEAD b3e1bdd8、request_changes）で major 3件が出た。いずれも妥当と判断した。修正してほしい。

### pr-2052-1 .agents/ops/worktree_cleanup/lifecycle.py:648（major）
静的推論（再現テスト未実行）。stop_movedの停止後走査は、新たに見つけたcheck['related']をreportへ追加するだけで、observedや停止journalへ保存しません。この走査で初めて参照元Pと子Cを観測した後、Pが参照を閉じ、Cが外へchdir・reparentすると、次のprepare_nativeで両者を失い、稼働中でも削除成功になります。confirm_referencesも同様で、失敗後の再実行では観測済みPIDが確認不能になっても関連未観測扱いになり得ます。各走査の新規観測を耐久保存し、停止処理へ引き継いでください。追加テストは停止準備時から存在する参照を検証しており、この後段での初観測を覆いません。

### pr-2052-2 .agents/ops/worktree_cleanup/references.py:106（major）
静的推論（再現テスト未実行）。cwdは対象外だが対象ファイルのfdを持つ親Pと、外へchdir済みの子Cの親子辺をsnapshotで取得した場合、Pのfd参照を読んだ後、最後のstatまでにPが終了すると、ここまたは115行のreturn Noneがmatchesごと観測を捨てます。Pはknownへ追加されず、既に取得したP→Cも停止対象へ展開されません。その後Cがreparentすると、Cを残したまま削除成功へ進めます。最初から終了済みのPIDと、参照観測後に終了したPIDを区別し、後者の観測を子孫追跡へ残してください。終了PIDのunitテストは走査開始前の終了だけを検証しています。

### pr-2052-3 .agents/ops/worktree_cleanup/migration.py:116（major）
静的推論（再現テスト未実行）。旧版がGit登録解除後・実体削除前に中断したphase=unregisteredの記録は、現在のGit一覧に登録がないためactual=Noneのままsupersededになります。残る.expired-<id>を確認せずCleaner.observeの再開対象から外すため、この対象だけならresults=[]・outcome=success・終了0となり、隔離実体が以後も放置されます。旧削除権限を閉じる場合でも、残存実体や復旧義務は未完了・失敗として保持してください。test_disappeared_legacy_intent_is_closed_without_a_permanent_failureは登録と実体を両方消しており、登録だけ消えた中断状態を検証していません。

## 親の指示
- 3件とも、新方針（隔離パスの参照がないことをゲートにし、観測できた参照プロセスとその子孫は停止する。ホスト全体の孤児不在証明はしない）の範囲で、狭く直す。ホスト全体の不明判定や不在証明を再導入しない。
- pr-2052-1: 停止準備・停止後走査・confirm_references の各走査で新たに観測した参照プロセスと子孫を、停止journal（observed）へ耐久保存し、以降の停止処理と再実行へ引き継ぐ。観測済みPIDが後で確認不能になった場合は「関連あり」として扱う。後段で初めて観測したケースの回帰テストを置く。
- pr-2052-2: 参照を読めた後に終了したPIDは、その観測（参照の一致と取得済みの親子辺）を捨てず、子孫追跡に残す。走査開始前から終了していたPIDとは区別する。参照読取後・最後のstat前の終了を故障注入で再現するテストを置く。
- pr-2052-3: 旧 phase=unregistered（Git登録解除後・実体削除前の中断）の記録は、残っている隔離実体（.expired-<id> など）を確認し、実体が残れば再開対象として削除を完了させる（新形式の照合で個体を確かめたうえで）か、完了できなければ失敗として保持する。登録だけ消えて実体が残る状態のテストを置く。
- 修正を外すと新しいテストが失敗することを確かめる。関連テスト（worktree_cleanup・references・reference_gate・generation_races・checkout_job）と pre-commit を通し、通常の push（force pushしない）、PR本文の更新、対象HEADのCI確認まで進める。pytest の並列は `-n 4` 以下。
- 実ホストでは読み取りだけの診断と本番stateのコピーでの dry-run だけを行う。本番の削除・unit操作・state更新・signal はしない。作業時間の目安は最大4時間。
