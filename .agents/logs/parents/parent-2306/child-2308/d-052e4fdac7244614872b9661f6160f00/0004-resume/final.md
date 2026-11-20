空の最終報告を扱うテストを修正し、既存のドラフトPRへpushしました。原因と検証結果をPR本文に追記しました。[PR #2313](https://github.com/Motoki0705/tennis-lab/pull/2313)

**Branch** `codex/parent-2306-issue-2308-primary`  
**HEAD** `4e05883b6fc0f3cb3f8679190b437b6d764f1e67`

原因は、擬似ランナーが空の最終報告に有効な `PARENT_RESULT` を補い、判断記録が `success` になっていたことです。空の出力をそのまま保存するよう修正しました。空の報告と宣言のない本文の両方で、`run-worker` と `wait-worker` が終了値2を返し、親の判断記録が `failure` になることを確認しました。空の報告には `missing_final` が残ります。

- `test_decision_capture.py` 全14件と前回の回帰テストを含む関連 **589件が通過**しました。pre-commitのRuff・mypyも通過しました。
- CPUのe2e全体は **4,277件通過、5件失敗、1件スキップ**でした。HTML関連4件は `origin/main` でも同じエラーを再現しました。レビュー通知の1件は通常の再実行で両方通過し、通知の配送直前で処理を止めると両方で `incomplete` を再現しました。

新HEADのGitHub CI結果は未確認です。