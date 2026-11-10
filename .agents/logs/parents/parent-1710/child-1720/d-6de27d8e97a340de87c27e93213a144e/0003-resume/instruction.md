CIの失敗（CPU tests (7)、`tests/integration/tasks/annotation/test_scene_workflow.py::test_blank_cli_pins_policy_runs_isolated_condition_and_reports_without_publication` の `BudgetStopped: owned-process PSS budget reached`）は、main（3f11ab019 のCI run 42599522810 は成功）では起きていない。このPRに起因すると判断する。

親の仮説: annotationのresource schedulerは、自分と同じプロセスグループ／セッションのプロセスのPSSを合計して上限（1536 MiB）と比べる（`src/tasks/base/annotation/resources/scheduler.py` のsample・`process_snapshot`）。このPRのブラウザ検証が共有fixtureでChromiumを起動し、pytest workerと同じプロセスグループに残したまま同じxdist workerが後続テストへ進むと、Chromiumの分がannotationテストの計測に入る。

1. この仮説を確かめる（ローカルで、ブラウザ検証と当該annotationテストを同じworkerで順に実行する。または計測対象のプロセス一覧を出す）。違っていれば実際の原因を特定する。
2. 原因がブラウザのプロセスなら、ブラウザ検証側で直す。例: fixtureのscopeを当該moduleに限りteardownで確実にChromiumを終了・回収する、またはChromiumを別のプロセスグループ／セッションで起動する。#1786 の既存ブラウザ検証（`tests/e2e/development/test_javascript.py`）にも同じ問題がないかを確かめる。
3. annotationの予算値を上げる、テストをskipする、shardの割当てを手で変える、といった回避はしない。
4. force-push後の対象HEADのCIが成功するまで確認する。CIの再実行は、変更と無関係と確認できた失敗に限る。

最終報告: 原因の確認方法と結果、修正内容、CI結果（HEAD・run）。目安は最大2時間。
