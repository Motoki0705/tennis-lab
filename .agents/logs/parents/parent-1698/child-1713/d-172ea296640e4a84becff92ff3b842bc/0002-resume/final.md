[PR #1769](https://github.com/Motoki0705/tennis-lab/pull/1769)へ修正をpushしました。commitは`1ec8248f7`です。

登録時にpolicyを固定し、復旧時のDEFAULT_POLICY選択を廃止しました。欠損・破損・不一致は明示エラーで停止します。

追加テストは全12ケース成功しました。

- `test_cli_recovery_after_registration_uses_the_scene_policy`：3件。修正前は3件とも失敗。
- `test_registration_freezes_default_or_explicit_policy_and_rejects_replacement`：2件。
- `test_cli_recovery_rejects_missing_or_conflicting_registered_policy`：4件。
- `test_benchmark_cli_uses_its_frozen_condition_before_controller_initialization`：3件。

元workflowの完了まで確認するE2E、pre-commitも成功。[対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42540466202)は全10分割・Lint・`Python`成功です。

親へ指摘解消の判定・merge・merge後mainのCI確認を返します。