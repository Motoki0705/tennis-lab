`origin/main` の `6c9b2bbf` をmergeし、`codex/parent-1739-issue-2458-primary` にpushしました。新HEADは `0b12a3fa27d4cdc49f163981a542bcad91499a21` です。 [PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462)

- `.sequence` の競合を解消し、run IDを維持してノードを `000105` に移しました。
- `scene_export.py` の競合を解消し、v2は `scene_assembly`、v3は新しい `scene_assembly` に依存する `event_detection` を指すようにしました。保存済みv3の注釈を使い、公開・読込・途中停止からの再開・復元を確認しました。旧軌道に基づく推定イベントがあるclipは切替前に停止します。#2470向けの共通APIはpipeline READMEに記載しました。
- `test_fasthmr_recipe.py` の競合を解消し、互換表のハッシュをGit原本と照合しました。完全一致では旧identityを再現し、対象ソースの編集後は互換判定が失効することを確認しました。

CPUテスト160件、pre-commit、knowledge検査758ノードが通過しました。公開・復元の検証は合成fixtureで行いました。PR本文を更新し、GitHub CIを親CLIに引き渡します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2462,"head":"0b12a3fa27d4cdc49f163981a542bcad91499a21","resume_on_success":false}}