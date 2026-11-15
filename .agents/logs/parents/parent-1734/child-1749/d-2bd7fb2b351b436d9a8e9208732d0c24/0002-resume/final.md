[PR #2156](https://github.com/Motoki0705/tennis-lab/pull/2156)を修正しました。  
branch: `codex/scene-shared-plan` / HEAD: `f3a45b22f`

- callbackの試行済み状態と成否を分離。Aの公開失敗を保持し、同じexecutorでBを再開してもAを再通知しません。
- CPU回帰230件成功（今回4件追加）、pre-commit成功。対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43736982272)も`Python`を含む全check成功です。
- baseと最終HEADを別checkout・別プロセス・空storeで独立比較し、31nodeの実行[local path] identity、生成時刻除外）。

比較の[再現スクリプト・固定記録](https://github.com/Motoki0705/tennis-lab/tree/f3a45b22f/tests/support/tennis_scene/entrypoint_comparison_records)と[最終HEADの実行記録](https://github.com/Motoki0705/tennis-lab/pull/2156#issuecomment-6897820695)を保存済みです。

追加validatorは未指定で試行0／完了0。親による指摘解消確認・merge・merge後main CI確認が残ります。