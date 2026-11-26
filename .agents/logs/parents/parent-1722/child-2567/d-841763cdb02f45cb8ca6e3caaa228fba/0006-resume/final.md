[PR #2571](https://github.com/Motoki0705/tennis-lab/pull/2571) の同一HEAD `4afdccc7c348a946cc5bb86aac583ae220ce652f` で失敗ジョブを再実行し、[CI第2回](https://github.com/Motoki0705/tennis-lab/actions/runs/45732828428/attempts/2) がキューに入ったことを確認しました。

失敗した注釈テストは、同じHEADの隔離環境では50.22秒で通過し、rightカメラの再開も終了コード0でした。前回CIでも同テストは成功していました。今回のCIでワーカーが終了した理由は未特定です。 [検証記録](https://github.com/Motoki0705/tennis-lab/pull/2571#issuecomment-7180049276)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2571,"head":"4afdccc7c348a946cc5bb86aac583ae220ce652f","resume_on_success":false}}