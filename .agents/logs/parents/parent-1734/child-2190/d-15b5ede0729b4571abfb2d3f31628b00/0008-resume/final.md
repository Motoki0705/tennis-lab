revision 4の受入を満たし、[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)を更新しました。ready for reviewです。

- branch: `codex/court-acceptance-postprocess`[local path]`9507c5882`
- 全3,545画像・8,530領域で、旧処理と採否・選択領域・Hが完全一致。
- 新方式は明示設定で保持し、中間2方式を削除。knowledge 75・summaryを更新。
- ローカル245ケース成功、CUDA専用1件skip。pre-commitと最終HEADのPython・Knowledge CIも成功。

CUDA実機は未確認です。1frame精度目標は#2191へ移管済み。merge・closeは行っていません。validatorは未指定、試行／完了とも0回です。