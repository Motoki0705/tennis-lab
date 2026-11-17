[PR #2155](https://github.com/Motoki0705/tennis-lab/pull/2155) を **ready for review** にしました。`codex/scene-x3d-dpt-ball` / `c4ea38b63`。

mainへrebaseし、knowledgeを000043へ再採番しました。人間の受入判断を反映し、比較数値・許容差は維持しています。動画11本を旧E:から新D:へコピーしてSHA一致を確認し、scene用checkpointも存在・SHA一致を確認しました。

pytest **1,347件成功**、pre-commit成功、knowledge検査0 errors。最終HEADの **CI Python・Knowledge・Web UIも成功**です。独立validatorは指定なし、試行／完了0／0回。

残件は、人間による各clipの確認・court失敗への判断と、親の統合レビューです。merge・issue closeは行っていません。