新方針で**全69clip・4cameraのコートを利用可能**にしました。[PR #2222](https://github.com/Motoki0705/tennis-lab/pull/2222) はReady for reviewです。

- branch: `codex/court-video-shared`
- HEAD: `a1b7029c5`
- `unobservable_policy: warn|fail`を追加。既定は`warn`。確かな10/25px超過と共通H不成立は失敗を維持します。
- H・診断成果物のhashと測定値は不変です。

|camera|使用可|warning|失敗|
|---|---:|---:|---:|
|cam0|69|48|0|
|cam1|69|12|0|
|cam2|69|30|0|
|cam3|69|69|0|

追加12ケース、関連75・508ケース、pre-commit成功。[最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44078330312)・Knowledge CIも成功しました。初回の既存storage-tier競合テスト失敗は、同じHEADの再実行で成功し、経緯をPRへ記録しています。validatorは未指定・0回です。

[run 78](https://github.com/Motoki0705/tennis-lab/blob/a1b7029c52ad212b6954ae31fef8589678a78679/knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md) と [更新済みgallery](https://github.com/Motoki0705/tennis-lab/blob/a1b7029c52ad212b6954ae31fef8589678a78679/outputs/tennis_scene/evaluate/court-video-drift-lines-20261117/policy/review/index.html) に、人間の確認範囲（3clip×4camera＋原寸crop）と新方針を記録しました。**clip途中のcamera移動を自動で見逃す可能性は残ります。**

#1752接続・merge・closeは行っていません。