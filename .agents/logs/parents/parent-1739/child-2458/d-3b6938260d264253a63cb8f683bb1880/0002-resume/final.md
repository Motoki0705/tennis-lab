`i2363-all-v1/inputs/dataset` の全69クリップで、`scene.json` と `annotation.json` を新版に切り替え、読み戻しました。55件は既存のカメラ幾何を使い、ボール以外の1,100配列が旧版とSHA-256で一致しました。残る14件は既存校正と共有コート向きでボールを復元し、選手軸を空にして停止理由を記録しました。この14件のカメラ幾何は人物による再検証を行っていません。

- **ボール3D有効率は45,958／46,228フレーム（99.42%）**。欠測270フレームはすべて clip_064 のview不足で、solver不成立は0件でした。
- 有効3Dの内訳は2 viewが774、3 viewが1,559、4 viewが43,625フレームでした。
- 3Dに寄与した2D点は `observed` 121,840、`interpolated` 6,468、`candidate_supported` 6,830、`estimated` 45,587件でした。画面外の4,136点は欠測として扱いました。
- 全件コマンドは18分52秒、ピークRSSは2.46 GiB、独立した読戻しは62.45秒でした。テスト140件、pre-commit、knowledge検査が通りました。独立した3D正解に対する精度は未計測です。

[PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462) にクリップ別集計、復元手順、再投影3例を掲載しました。ブランチは `codex/parent-1739-issue-2458-primary`、HEADは `7cb45ada9de0c42bed8067b6ed96668e043e5e5e` です。CI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2462,"head":"7cb45ada9de0c42bed8067b6ed96668e043e5e5e","resume_on_success":false}}