[draft PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) に更新済みの #2498・#2497 を取り込み、ViTPoseと共通adapterを `measure_forward()` に揃えました。補助担当にも仕様の変更点を共有しました。 [#2490の仕様更新](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7131482779)

前回CIで失敗した3件を含む38件がローカルで成功しました。pre-commitとknowledge検査も通過し、新HEAD `33ae11e6ff421a61d85aac139fa7a0fbfab1da39` のCIが起動しました。ViTPose-HのGPU採点はCI確認後に共有queueへ登録します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"33ae11e6ff421a61d85aac139fa7a0fbfab1da39","resume_on_success":true,"next_step":"固定疑似BBoxでViTPose-Hのflipあり・なしの精度採点を共有GPU queueへ登録する。TRT16はflip設定とprofileが一致するengineを照合する。速度には #2496 の基準値を使い、補助担当の成果を同じPRへ統合する。"}}