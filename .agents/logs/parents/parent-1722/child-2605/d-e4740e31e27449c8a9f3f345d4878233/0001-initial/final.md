#2605 の SceneResult v3、`event_detection` の3方式、multiview-v3 生成コマンドを実装し、`codex/parent-1722-issue-2605-primary` の `0fb9845e8474f5d1e7b00ed735327115986ebaf0` にコミットしました。pre-commit の Ruff・mypy は通過しました。

- 実データの `clip_057` で注釈・規則・3クラスRefinerをCPU実行し、3方式とも元のv2配列とバイト一致するv3出力を確認しました。
- 69件の注釈ファイルから、打球421・バウンド457・ネット接触35の計913イベントと、判定不能1,885フレームを確認しました。clip_026・034・036・039は現行注釈にcam0基準の記載があり、元映像と重ね合わせでも手前が負のy側と確認しました。
- 分割はseed 42で、train・val・testが47・11・11クリップ、ネット接触ありは22・5・5クリップです。指定の採点用6クリップをtestに固定し、過去の調整用2クリップはtestから除外しました。

コミット済みHEADで回帰テスト、69クリップの生成、同じコマンドでの再検証を行うローカルジョブを起動しました。全69出力の照合とPR作成は、その結果を受け取る継続ターンで行います。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T044546Z_0c1fb6689a8e_scene-events-v3"]}}