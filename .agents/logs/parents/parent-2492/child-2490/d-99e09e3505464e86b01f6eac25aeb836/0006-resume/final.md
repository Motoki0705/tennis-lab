Sapiens2-1B／5B の PT32・PT16・TRT16 全6条件を、ユーザー判断により GPU 時間の制約で対象外としました。精度・速度欄は空欄です。 [対象外の判断 #2492](https://github.com/Motoki0705/tennis-lab/issues/2492)

- **5B PT32** は CUDA ロードに297.37秒かかり、PyTorch のピーク割当は20.54 GB、CUDA が報告した GPU 容量は17.07 GBでした。親の記録では超過分に WSL の共有メモリが使われました。
- **1B PT32** は約8時間、全26,276件の約34%で打ち切りました。約3.2秒／人は親が動画の読み込み位置から見積もった値です。保存された予測は0件で、残る4件の精度ジョブは実行前に取り消されました。
- **1B・5B TRT16** は TensorRT 10.14 のパーサーが `ReduceMean` の軸入力に initializer を要求し、両ビルドが失敗しました。

統合用コミット `fce4d4b2c428b4064ff1bebd48e01f8f3f60767d` を `codex/parent-2492-issue-2490-sapiens2` に push しました。テスト51件と pre-commit が通過しました。ViTPose-H・RTMPose・PMPose の比較へ引き継ぐ対象外の一覧と、本番採用前のライセンス確認3点を残しました。 [比較記録](https://github.com/Motoki0705/tennis-lab/blob/fce4d4b2c428b4064ff1bebd48e01f8f3f60767d/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2.md) [実行記録](https://github.com/Motoki0705/tennis-lab/blob/fce4d4b2c428b4064ff1bebd48e01f8f3f60767d/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2_attempts.json)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}