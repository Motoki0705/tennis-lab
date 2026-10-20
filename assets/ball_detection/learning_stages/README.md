# Ball Learning の実測系列

`scalars.json` は、採用済み動画が使用する事前学習・事後学習の
`train-summary.jsonl` から必要な列だけを抽出したものです。
原本SHA-256・バイト数・抽出列・件数・step範囲・生成物SHA-256の正本は
[`derivation.json`](derivation.json) です。原本のhashは各runの削除記録とも照合します。
元runのファイルは変更・復元しません。

両runとも、完了update 50〜60,000（50刻み）の1,200行を保持しています。
`epoch / global_step / train_loss / learning_rate`、事後学習では追加で
`encoder_frozen / encoder_learning_rate` を保持します。
数値の丸め・間引き・補間・平滑化・正規化・schedulerからの再計算は行いません。
表示時の過去20点移動平均と段階別の正規化は従来の描画処理に残しています。

残存する `metrics.jsonl` は各runにepoch末の10行しかなく、lossはepoch集約値です。
50updateごとのloss・LR・凍結状態を代替できず、使うと採用済みの曲線が変わります。
評価Common/Full値、設定、best checkpointは引き続き既存bundleから読み、重複保存しません。
CLIと表示仕様は [可視化ガイド](../../../src/tasks/ball_detection/visualization/README.md) を参照してください。

原本を別途保持している場合の抽出コマンド（repo rootで実行）:

```bash
.venv/bin/python -m assets.ball_detection.learning_stages.export_scalars \
  --pretraining-raw /absolute/path/pretraining/train-summary.jsonl \
  --pretraining-bundle knowledge/runs/run-i1050-convnext-v2-pretrain-s42 \
  --posttraining-raw /absolute/path/posttraining/train-summary.jsonl \
  --posttraining-bundle knowledge/runs/run-i1049-convnext-query-posttrain-s42 \
  --output-dir outputs/learning-stages-extraction
```

原本のhash・サイズ、必要列、1,200行、stepの列が食い違えば出力前に停止し、
既存の抽出結果は上書きしません。原本はGit履歴の存在に依存せず、呼出側が明示します。
可視化も抽出結果のhash、原本の削除記録、列・件数・stepを検証し、不備を別入力で補いません。
