公開資産5点を `ckpt/fasthmr/` に配置し、全ファイルのサイズとSHA-256を確認しました。追加Python依存を導入し、固定版の公式推論モジュール6つをCPUで読み込めました。 [照合記録](https://github.com/Motoki0705/tennis-lab/blob/877f60e6a6aedac30139df61e315f71277ad32b7/outputs/fasthmr-setup/public-assets.json)

初回CIの設定監査エラーを修正し、`codex/parent-2492-issue-2491-primary` の `877f60e6a6aedac30139df61e315f71277ad32b7` をpushしました。同じ差分でローカル76件成功、1件skip、pre-commit通過です。追加の再実行は時間超過で中断し、上記の件数には含めていません。新HEADのCIは実行中です。 [Draft PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501)

**CameraHMR資産待ちです。** `cam_model_cleaned.ckpt`、`smpl_mean_params.npz`、`SMPL_NEUTRAL.pkl` の配置後に実重み検証へ進みます。速度比較は #2496 の固定入力と基準値の公開待ちです。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}