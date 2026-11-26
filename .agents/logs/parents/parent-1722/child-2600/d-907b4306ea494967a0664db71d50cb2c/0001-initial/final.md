#2600 の3方式の入力融合、SMPL前処理、追加学習・評価CLIを実装し、ドラフト [PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613) を作成しました。HEAD は `820732011404a0c5df8b7867060d6a2420f998dc` です。

- 学習側の `clip_057` で、3方式の初期座標とイベント logits が事前学習モデルと全285フレームでビット一致しました。手首192点の位置差は、描画済みメッシュに対して中央値2.35 cmでした。SMPL欠測時にrootを保持し、手首速度が欠測をまたがないことも確認しました。
- 新規12ケース、既存の単体・統合回帰、CLI設定監査が通過しました。pre-commitも通過し、GitHub CIは進行中です。
- 手首の順運動学には中立SMPL-X骨格を使いました。ポイント内の誤検出は、イベント注釈から推定した区間で数えます。

69クリップのCPU前処理を起動しました。完了後に #2610 の最新の実行証拠と manifest を再照合し、3方式×3 seedとball-only参照の計10件を共有queueに投入します。実映像での比較とknowledge記録はその結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T065934Z_25ce651a49fd_meiji-v3-motion-inputs"]}}