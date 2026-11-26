新recipeを [ドラフトPR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470) に追加しました。ブランチは `codex/parent-1739-issue-2446-primary`、HEADは `508195d5a7a028c431872077465bbec5dcd855d5` です。

- 統合済み3D IDと元の検出行から身体cropを作り、GVHMR・身体配置・SceneResultへ接続しました。採用中のカメラとボールを固定し、元artifactの参照とSHA-256を記録します。採用中のデータセットの参照は今回変更していません。
- CPU準備は69/69本で成功し、全クリップで各側2人のIDと身体cropを確認しました。旧経路は52/69本が完了基準を満たし、17本で人物側に不足がありました。ボール3Dは69本すべてにあり、`clip_064` の有効範囲は982/1,252フレームです。[新旧の途中表](https://github.com/Motoki0705/tennis-lab/blob/508195d5a7a028c431872077465bbec5dcd855d5/outputs/i2446)
- 統合後のソースを一組として照合するキャッシュ互換定義を追加しました。失敗していた既定recipeの回帰テスト2件を含むpytest 46件、設定監査、pre-commitが通りました。最新HEADのCIは結果待ちです。

`clip_000` と `clip_020` のGPUパイロットを共有キューへ投入しました。実モデルの身体とボールを読み戻して確認した後に、全69本を投入します。最終の完了表と定量比較はその後です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204500151283287433_2026-11-24T05-57-28-716712566Z_364832_meiji-multiview-scene-pilot.job"]}}