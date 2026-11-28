# 保存済み3D追跡からの SceneResult

`python -m src.tennis_scene.multiview_scene` は、統合済みの `multiview_pose_tracks` v1を
選手の同一性に使う明示的なrecipeです。新経路の前段は
[フレーム別の3D候補](../multiview_pose/README.md) と [時間追跡・選手枠統合](../multiview_tracking/README.md) にあります。
既定recipeは [run_video_batch と run_pipeline](../README.md#標準経路) から選べます。
`prepare` と `run` の `--progress` には、標準recipeで実行したバッチの記録を指定します。
保存済み設定から使うのは、パスのルート、HMR2・GVHMR・SMPLの資産と実行設定、crop拡張、
サンプリング上限、人物可視性と身体配置の設定です。出力はSceneResult v2です。

選手IDと元検出行でカメラごとのcrop、COCO17関節、実観測maskを組み立てます。
`body_view_selection` はその観測で代表カメラと区間を決め、`hmr2_features`、`gvhmr`、
`body_placement`、`scene_assembly` は既存の処理を使います。身体配置へ渡す骨格は実測3Dです。
補間関節とフレーム・時間IDの曖昧さは、元検出行と共に `multiview_scene_inputs` に保存します。
SceneResult のsidecarの `player_reconstruction_method` と `multiview_inputs` から方式と品質の来歴を確認できます。

#2458で採用されたカメラと `ball_triangulation` を固定入力とし、元artifactのIDと記述子SHA-256を保存します。
公開前には元シーンの球配列と照合します。選手人数・元行・カメラ行列が入力と一致しない場合はclipを停止します。

## 実行と確認

`I2363_RUN` は `i2363-all-v1`、`POSE_OUTPUT` は #2444 の `outputs/i2444`、
`TRACK_OUTPUT` は #2445 の `outputs/i2445` です。リポジトリのrootで実行します。

```bash
args=(
  --dataset "$I2363_RUN/inputs/dataset"
  --progress "$I2363_RUN/all/progress/d1df39596d4f47f1896cb535191e7040/progress.json"
  --tracks "$TRACK_OUTPUT/integrated/meiji4"
  --candidates "$POSE_OUTPUT/refine/timing/meiji4"
  --output outputs/meiji-multiview-scene
  --players-per-side 2
)
.venv/bin/python -m src.tennis_scene.multiview_scene prepare "${args[@]}"
```

`prepare` はCPUで全クリップの入力とcropを確認します。GPUの `run` は共有training queueへ登録します。
`--clip video_000/clip_000 --clip video_000/clip_020` で対象を絞れます。GPUのモデル設定は
指定した当初のバッチ実行記録に従います。出力は `--output/<video_id>/<clip_name>/` のclip storeで、
`exports/<scene_assembly ID>/scene.npz` にそのclipのSceneResultを保存します。

完了表は `run` が保存します。既存のSceneResultから表を再集計するときは、同じ出力先を指定してCPUで実行します。

```bash
.venv/bin/python -m src.tennis_scene.multiview_scene report "${args[@]}"
```

新経路を採用したデータセットでは、`report "${args[@]}" --legacy-generation before-multiview` を指定します。
採用時に記録した旧indexとexportを照合し、保存時の世代を旧経路の集計に使います。
採用前のデータセットでは `--legacy-generation active` が既定です。
出力先の `legacy-completion.csv` と `multiview-completion.csv` に、clipごとの結果と不足理由を保存します。
`report` に `--clip` を渡すと、その対象だけで両方のCSVを書き直します。全件を確認するときは `--clip` を省略します。
GVHMRの身体フレーム数は実際に選ばれた約30 fpsの元フレーム番号を数えます。
身体配置は有効なメッシュmaskで採点し、全員が揃う割合とボール3Dの割合の分母は元フレーム数です。
完了には各側の指定人数、各IDのGVHMRと有効な身体配置、sceneに入った有効なボール3Dを要します。
未使用視点の残差は、採用IDに対応する実2D関節のうち、その3D関節のinlierに入らなかった視点で集計します。
対応2D関節が得られた点数を併記します。

採用先を更新する判断後は同じCLIの `publish` を実行します。clipごとに現在の採用参照を照合し、
成果物と切替前後のポインタを保存します。[共用の切替処理](../pipeline/README.md#成果物) は、
`scene.json` と `annotation.json` の読み戻しが済むまで、保留ファイルを通じて復旧記録を示します。
中断後の読込みには復旧記録と操作が表示されます。同じ引数で `publish` を再実行すると切替を続けられ、
`restore` では切替前に戻せます。通常の `prepare`・`run` は別ルートに書き出します。
