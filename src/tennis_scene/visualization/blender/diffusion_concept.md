# Ball Diffusion の構想

三角測量点と欠測maskを条件に3D軌道を補完する構想を、保存済みBall 3D shotの続きとして
示します。**学習済みモデルの推論結果ではありません。** `ball_refiner_3d` の学習・推論は
呼び出しません。条件は取り込んだ注釈を上流に持つ実データ、軌道と生成過程は模式表現、
全体は構想です。「構想」バッジと色の凡例を全場面に表示します。

## 入力と数値計算

入口は `src.tennis_scene.scripts.export_blender_diffusion`、設定の正本は
[export_blender_diffusion.yaml](../../configs/export_blender_diffusion.yaml) です。
`input` はARTIFACT root相対のBall 3D deliveryで、`bundle/`・`shot/ball_rays.blend`・
`shot/shot_manifest.json`・deliveryの `manifest.json` を読み取りだけで使います。
既定は採用済みMeiji `[282,942)` のdeliveryです。別入力は明示して指定します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_diffusion \
  paths.artifact_root=/absolute/main/outputs \
  paths.external_asset_root=/usr/share/fonts \
  paths.output_root=/absolute/task-worktree/outputs \
  output=tennis_scene/visualize/export_blender_diffusion/meiji-concept-preview
```

日本語フォントはEXTERNAL_ASSETで指定します。上の例はNoto CJKがある環境向けです。
`badge_font` はそのrootからの相対パスで、存在しなければ停止し、別フォントへ切り替えません。
出力はrepo共通の `<OUTPUT>/tennis_scene/visualize/export_blender_diffusion/<run-id>/` に従います。
親issueの旧 `presentation/<機能名>` 案とは階層が異なり、既存のrun契約を優先します。

- 完成軌道: 秒を横軸とした座標ごとの3次平滑化スプライン。有効な点をすべて使い、
  各点の3D距離が `point_tolerance_m` 以下になるまで、要求した平滑化係数を32回の二分法で
  下げます。欠測も同じスプライン上で評価します。観測のない両端の外挿は拒否します。
  物理積分器は初期状態・飛行区間・接触の推定が別途必要なため、この説明用処理では使いません。
- 初期ノイズ: seed固定の独立な正規乱数制御点をtanhで領域内に収め、PCHIPでつなぎます。
  `smoothstep` の重みで、ノイズ曲線全体を完成軌道へ補間します。Flowの実際の逆過程ではありません。
- 条件: 水色が有効点、赤が外れ値のviewを含みながら採用された三角測量点です。
  欠測には測定された3D位置がないため、黄色の中空マーカーを**模式軌道上**に置きます。
  この位置は `schematic` と記録し、元のNaNと理由コードも保持します。

計算結果は `bundle/manifest.json` に残します。入力・フォントのSHA-256、git commit、解決済みconfig、
`ball_source`、点の誤差、補完frame、乱数と補間の規則、各要素の由来区分を含みます。
配布するときは `bundle/` と `shot/` をまとめて保持してください。

## 撮影と検証

次のBlenderコマンドは **training queueへ登録して** 実行します。
`BUNDLE`・`SOURCE_BLEND`・`SHOT_DIR` は絶対パスにします。`SOURCE_BLEND` は入力deliveryの
`shot/ball_rays.blend` です。Windows版Blenderでは `wslpath -w` でそれぞれのパスを変換します。
入力のhashがexport時と異なれば停止します。

```bash
blender --background --factory-startup --threads 4 --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/ball_diffusion.py -- \
  --bundle "$BUNDLE" --source-blend "$SOURCE_BLEND" --output "$SHOT_DIR" \
  --samples 8 --resolution-percentage 50 --render
blender --background "$SHOT_DIR/ball_diffusion.blend" --threads 4 --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/validate_diffusion.py -- \
  --bundle "$BUNDLE" --report "$SHOT_DIR/validation.json"
ffmpeg -n -framerate 30 -start_number 0 -i "$SHOT_DIR/frames/%05d.png" \
  -c:v libx264 -threads 4 -crf 16 -pix_fmt yuv420p -movflags +faststart "$SHOT_DIR/ball_diffusion.mp4"
ffmpeg -n -i "$SHOT_DIR/ball_diffusion.mp4" -filter_complex_threads 1 \
  -filter_complex 'fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none' \
  -loop 0 "$SHOT_DIR/ball_diffusion.gif"
```

`--render` を外すとsceneの生成・保存だけを行います。既存のbundle/shotは上書きしません。
プレビューは960×540です。ユーザー確認後、最終版は `--samples 64 --resolution-percentage 100` で
1920×1080を描き直します。configの値を変えた場合は新しいbundleもexportしてください。

Blender側では入力 `.blend` の最終frameを評価し、カメラ・コート・ネット・錐体をそのまま残します。
以前の光線・検出リング・球・trailを取り除き、画像面には最終source frameの静止画を使います。
完成軌道・ノイズ・tube頂点・球の各時刻の位置はすべてexporterで計算し、Blenderは描画だけを行います。
既存の `common.py` と `ball_rays.py` の表示部品を読み取りだけで再利用しています。

既定は次の15秒です。全区間の模式軌道を9秒で再生するため、最後の場面は元の時間軸より速くなります。
その速度倍率とsource frame対応はmanifestに明記します。

| 秒 | 場面 | 代表frame候補 |
|---|---|---|
| 0–2 | 三角測量点と欠測 | 30 |
| 2–6 | ノイズの曲線が軌道へ整う | 60（初期）・120（途中）・179（完成） |
| 6–15 | 完成軌道上で球を再生 | 300 |

検証スクリプトは全450frameの評価済みtube頂点・表示状態と、表示中の球、前shotからのカメラ・錐体の連続性、
条件の位置・色分け・由来、常時表示する構想バッジを調べます。座標許容誤差は1e-5mで、不一致は
JSONへ理由を残し非ゼロで終了します。`ffprobe -count_frames` とMP4の全frame復号も確認してください。
プレビュー・代表静止画は候補として親へ渡し、採否はユーザーが決めます。

## CPUテスト

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/visualization/blender -n 4 -q
.venv/bin/python -m pytest tests/unit/utils/configuration tests/unit/utils/test_output_layout.py \
  tests/e2e/development/test_architecture_boundaries.py -n 4 -q \
  -k 'not test_branch_contains_configuration_contract_merge'
```

上の除外は履歴書き換え前の祖先commitを要求する既知のテストです。親指定により除外し、
履歴側で解決します。テスト本体や既存shotの実装は変更しません。
