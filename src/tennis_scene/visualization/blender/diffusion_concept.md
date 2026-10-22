# Ball Diffusion の構想

三角測量点と欠測maskを条件に3D軌道を補完する構想を、保存済みBall 3D shotの続きとして
示します。**学習済みモデルの推論結果ではありません。** `ball_refiner_3d` の学習・推論は
呼び出しません。条件は取り込んだ注釈を上流に持つ実データ、軌道と生成過程は模式表現、
全体は構想です。「構想」バッジと色の凡例を全場面に表示します。

2026-10-21にユーザーが960×540のプレビューを確認し、構成・尺・seed・表示をそのまま採用しました。
最終版の既定は1920×1080・30fps・64samples、GIFは幅960・15fpsです。
プレビューから変更するのは描画解像度とsamplesだけです。

## 入力と数値計算

入口は `src.tennis_scene.scripts.export_blender_diffusion`、設定の正本は
[export_blender_diffusion.yaml](../../configs/export_blender_diffusion.yaml) です。
`input` はARTIFACT root相対のBall 3D deliveryで、`bundle/`・`shot/ball_rays.blend`・
`shot/shot_manifest.json` を読み取りだけで使います。bundleのmanifest・数値配列、shotのmanifest・
blend、選択した静止plate、フォントを入力hashに記録します。
delivery直下の `manifest.json` は入力に含めません。必要なファイル・項目が欠けていれば停止します。
既定は採用済みMeiji `[282,942)` のdeliveryです。別入力は明示して指定します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_diffusion \
  paths.artifact_root=/absolute/main/outputs \
  paths.external_asset_root=/usr/share/fonts \
  paths.output_root=/absolute/task-worktree/outputs \
  output=tennis_scene/visualize/export_blender_diffusion/meiji-concept-final
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

## 上流の生成物から再生成する

採用素材と同じstore・区間・カメラで、新しいBall 3D bundleとshotを作る例です。
worktreeのrootをcwdにし、`MAIN_OUTPUTS` とqueueのパスを実環境の絶対パスへ置き換えます。
run名は未使用のものにします。上流の設定・検証の詳細は[Ball 3Dの手順](README.md)を参照してください。

```bash
MAIN_OUTPUTS=/absolute/main/outputs
TASK_OUTPUTS="$PWD/outputs"
RAYS_RUN=tennis_scene/visualize/export_blender_rays/meiji-rays-rebuild
DIFFUSION_RUN=tennis_scene/visualize/export_blender_diffusion/meiji-concept-rebuild
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  paths.artifact_root="$MAIN_OUTPUTS" paths.output_root="$TASK_OUTPUTS" \
  output="$RAYS_RUN"
TRAINING_QUEUE_DIR=/absolute/main/.training_queue \
  bash .agents/skills/training-queue/scripts/training_queue.sh add \
  "blender --background --factory-startup --threads 4 --python-exit-code 1 \
    --python '$PWD/src/tennis_scene/visualization/blender/shots/ball_rays.py' -- \
    --bundle '$TASK_OUTPUTS/$RAYS_RUN/bundle' --output '$TASK_OUTPUTS/$RAYS_RUN/shot' \
    --camera-id cam1 --time-mode continuous --two-d-seconds 3 \
    --pullback-seconds 3 --playback-seconds 5 --samples 64" \
  --name ball-rays-for-diffusion --provider codex --session "$CODEX_THREAD_ID" \
  --issue 1214 --resource half
```

workerは親が管理します。queueの `list` で成功を確認した後、生成された
`bundle/manifest.json`・`shot/shot_manifest.json`・`shot/ball_rays.blend` をそのまま渡します。
上のshot作成は `--render` を指定せず、Diffusionに必要なsceneを保存します。
Windows版Blenderではコマンド中の実行ファイルとパスを `wslpath -w` で変換してください。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_diffusion \
  paths.artifact_root="$TASK_OUTPUTS" paths.output_root="$TASK_OUTPUTS" \
  paths.external_asset_root=/usr/share/fonts \
  input="$RAYS_RUN" output="$DIFFUSION_RUN"
BUNDLE="$TASK_OUTPUTS/$DIFFUSION_RUN/bundle"
SOURCE_BLEND="$TASK_OUTPUTS/$RAYS_RUN/shot/ball_rays.blend"
SHOT_DIR="$TASK_OUTPUTS/$DIFFUSION_RUN/shot"
```

引き継ぐ時刻はshotの `source_frames[-1]` です。`camera_continuity.source_frame` と
`source_output_frame` に元動画・出力のframe番号を残し、全カメラの同じsource frameのplateを選びます。
例えば `[282,942)` を30fps・1xで表示した最終frameは、元60fpsでは940、59.94006fpsでは941です。
区間の `stop-1` から推定しません。以下の撮影・検証へ進めば、上流の納品manifestを別途作らず再生成できます。

## 撮影と検証

次のBlenderコマンドは **training queueへ登録して** 実行します。
`BUNDLE`・`SOURCE_BLEND`・`SHOT_DIR` は絶対パスにします。`SOURCE_BLEND` は入力deliveryの
`shot/ball_rays.blend` です。Windows版Blenderでは `wslpath -w` でそれぞれのパスを変換します。
入力のhashがexport時と異なれば停止します。

```bash
blender --background --factory-startup --threads 4 --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/ball_diffusion.py -- \
  --bundle "$BUNDLE" --source-blend "$SOURCE_BLEND" --output "$SHOT_DIR" \
  --samples 64 --resolution-percentage 100 --render
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
最終版は必ずプレビューと別のrunへ出力します。既定configのまま上のコマンドを使うと
1920×1080・64samplesになります。確認用の960×540は `--samples 8 --resolution-percentage 50` で
生成できます。seed・軌道・尺・カメラ・材質・構想バッジは両者で共通です。
queueは `TRAINING_QUEUE_DIR=<main repoの絶対パス>/.training_queue`、このworktreeをcwdとして
`--provider codex --session "$CODEX_THREAD_ID" --issue 1214 --resource half` で登録します。
workerは親が管理します。実行driverとqueueログは各runへ保存します。

Blender側では入力 `.blend` の最終frameを評価し、カメラ・コート・ネット・錐体をそのまま残します。
以前の光線・検出リング・球・trailを取り除き、画像面には最終source frameの静止画を使います。
完成軌道・ノイズ・tube頂点・球の各時刻の位置はすべてexporterで計算し、Blenderは描画だけを行います。
既存の `common.py` と `ball_rays.py` の表示部品を読み取りだけで再利用しています。

採用した構成は次の15秒です。全区間の模式軌道を9秒で再生するため、最後の場面は約1.226倍速です。
その速度倍率とsource frame対応はmanifestに明記します。

| 秒 | 場面 | 代表frame候補 |
|---|---|---|
| 0–2 | 三角測量点と欠測 | 30 |
| 2–6 | ノイズの曲線が軌道へ整う | 60（初期）・120（途中）・179（完成） |
| 6–15 | 完成軌道上で球を再生 | 300 |

検証スクリプトは全450frameの評価済みtube頂点・表示状態と、表示中の球、前shotからのカメラ・錐体の連続性、
条件の位置・色分け・由来、常時表示する構想バッジを調べます。座標許容誤差は1e-5mで、不一致は
JSONへ理由を残し非ゼロで終了します。`ffprobe -count_frames` とMP4の全frame復号も確認してください。
最終版のMP4は1920×1080・30fps・450frame・15秒を要求し、全frameを復号します。
GIFは最終版のMP4から960×540・15fpsで生成し、全frameの復号と尺も確認します。
静止画は最終版のframe 30（条件）と179（完成軌道）から取り出します。
各素材のSHA-256をdeliveryの `manifest.json` と `SHA256SUMS` に保存します。

`.blend` の外部依存は `bundle/process.pc2`、`bundle/badge.png`、`bundle/plate_0.jpg`・
`plate_1.jpg`・`plate_2.jpg` です。コート表面は `.blend` に内包されています。
再配布では `bundle/` と `shot/` を一緒に保持し、移動後は外部参照先も確認してください。
数値の再検証にはその他の `.npy` と `bundle/manifest.json` も必要です。
入力の `ball_rays.blend` と日本語フォントはexport・shot再構築の際に使い、完成した `.blend` の
再描画では新しいbundle内のcache・画像を使います。

## CPUテスト

```bash
flock /home/kamimura/.local/state/tennis-lab-agents/parent-1204/pytest.lock \
  .venv/bin/python -m pytest tests/unit/tennis_scene/visualization/blender -n 4 -q
flock /home/kamimura/.local/state/tennis-lab-agents/parent-1204/pytest.lock \
  .venv/bin/python -m pytest tests/unit/utils/configuration tests/unit/utils/test_output_layout.py \
  tests/e2e/development/test_architecture_boundaries.py -n 4 -q
```

境界テストは上流exporterの実生成物と時間軸関数を使い、60fps・59.94006fpsの最終frame・
plate選択・入力hashを検証します。上流が生成しない空の納品manifestは用意しません。
