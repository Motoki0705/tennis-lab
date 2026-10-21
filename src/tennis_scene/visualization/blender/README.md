# Ball 3D の光線による説明

保存済みcomponent storeから2D観測と三角測量の結果を読み、元動画と2D観測のリングから始め、
カメラを引いて画像面・光線・球・軌跡の3D再生へつなぎます。推論・三角測量の再実行やstoreの更新はしません。
入力映像からシーンへ切り替える既存shotは[revealの手順](export/README.md)を参照してください。

錐体と画像面は [shots/camera_planes.py](shots/camera_planes.py) の
`camera_frustum(camera, color=..., depth=1.5) -> (root, plane)` で作ります。
`root` はカメラ中心にあり、local軸はOpenCVの右・下・前です。
`image_plane_point(camera, uv_px, depth=1.5)` は元画像pxを同じlocal画像面に写します。
他のshotもこの2関数を再利用できます。コート・ネット・材質・compositorは既存の
`shots/common.py` を使います。

## 採用した構成と再生成

設定の既定は、ユーザーが採用した Meiji の **continuous** 構成です。

| 項目 | 採用値 |
|---|---|
| store（ARTIFACT root相対） | `tennis_scene/evaluate/i931-default-meiji-clip000-20260927/store` |
| 入力区間 | `[282,942)`、660frame、59.94006fps |
| カメラ | `cam1` に固定。区間内の観測数が最多（cam0=569、cam1=642、cam2=607）だったため |
| 2D → 引き → 3D | 3秒 → 3秒 → 約5秒、全場面で1x |
| 出力 | 1920×1080・30fps・331frame（11.033秒）。30fpsへの丸めで3Dは5.033秒 |

`camera_id=null` は新しい素材で観測数による自動選択を要求する指定です。
採用素材では選択結果を固定し、exportの `camera_selection` と撮影のmanifestに件数を残します。

worktreeから共有入力を読み、既定の区間とカメラで新しいbundleを作る例です。
`output` は毎回新しいrun名にしてください。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  paths.artifact_root=/absolute/main/outputs \
  output=tennis_scene/visualize/export_blender_rays/meiji-rays-final
```

次のBlenderコマンドは、同じworktreeをcwdとして共有training queueへ
`--provider codex --issue 1213 --resource half` で登録します。workerは親が管理します。
`BUNDLE`・`SHOT_DIR` は絶対パス、Windows版では対応するUNCパスを渡してください。

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/ball_rays.py -- \
  --bundle "$BUNDLE" --output "$SHOT_DIR" --camera-id cam1 \
  --time-mode continuous --two-d-seconds 3 --pullback-seconds 3 --playback-seconds 5 \
  --samples 64 --render
blender --background "$SHOT_DIR/ball_rays.blend" --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/validate_rays.py -- \
  --bundle "$BUNDLE" --report "$SHOT_DIR/validation.json"
ffmpeg -n -framerate 30 -start_number 0 -i "$SHOT_DIR/frames/%05d.png" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart "$SHOT_DIR/ball_rays.mp4"
ffmpeg -n -i "$SHOT_DIR/ball_rays.mp4" \
  -filter_complex 'fps=15,scale=1920:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none' \
  "$SHOT_DIR/ball_rays.gif"
```

スライド用の小さいGIFは同じ式の `scale=960:-1` で生成します。
代表静止画は連番のframe 44（2D）、134（引きの途中）、255（3D playback）です。
MP4は `ffprobe -count_frames` と `ffmpeg -xerror -i <mp4> -f framemd5 -` で全frameを復号し、
331frame・1920×1080・30fps・11.033秒を確認します。

3D代表静止画の出力frame 255はsource frame 791に対応し、3本ともinlierです。
光線の材質はBLENDED・重なりなしで、破線maskは単独viewだけに適用します。
半径0.012mの線はプレビューで1px未満になるため、960×540・8sampleでは実線が
細切れに見える場合があります。最終版は64sampleを既定とし、1920×1080で確認します。

## 候補とbundle

入口は `src.tennis_scene.scripts.export_blender_rays`、設定の正本は
[export_blender_rays.yaml](../../configs/export_blender_rays.yaml)です。
`input` はARTIFACT root相対、`output` はOUTPUT root相対です。worktreeから共有入力を
読むときは、`paths.artifact_root=/absolute/main/outputs` を指定します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=rank \
  start=null stop=null t0=null camera_id=null window_frames=660
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=export \
  window_frames=660 start=282 stop=942 t0=611
```

既定出力は `<OUTPUT>/tennis_scene/visualize/export_blender_rays/<run-id>/` です。
親の発表用配置案より、repoの共通run契約を優先しています。

- `mode=rank`: `ranking.json` の上位10件を人に提示します。各窓に、外れ値の光線と
  単独viewのframeをそれぞれ要求します。valid率の降順、同率では開始frame順です。
  `t0` は区間の時間的な中央（偶数長では前側）を記録するだけです。新構成は `t0` を
  固定表示しないため、中央の3view以上のinlier・validという旧条件は要求しません。
  `window_frames` は下記の秒数と元fpsから計算した値を明示します。候補を自動採用しません。
  `ranking_suppression=non_overlapping`（既定）は、有効率の順位順に採用済みの
  `[start, stop)` と重なる窓を抑制します。境界で接する窓は許可します。
  `t0_distance` は採用済みの全 `t0` から `window_frames` 以上離れた窓だけを残します。
  抑制してから上位10件に制限し、条件を満たす窓が足りない場合は得られた件数を返します。
  抑制規則・返却件数と、外れ値／単独view／棄却frameの番号・件数をrankingに記録します。
  棄却frameの件数は「可視2view以上で球が無効」のframe数で、単独viewと分けて数えます。
- `mode=export`: 元動画の `[start, stop)` だけを復号し、`bundle/` を作ります。
  区間長は明示した `window_frames` と一致する必要があります。`t0` はschema上の
  時刻メタデータとして同区間内を要求しますが、valid・3view条件はありません。
  元動画のhash・寸法・FPS・frame数を検査します。uvは元解像度のpxです。
- 採用artifactのschema・依存を検査し、完成sceneが `exports.scene` にあれば、
  `ball_3d` と有効maskが三角測量結果に一致することも要求します。
- `ball_source.artifacts` は依存をたどった検出器または注釈importの由来・入力hash・レビュー
  情報を記録します。過去storeの検出artifactへの直接依存と、現行の `ball_points` adapterを
  介した依存を、そのままのlineageで読みます。過去schemaを現行の推論結果とは扱いません。

bundle schemaは `tennis_scene_blender_rays_v1` です。`ray_origin (V,3)`、
`ray_direction (V,T,3)`、`ray_length (V,T)`、`ray_length_rule (V,T)`、
`ray_state (V,T)`、`closest_point (V,T,3)`、`ball_position (T,3)`、
`ball_reason (T,)`、`reprojection_uv / detection_uv (V,T,2)` を `.npy` で保存します。
欠測座標はNaN、状態・理由はuint8です。列挙値はmanifestに記録します。
各cameraのJPEGは `plates/<camera_id>/<source-frame:05d>.jpg` です。

光線の長さは、有効な球を15%越す距離、前方の地面との交差距離、明示的な
`fallback_ray_length_m` の順で決め、適用した規則を保存します。
既存bundleを上書きせず、失敗時は途中のbundleを削除します。

## Blenderでの撮影と検証

Blender 5.2の同梱Pythonで動き、repoはimportしません。CPUだけで使える
[shots/ray_timeline.py](shots/ray_timeline.py) の `RayShot` が設定の正本です。
既定は1920×1080・30fps・64sampleです。旧来の `t0` 固定による段階的な説明は行いません。

1. `two_d`: 実カメラと一致するK・R・tで画像面を画面いっぱいに映し、元映像と
   カメラ色の検出リングを再生します。リングは同じ画素の光線に沿って画像面より手前に置きます。
2. `pullback`: 実カメラの位置から同じ側の俯瞰へ、位置・姿勢・レンズを連続的に変えます。
   画像面は錐体の底面に残り、他のカメラの錐体・画像面とコートも見えてきます。
3. `playback`: 俯瞰を保って全カメラの光線・球・軌跡を再生します。光線は最初から
   bundleの `ray_length` の長さです。最近点の線分・再投影の十字・fade-inは使いません。

`camera_id=None`（CLIの未指定）は、**入力区間全体**で `detection_uv` が有限のframeが
最も多いカメラを選びます。同数ならcamera IDの昇順です。`--camera-id cam1` で固定できます。
選択規則・全カメラの観測frame数・選択カメラの件数を `shot_manifest.json` に記録します。
未知のID・観測のないカメラでは停止し、途中でカメラを切り替えません。
実データの校正と同じく、主点は画像中心、歪み・skewなし、fx=fyを要求します。
選択カメラと出力の縦横比が異なる場合も、切り抜きや余白で近似せず停止します。

時間の接続は `--time-mode continuous|replay` で明示します。

| 設定 | 2D | 引き | 3D playback |
|---|---|---|---|
| `continuous` | 約3秒を1x再生 | 次の約3秒を1x再生しながら引く | 残りの約5秒を1x再生 |
| `replay` | 約5秒の区間全体を1x再生 | 最終source frameを3秒保つ | 同じ区間を最初から1x再生 |

`--two-d-seconds`（continuousは既定3秒、replayはplaybackと同じ秒数）、
`--pullback-seconds`（3秒）、`--playback-seconds`（5秒）で尺を明示します。
元動画の再生速度は**常に1x**です。区間が短くてもスロー・補間・繰り返しで尺を延ばしません。
replayの2Dと3Dは同じ区間なので、両者の秒数が異なる指定は拒否します。

元fpsを `f` とすると、必要な入力は次のframe数です。

- continuous: `ceil(two_d_seconds*f) + ceil(pullback_seconds*f) + ceil(playback_seconds*f)`
- replay: `ceil(playback_seconds*f)`。2Dで同じframeを使い、引きだけ意図的に静止します。

59.94006fpsでは、それぞれ660frame（約11秒）と300frame（約5秒）です。
rank・exportのconfigにも同名の秒数と `time_mode` を指定し、`window_frames` を明示します。
秒数からの計算値と窓長が異なる場合は停止します。

```bash
# continuous: 3秒 + 3秒 + 5秒を1xで。
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=rank \
  start=null stop=null t0=null camera_id=null time_mode=continuous two_d_seconds=3 pullback_seconds=3 playback_seconds=5 window_frames=660
# replay: 同じ5秒の区間を前後で再生。
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=rank \
  start=null stop=null t0=null camera_id=null time_mode=replay two_d_seconds=5 pullback_seconds=3 playback_seconds=5 window_frames=300
```

元fps・秒数・frame数の丸め方・計算した窓長はrank/bundle/shotの `source_window` に記録します。
continuousでは全場面を通して1つの1x時計を使い、場面の境界でも時計をリセットしません。
各出力時刻に最も近いsource frameを選び、欠測は補間しません。30fpsへの丸めにより尺は
1出力frame未満だけ延びる場合があります。`shot_manifest.json` に各場面の出力frame範囲・
実際の秒数・source範囲・速度倍率、全出力frameのsource対応と `ball_source` を残します。
時間の接続方法・区間の採用はプレビューを見て人が決めます。
軌跡は16点・source 2frameおきで、欠測を跨ぎません。
光線の円柱は中点からの相対座標をPC2に保存し、長い棄却光線でも端点の精度を保ちます。
単独viewの破線は長手方向の `ray_axis` 属性によるものです。

採用構成の描画・検証・MP4/GIF化は[再生成手順](#採用した構成と再生成)を参照してください。
`--render` なしでは `.blend` の生成だけを行い、既存の出力先は拒否します。

960×540のプレビューは `--samples 8 --resolution-percentage 50` を付けます。
`--frame-step 3` なら動画化を `-framerate 10 -pattern_type glob -i "$SHOT_DIR/frames/*.png"`
に変更し、再生時間を保ちます。`--frames start:end` は出力frameの範囲を両端含みで限定します。
代表静止画はmanifestの `stages[].output_frames` にある各場面の中央から取り出せます。
候補の区間・カメラ・つなぎ方・静止画の採否は人が決めます。

検証は全出力frameの光線の端点・球・表示状態・材質と、錐体の位置／姿勢・画像面の角・
検出リングの位置・plateのsource frameをbundleと照合します。座標の許容誤差は1e-5 mです。
2D場面では実カメラのK・R・t、画像面が画面全体を覆うこと、リング中心と `detection_uv` の
投影差（許容誤差0.01 px、元画像解像度）も調べます。不一致でも理由をJSONに残し非ゼロで終了します。
`material_samples` には線幅の概算を残します。低解像度で実線が細切れに見えても、
`Dashed` は単独viewだけで有効なことを全frameで検査します。
