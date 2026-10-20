# Ball 3D の光線による説明

保存済みcomponent storeから2D観測と三角測量の結果を読み、カメラの画像面・光線・球・
最近点の線分・再投影をBlenderで表示します。推論・三角測量の再実行やstoreの更新はしません。
入力映像からシーンへ切り替える既存shotは[revealの手順](export/README.md)を参照してください。

錐体と画像面は [shots/camera_planes.py](shots/camera_planes.py) の
`camera_frustum(camera, color=..., depth=1.5) -> (root, plane)` で作ります。
`root` はカメラ中心にあり、local軸はOpenCVの右・下・前です。
`image_plane_point(camera, uv_px, depth=1.5)` は元画像pxを同じlocal画像面に写します。
他のshotもこの2関数を再利用できます。コート・ネット・材質・compositorは既存の
`shots/common.py` を使います。

## 候補とbundle

入口は `src.tennis_scene.scripts.export_blender_rays`、設定の正本は
[export_blender_rays.yaml](../../configs/export_blender_rays.yaml)です。
`input` はARTIFACT root相対、`output` はOUTPUT root相対です。worktreeから共有入力を
読むときは、`paths.artifact_root=/absolute/main/outputs` を指定します。

```bash
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=rank
.venv/bin/python -m src.tennis_scene.scripts.export_blender_rays \
  input=tennis_scene/evaluate/<run>/store mode=export start=782 stop=872 t0=826
```

既定出力は `<OUTPUT>/tennis_scene/visualize/export_blender_rays/<run-id>/` です。
親の発表用配置案より、repoの共通run契約を優先しています。

- `mode=rank`: `ranking.json` の上位10件を人に提示します。既定90frameの各窓に、外れ値の
  光線と単独viewのframeをそれぞれ要求します。`t0` は中央1/3にある3view以上のinlierを持つ
  有効frameから中央に最も近いものです。同率では開始frame順です。候補を自動採用しません。
  `ranking_suppression=non_overlapping`（既定）は、有効率の順位順に採用済みの
  `[start, stop)` と重なる窓を抑制します。境界で接する窓は許可します。
  `t0_distance` は採用済みの全 `t0` から `window_frames` 以上離れた窓だけを残します。
  抑制してから上位10件に制限し、条件を満たす窓が足りない場合は得られた件数を返します。
  抑制規則・返却件数と、外れ値／単独view／棄却frameの番号・件数をrankingに記録します。
  棄却frameの件数は「可視2view以上で球が無効」のframe数で、単独viewと分けて数えます。
- `mode=export`: 元動画の `[start, stop)` だけを復号し、`bundle/` を作ります。
  `t0` は同区間内の有効frameで、3view以上のinlierが必要です。元動画のhash・寸法・FPS・
  frame数を検査します。uvは縮小JPEGの座標ではなく元解像度のpxです。
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

Blender 5.2の同梱Pythonで動き、repoはimportしません。`RayShot` に尺・色・半径を
まとめています。既定は1920×1080・30fpsです。frame 0–315で原理を説明し、その後
入力区間を元の秒数で再生します。90frame・59.94fpsの素材では全体約12秒になります。
画像面での説明の後、再生開始時には3D全体へカットし、短い外れ値・欠測も見える構図で
ゆっくり俯瞰へ引きます。
30fpsの各出力時刻に最も近いsource frameを選び、対応は `shot_manifest.json` に残します。
欠測を補間せず、球の軌跡は16点・source 2frameおきで、欠測を跨ぎません。

次のBlenderコマンドは共有training queueの `--resource half` へ登録します。
`BUNDLE`・`SHOT_DIR` は絶対パス、Windows版Blenderでは対応するUNCパスを渡してください。
新規出力先が必要です。`--render` なしでは `.blend` の生成だけを行います。

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/ball_rays.py -- \
  --bundle "$BUNDLE" --output "$SHOT_DIR" --render
blender --background "$SHOT_DIR/ball_rays.blend" --python-exit-code 1 \
  --python src/tennis_scene/visualization/blender/shots/validate_rays.py -- \
  --bundle "$BUNDLE" --report "$SHOT_DIR/validation.json"
ffmpeg -n -framerate 30 -start_number 0 -i "$SHOT_DIR/frames/%05d.png" \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart "$SHOT_DIR/ball_rays.mp4"
ffmpeg -n -i "$SHOT_DIR/ball_rays.mp4" \
  -filter_complex 'fps=15,scale=960:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse' \
  "$SHOT_DIR/ball_rays.gif"
```

低負荷プレビューは `--samples 8 --resolution-percentage 25 --frame-step 3` を付けます。
frame step 3なら動画化は `-framerate 10 -pattern_type glob -i "$SHOT_DIR/frames/*.png"`
に変更し、再生時間を保ちます。`--frames 150:240` は範囲を両端含みで限定します。
静止画は連番の段階中央（overview=45、detections=120、rays=180、ball=240、
reprojection=300、playback=315以降）から候補を取り出します。採否は人が決めます。

検証は全出力frameの評価済み光線の端点、球、表示状態、カメラ位置、画像面上の観測と
再投影、plateのsource frameを照合します。座標の許容誤差は1e-5 mです。
不一致でも失敗理由をJSONに保存し、非ゼロで終了します。
材質についても全frameで割り当て先・カメラ色／赤／灰・不透明度・破線フラグを検査します。
伸長途中の材質と画面上の線幅の概算は `growth_material_samples` に記録します。
