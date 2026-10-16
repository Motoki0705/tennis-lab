# 保存済み統合シーンのレビュー

`visualize_component_store`は指定した`scene.json`だけを正本として、採用component・入力依存・
source・統合sceneを読み取り専用で表示します。推論・学習・注釈更新は実行しません。
指定以外のstoreから欠けた結果を補いません。

```bash
ROOT=/home/kamimura/projects/tennis-lab
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;2' \
  "$ROOT/.venv/bin/python" -m src.tennis_scene.scripts.visualize_component_store \
  --store "$ROOT/outputs/tennis_scene/evaluate/i964-qualification-r14-20261001/store" \
  --output "$ROOT/outputs/dataset_review_campaign_20261004/tennis_scene/gallery-current" \
  --serve --port 8903
```

`http://127.0.0.1:8903`で開きます。3Dを含む履歴snapshotを確認する場合は、
`--store`を`outputs/tennis_scene/evaluate/i931-default-meiji-clip000-20260927/store`へ、
`--output`を別のgallery directoryへ変更します。これは保存履歴の閲覧であり、現行推論への再利用ではありません。
`--store`はARTIFACT、`--output`はOUTPUTの明示的な境界です。galleryはstoreの外へ出力します。

## 確認の流れ

1. **Source → 採用artifact → 品質**: 元clip・camera・source checksum、保存indexの採用版、
   未生成、履歴schema、古い依存、完成exportの有無を確認します。
   sourceとexportは別の状態です。indexだけでは未生成と実行失敗を区別できない場合があります。
2. **同じsource frameで確認**: RGB3視点と保存2D、maskが有効な3D関節/root/球、
   観測/root/heading/meshのmask・保存棄却理由を照合します。RGBはクリックでframe画像を開けます。
   source frameは0始まりです。時間の秒表示は保存FPSによる目安で、PTSではありません。
3. **欠損区間へ移動**: `[start,end)`ボタンやquality timelineから拒否frameへ移動します。
   直前60frameの軌跡は欠損で切れ、孤立した有効点は点として残ります。
   球の高さ・身体配置・統合sceneの静的plotも同じ規則です。
4. **component gallery**: 採用artifactと入力portの参照から各componentの集計図・保存観測へ進みます。
   対応外schema・古い依存は明示し、別形式へ読み替えません。

2Dが存在しても3Dの有効性を復活させません。無効な座標0を原点の観測と扱いません。
maskだけで人物/球の不在・画面外・遮蔽を推定せず、未知の理由コードは`UNKNOWN_CODE`と表示します。
現行SceneResult v2のmask・理由・shapeの契約は[../README.md](../README.md)が正本です。

3Dはモデル推定です。再投影は保存された単一平面pinhole近似で、歪み補正・独立3D GTとの比較ではありません。
校正のRMSEも保存平面fitの指標です。SMPL meshの詳細レビューはこの統合画面の対象に含めません。

## 静的gallery・動画

`--serve`なしでもgalleryを保存できます。`index.html`、`review.json`、static assets、
代表frameのRGB縮小画像とplotを生成します。HTTP serverでこのdirectoryを開くと、
RGB収録frameを確認できます。未収録frameではRGB未収録を明示し、隣のframeの画像を代用しません。
`--serve`では同じ入力に固定したcamera/frameだけをAPIから読み、任意source frameのRGBを表示します。
入力動画のhash・寸法・FPS・frame数、sceneの時間軸、descriptor/array checksumを検証します。
起動後にsource/indexが変更された場合、APIは再生成を要求するエラーを返します。

`--videos`を加えると従来のcomponent overlay動画をCPUで生成します。
frame slider・動画seekは保存FPSに従って連動し、再生した1本の動画がframeの基準になります。
元source frameの厳密な照合にはRGB APIとmask tableを使ってください。
生成したgallery自体には元動画や推定配列の一部が含まれるため、公開対象は選択したスクリーンショットに限定します。
