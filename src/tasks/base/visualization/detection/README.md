# Detection Review / Inference

Court Detection と Ball Detection の画像座標系Web UI。タスク別のsource readerとmodel-I/Oを維持し、画像ビューとHTTP境界を共有します。既存のGIF可視化は変更しません。

## 起動

repoのPython環境から、次のmoduleを実行します。既定のproject rootはgit common rootなので、worktreeからも元repoのデータとcheckpointを参照します。

| Task | Dataset Review | Inference | Ports |
|---|---|---|---|
| Court | `src.tasks.court_detection.scripts.review_dataset` | `src.tasks.court_detection.scripts.inference_ui` | 8774 / 8775 |
| Ball | `src.tasks.ball_detection.scripts.review_dataset` | `src.tasks.ball_detection.scripts.inference_ui` | 8776 / 8777 |

コピー可能なコマンドは[Court利用ガイド](../../../court_detection/visualization/README.md)・[Ball利用ガイド](../../../ball_detection/visualization/README.md)を参照してください。

共通引数: `--project-root`, `--data-root`, `--outputs-root`, `--checkpoints-root`, `--port`。
データは既定で `<project>/data`、checkpoint候補は `<project>/outputs/<task>` と `<project>/ckpt/<task>` の再帰スキャンです。checkpointは信頼できるローカルファイルだけを配置してください。Lightning checkpointはpickleを含み、読み込み自体がコードを実行し得ます。

サーバは127.0.0.1だけにbindし、外部Hostとcross-origin推論リクエストを拒否します。認証された公開サービスではありません。GPU実行のライフサイクルは[共有可視化README](../README.md)を参照してください。CPUは明示選択時だけ実行し、CUDA失敗からの自動切り替えは行いません。

## 表示

左にデータセット・検索とページ付きシーン一覧、推論modeではcheckpoint検索と候補を表示します。未選択時は全データセット、選択後は保存契約と互換なデータだけを提示します。未配置sourceや非互換checkpointには理由があります。

中央は実画像のpan/zoom、GT（緑の輪郭）と予測（赤の点）、選択したdense layerの重ね表示です。右でGT/予測/ラベル、レイヤー、不透明度、実行device・開始frame・frame数・しきい値を操作できます。表示PNGを保存できます。Ballはフレーム再生、Courtは単画像です。推論対象外フレームを予測なしとして明示し、古い応答で現在のシーンを上書きしません。

画像座標はoriginal image pixelの`x,y`です。2Dラベルから未観測の3Dコートやcamera poseを作りません。データ固有のschema、表示可能な教師、checkpoint互換性は各タスクのREADMEを正本とします。

## API

- `GET /api/catalog`: source、checkpoint、mode、CUDA availability。
- `GET /api/scenes?dataset=...&search=...&offset=0&limit=100&checkpoint=...`: source内scene一覧。
- `GET /api/preview?scene=...&start=0&count=1`: original sizeとframeごとのGT。
- `GET /api/image?scene=...&frame=0`: 原画像JPEG。
- `POST /api/infer`: `{checkpoint,scene,start,count,threshold,device}`。review modeでは拒否。

frame layerは`points`とPNG data URLの`rasters`を持ちます。pointsはfiniteなpixel座標、rastersは`name,data`と任意の`legend`です。バックエンドはscene IDをserver-side catalogで解決し、任意ファイル読み込みAPIは提供しません。1要求最大64frames、推論はserverごとに同時1件です。

## 検証

共有HTTP/queueテスト、タスク別source/inferenceテストに加え、`tests/e2e/tasks/detection/ui_browser.cjs` は実画像とmock inferenceでpan/zoom、checkpoint filtering、seek後の再生、推論中scene変更、再実行、desktop/mobile boundsを検証します。ブラウザテストは`PLAYWRIGHT_MODULE`と`CHROMIUM_PATH`でローカルのPlaywright/Chromiumを指定できます。
