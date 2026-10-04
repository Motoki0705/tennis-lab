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
- Ballのみ: `GET /api/players?scene=...&dataset=players/...&mode=reviewed&start=0&count=32`。
  `mode=raw`で生成結果。状態と利用可否、各frameのpose・枠・ID・軌跡を返す。
  `/api/scenes`は`player_dataset`と`player_status`による状態表示・絞り込みにも対応する。
- `POST /api/infer`: `{checkpoint,scene,start,count,threshold,device}`。review modeでは拒否。

frame layerは`points`とPNG data URLの`rasters`を持ちます。pointsはfiniteなpixel座標、rastersは`name,data`と任意の`legend`です。バックエンドはscene IDをserver-side catalogで解決し、任意ファイル読み込みAPIは提供しません。1要求最大64frames、推論はserverごとに同時1件です。

## 検証

共有HTTP/queueテスト、タスク別source/inferenceテストに加え、`tests/e2e/tasks/detection/ui_browser.cjs` は実画像とmock inferenceでpan/zoom、checkpoint filtering、seek後の再生、推論中scene変更、再実行、desktop/mobile boundsを検証します。ブラウザテストは`PLAYWRIGHT_MODULE`と`CHROMIUM_PATH`でローカルのPlaywright/Chromiumを指定できます。

`static/playback.mjs`は注釈を32frameずつ取得し、画像を12frame先読みします。
画像は最大24枚 / 96 MiB、通信ジョブは最大4件（注釈ジョブはGTとposeの2要求）です。
画像と注釈が揃ってから同じframeをCanvasに渡します。シークや選択変更で不要な要求を
中止し、キャッシュ外のImageBitmapを解放します。時刻基準で再生し、遅延時は待機して
再開時の連続スキップや高速な追いつき再生を避けます。

`player_playback_browser.cjs`はローカルのball-mix-v2 / player-pose-v1の代表3clipで、
全重畳ONの25fps（24.5〜25.5fps）・欠落0・ラベル取得のバッチ化を計測します。
2選手、4選手、raw多数人物、未採用状態、意図的な通信待ちと復帰も検査します。
サーバー起動後、`DETECTION_URL=http://127.0.0.1:8776`を指定して実行します。
画像とJSONの結果は`SCREENSHOT_DIR`（既定`/tmp/ball-player-playback`）へ保存します。
このテストは既存データを必要とし、GPU推論を起動しません。
