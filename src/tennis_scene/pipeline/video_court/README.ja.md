# 動画共有コート成果物

[English](README.md) · [简体中文](README.zh-CN.md)

既定の `court_kp.mode=clip` は維持する。`video_shared` は明示したdataset・video・cameraごとに共通Hを作る。モデル候補を初期値に複数frameのRGB白線を測り、線の有限端とセンターサービス線で対応を確かめる。同等の解から線残差で1つを選び、Hの係数は平均しない。幾何と単位は[共有幾何](../../../tasks/court_detection/geometry/video_shared/README.ja.md)を参照。

現在のbuild入力はhashを固定した**凍結モデル候補**。新たなモデル推論は呼出側の責務で、入力不足時にモデルへ切り替えない。CPUだけで準備できる。

`court_kp.video_shared` で取得方法を明示する。

|項目|build|load|
|---|---|---|
|`action`|`build`|`load`|
|`directory`|ARTIFACT root相対のstore|同じstore|
|`dataset_id`, `video_id`|対象ID必須|対象ID必須|
|`request`, `request_sha256`|ARTIFACT相対のrequestとhash|`null`|
|`artifact_digest`|`null`|準備APIが返したdigest|
|`rebuild`|明示再計算・破損修復を許可|`false`|

通常の単一clipコマンドに、例えば次を追加する。

```bash
court_kp.mode=video_shared \
court_kp.video_shared.action=load \
court_kp.video_shared.directory=tennis_scene/court_shared/match \
court_kp.video_shared.dataset_id=match \
court_kp.video_shared.video_id=video_000 \
court_kp.video_shared.artifact_digest="$ARTIFACT_DIGEST" \
clip_id=video_000/clip_000
```

requestのJSONと候補のfield一覧は[英語版の入力契約](README.md)を正本とする。camera全件の候補manifest `{path, sha256}` を指定する。clip選択がnullなら同じvideoの全clip、空・重複・別video・不明IDはエラー。frameは `floor(numerator*(N-1)/denominator)` で選び、重複を拒否する。bootstrapがnullならfit対象の先頭・中央・末尾clipを使う。候補のRGB hashと動画hashを実入力に照合し、有限な不採用モデルHも初期値には使える。

共通APIは `prepare_video_court(config, resolver, code_identity=...)`。返された成果物を各 `ClipExecution` の `standard_definition(..., video_court=artifact)` に渡せる。orchestratorは単一clipと既存dataset入口で準備・束縛する。将来の一括CLIとの接続は別タスクで、このmoduleはCLI全体・モデル寿命を実装しない。

`run.json` はbundleとcameraのdigestを記録する。loadではCourt checkpoint、検出、build、暗黙修復は不要。camera内容hashがclipのcourt node identityへ入り、bundleのcamera参照がdataset marker identityへ入る。1cameraの変更はそのcourtと依存nodeを無効化し、同じ実装の人物・ボールrootは再利用できる。既存のPython全体fingerprintによる実装変更時の全cache無効化は維持する。

共通Hが成立しないcameraと、信頼できない／大きく動いたclipは明示的に失敗する。整合する3px超はwarning、信頼できる測定変位が平均10px超または最大25px超なら失敗。drift不明も失敗。閾値は1920×1080からscaleする。少数の診断frame間での動きは保証できない。区間分割・局所Hへの置換はしない。

共有観測はschema v3で、frame 0へ適用した座標と動画来歴を持つ。校正は直接のframe 0観測と呼び替えない。固定幾何import・direct annotation・旧ball3d readerは共有来歴を明示的に拒否する。

## 共通Hを変えずに診断を差し替える

`drift_build.build_video_drift` は、元の成果物・全clipの `ClipSource`・`frame_plan[camera][clip]` の明示frame一覧・来歴・`DriftConfig` を受け取るCPUの準備API。全camera/clipを要求し、動画hash・所属・時間軸を検査して、計測後にも照合する。Hを再fitせず、別storeへ[線offsetによる診断](../../../tasks/court_detection/geometry/video_drift/README.ja.md)を保存する。frameと除外集合は呼出側で評価前に固定する。

元の共通Hを `action=load` し、`court_kp.video_shared.drift_directory` と `drift_digest` の両方を追加すると新診断を使う。両方nullなら明示的に元のv1埋込み診断を使う。片方だけ・破損・別の共通Hへの診断はエラーで、旧診断へ戻さない。orchestratorの `prepare_video_court()` が返す診断付き成果物と、同じruntime設定を複数clip計画にも渡す。

clip cacheは元のcamera digestと診断camera digestを別々に持ち、dataset markerにも診断参照を含める。receiptとcourt観測へ両digestと全frame判定を残す。診断だけ変わると対象courtと下流を再計算し、共通Hの内容hashは変わらない。元のbuild形式とv1診断は過去の再現用に保持する。
