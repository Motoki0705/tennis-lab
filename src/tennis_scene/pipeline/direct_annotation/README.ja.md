# sceneと白紙ボール注釈の一連実行

[English](README.md) · [简体中文](README.zh-CN.md)

`run_pipeline` の既定は `ball_detection.method=model`。`gpt_annotation` を選ぶと、
元動画の登録、予測を使わない注釈、時間切れからの自動継続、完成v3の正式再読込、
固定幾何による再構成、選択した場合だけactive-ball dataset公開までを接続する。
この経路は追加学習を投入しない。人物・身体の機能はそれぞれの設定に従う。

## requestの固定

```bash
.venv/bin/python -m src.tasks.annotation schema --type scene_annotation
```

`SceneAnnotationRequest` に次の入力を与える。

- `configuration`: [ローカル動画の設定](../../../tasks/annotation/README.md#ローカル動画の白紙入力登録)。
  明示したplan、予算scopeの作成/再利用、workerとサービスのcost、全cameraの元動画pathと
  `VideoAnnotationInput`、`publish_dataset`を含む。
- `policy`: ID・版・探索指示を持つ `AnnotationPolicy`。
- `publication`: 公開時の `ProductPlan`、公開なしでは `null`。generation policyは
  `target_semantics=active_ball`、元のQ90 JPEG、収録単位の明示splitまたは分割recipeを使う。
  互換active-ball snapshotをbaseとして追加できる。

同じ収録の全camera・関連clipでrecording IDを共有する。scene用にはcameraごとに元動画1本を指定し、
元frame 0からsceneの全frame数を覆う。元bytes・解像度・実復号PTS・time base・FPSを
登録契約どおり固定する。`max_frames`を指定する場合も注釈scopeと一致させる。
予算・costは運用側の実測入力から決め、fixtureの予算やtest clockを本番値へ流用しない。

```python
from src.tasks.annotation.scene_workflow import SceneAnnotationRequest
from src.tasks.base.annotation.artifacts import write_document
from src.tasks.base.annotation.paths import AnnotationLayout

layout = AnnotationLayout.production()
# 下記の各objectを上の型・入力条件に従って準備する。
# 公開なしではoutput_product = Noneとする。
request = SceneAnnotationRequest(configuration=local_configuration,
    policy=exploration_policy, publication=output_product)
reference = write_document(layout,
    layout.path("campaigns", "scene-inputs", "request.json"), request)
```

requestはcampaign内で不変。登録時に元動画identity・時間軸・scope・公開選択の変更を拒否する。
探索方針はworker受付前に固定し、attempt間で変更しない。
controller初期化前に登録receiptへ方針参照を固定する。復旧ではその参照を読み、scene requestや
benchmark conditionと照合し、欠損・破損・不一致は明示エラーとする。controllerのmirrorが
未作成でも既定値を選ばない。方針未指定の初回登録だけが既定方針を選び、その値を保存する。

## 一連実行

派生pathは動画がDATA、requestとbundleがARTIFACT root相対。digest引数には保存参照を使う。
`clip_id`は固定bundleのmanifestのclip IDと一致させる。

```bash
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[samples/cam0.mp4,samples/cam1.mp4,samples/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' 'clip_id=meiji_3cam/video_000/clip_000' \
  ball_detection.method=gpt_annotation ball_detection.checkpoint=null \
  direct_annotation.request=chat_annotation/campaigns/scene-inputs/request.json \
  direct_annotation.request_sha256="$REQUEST_SHA256" \
  direct_annotation.bundle=tennis_scene/evaluate/ball3d_comparison/clip000_20261103/fixed_input_bundle_v1/bundle.json \
  direct_annotation.bundle_sha256="$BUNDLE_SHA256" \
  direct_annotation.court_detection=verified_calibration_frame0 \
  direct_annotation.publish_dataset=true
```

指定clip000 bundleはcourt配列を省略しているため、`verified_calibration_frame0`を明示する。
通常のbundleは`stored_artifact`が既定。court検出・校正・向き・配置はload-onlyの外部入力とし、
[import契約](../imports/README.ja.md)で対応・出自を検証する。これらとballにcheckpointは不要。
GPU機能はrepoのtraining queue規則に従う。CPUのball単独実行では
`device=cpu person_observations.enabled=false player_reconstruction.enabled=false gvhmr.enabled=false`を指定する。

## receiptと復旧

component storeの`run.json`は`annotation`・`scene`・`dataset`・`learning`を分けて報告する。
scene metadataの`ok`/`partial`は従来の意味を保持する。cameraごとの完成と計測参照はcampaignの
`scene-workflow/annotations.json`、scene artifactとarchiveは`scene-completion.json`へ保存する。
任意公開は`scene-workflow/publication-request.json`と`publication.json`へ記録する。
`dataset=not_requested`は成功、`learning=not_requested`では学習を起動しない。

通常timeoutは自動継続する。増分なし・quota・保存破損・外部サービス障害は
[local controllerの復旧](../../../tasks/annotation/README.md#白紙注釈の自動継続)を使う。
統合注釈CLIの`run`・`tick`・`status`・`recover`・`resume-budget`・`report`は保存済み
`LocalAnnotationRuntime`とdigestを受け付ける。

公開だけが失敗した場合、同じrequest・幾何・storeを指定した元のsceneコマンドへ
`direct_annotation.resume=publication`を追加する。完成注釈とscene exportを検証してpublisherを再開し、
scene stageの構築・worker受付を行わない。publisher commit後、workflow receipt保存前の中断も
同じsnapshotへ収束する。新しい公開要求で保存注釈を上書きしない。

## 探索方針の比較と計測

`BlankConditionRegistration`（source plan・独立したcondition ID・版付きpolicy）を
`blank-register --operation PATH --sha256 VALUE`へ渡す。schemaは`blank_condition`と
`blank_comparison`で取得できる。返る`BlankCondition`のruntimeを通常の
`run --runtime PATH --sha256 VALUE --max-seconds N`へ渡すと、benchmark namespace内の担当cameraだけを処理する。
画像と予算を共有し、進捗・revisionは独立させる。planの公開選択はfalseに固定する。

`blank-compare --operation PATH --sha256 VALUE`は`BlankComparisonRequest`を受け、condition参照と
任意の親専用参照注釈から不変reportを保存する。本番完成時にも`BlankRunMeasurements`を保存し、
policy/run・attempt・画像要求/再閲覧・座標coverage・応答usage・指定時の参照誤差を結ぶ。
token欠測はnull。synthetic/real/mixed/未起動を区別し、`acceptance=unverified`を維持する。
比較runはscene公開compositionへ入れず、通常datasetへ自動公開しない。

## 集計と後続の受入

`annotation-summary.json`に全体・camera別の件数、状態別内訳、unknown/未処理の半開区間を保存する。
`summarize_video_annotations`は最終状態を使う。座標利用不可は処理済みで利用可能座標がない行であり、
座標なし遮蔽・画面外・no_active_ballを含む。未処理は別集計。
`summarize_reference`は最終`reference_kind`を使い、補完前の`source_status=unknown`を
保持した採用済み推定座標を最終unknownへ数えない。

`clip000_acceptance(report)`はcam0/cam1/cam2のframe 0〜1009全3030件にだけ、全camera合計の
「座標利用不可81以下・最終unknown25以下・未処理0」を適用する。小規模fixtureは`not_applicable`。
workflowは件数を保存し、認識品質gateを自動追加しない。

代表CPU E2Eは3camera×4frameの合成動画と固定幾何を使用する。fakeはモデル応答と5400秒の
時間境界だけで、登録・サービス・revision保存・再読込・幾何・scene export・publisher・dataset readerは実行する。
推定点の寄与と2view未満の欠測を確認する。integrationは公開なし/あり、workflow公開receipt直前の中断を確認する。
この結果は実GPTの認識性能を示さない。

後続#1714には旧注釈・元動画/frame/PTS identity・収録group/split・移行の出自が必要で、v3状態を捏造しない。
後続#1715には3本の元動画・固定plan/policy/予算・clip000固定bundleと参照、親専用の
`human_annotations/ground_truth_v1/ground_truth.json`が必要。探索方針比較後に本番policyを選び、
3030件の判断、集計、生成した3D座標/mask/出自、snapshotと再読込、overlayを残す。
実注釈・移行・認識受入・本学習は今回未実施。
