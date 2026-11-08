# 単一cameraの白紙注釈

[English](README.md) · [简体中文](README.zh-CN.md)

`BallVideoWorker` は登録済み `BlankAnnotationContext` を受ける
`ball_video_annotation / 3 / review`。予測receipt・overlay・`confirm_predictions` を使わない。
状態と由来の正本は [v3仕様](../README.md)。

## 道具と探索方針

[`instructions.md`](instructions.md) が生きたプレー球1球、由来の保持、unknown最小化とtool契約を定める。
`AnnotationPolicy(policy_id, version, instructions)` が探索方針を提供する。時系列一覧、任意の参照点、
crop/拡大、推測、自己修正で同じ道具を使い、閲覧順を固定しない。

`AnnotationRun` はcontext・plan・runtime・方針・tool/worker版を固定する。model・effort・code・資源は
plan、worker/service条件はruntimeが正本。次attemptで方針や条件を変更せず、同じrunを使う。

`camera-tools.v2` はhandoff・rows・imageに `read_id` を必須とする。clientが呼出しごとに自動生成し、
編集後に同じ範囲を読み直すと現在のrevision・進捗を返す。元の `read_id` とparametersを指定した再送は、
元のsnapshotを返す。編集の `expected_revision` によるCAS契約は維持する。

親はcamera、元frame/PTS、crop、所有者を検証してplain画像を返す。非zero開始でもframe引数は
**元動画のframe番号**。提供区間内の同じcameraのframeだけを要求・supportへ指定できる。
receiptにnative crop・倍率・元PTSを保持する。実行境界は
[private input boundary](../../../../base/annotation/workers/README.md#private-input-workers) を参照。

## 親からの接続と途中保存

保存済みLocalAnnotationPlanと `BallVideoWorker(policy)` をWorkerPoolへ渡し、Codex binary/home、
review timeout、scope/予約量、code revisionにruntimeの固定値を使う。productionは共有ResourceManagerが
必要。ServiceBrokerがinboxを受け、OperationsがCPU render stageでReviewOperationを実行する。
白紙planはcamera serviceだけを許し、比較は `benchmark-stage` を使う。
stdlibだけの `input/tools.py` を迂回しても親の検証は迂回できない。

`video_edit` は変更後のv3全体を固定入力と照合し、不変payload/command/revisionを保存してから
SQLiteのCASでheadを更新し、返答する。同一再送は同じrevisionへ収束する。最終応答前に中断しても、
commit済み進捗・過去の注釈・画像/toolの由来を保持する。

```python
from src.tasks.ball_detection.annotation.video.agent import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.usage import is_unprocessed

store = VideoRevisionStore(layout, task.run)  # task: saved AgentTask
revision, completion = store.current()
annotation = store.annotation(revision)
remaining = tuple(row.frame_index for row in annotation.frames if is_unprocessed(row))
```

`video_handoff` は未処理だけをpageで返し、最終unknownを含めない。次attemptのinitial_revisionは保存head。
時間切れの自動継続は[ローカルcontroller](../../../../annotation/local_controller.py)が担当し、
このモジュールはattemptの起動・再試行を行わない。

最終AgentDecisionは現在のrevisionを指定する。worker/rollout identity、構文、入力/frame/PTS、状態・
由来を機械検証する。推定frameの独立目視や人手QAを追加せず、未処理0ならunknownを残して完成できる。
WorkOutcome.adoptionはVideoCompletionを指し、そのannotationがscene/datasetに渡す正確なv3参照。
completion公開・seal・最終receiptの中断は元の完成記録へ収束する。dataset公開は別工程。

## 探索方針の比較

[白紙benchmark API](../../../../annotation/benchmark/blank.py) は既存namespaceと共通telemetryのusage集計を再利用する。

```python
from src.tasks.annotation.benchmark.blank import (
    compare_blank_conditions,
    register_blank_condition,
)

condition_a = register_blank_condition(layout, source_plan, "overview", policy_a)
condition_b = register_blank_condition(layout, source_plan, "adaptive", policy_b)
# Run each with its saved plan/runtime/policy and benchmark layout.
report = compare_blank_conditions(
    layout, (condition_a, condition_b), reference_annotations=references
)
```

画像とframe identityを共有し、worker/revision台帳を分離する。phase=benchmark・公開無効を固定して
通常dataset/進捗を変えない。参照注釈は親の採点専用でworkerには渡さない。

応答ID・attempt/thread、input/output/cache token、revisionと処理済み/unknown/座標利用可、画像要求・提示・
再閲覧、native pixel誤差を結ぶ。要求はcommit済み画像service、要求の再閲覧は同じ元frameのtile再要求、
応答の再閲覧は同じ元frameの再提示を数える。誤差は双方に座標がある組だけで計算し、組数とcamera/元frame/PTSを
保持する。usage・参照座標の欠測はnull、既知小計は別。共通telemetryがusageを重複排除する。
synthetic/real/mixed/未起動を区別し、acceptanceはunverified。合成動画・実保存/プロセス・fake worker・合成usageの
試験であり、実GPTの精度・token削減・実動画受入は #1715 に残る。
