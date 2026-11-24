# 固定入力の推論ベンチマーク

`src.utils.inference_benchmark.benchmark` は、固定した入力順・GPU・batchで推論方式を比較する。
シーン実行中のcache・prefetch・並列処理を含む計測には `InferenceRecorder` を使う。
ベンチマークは同じNVMLのデバイス識別を使い、同期済みの1 batchごとの値を集計する。
[シーン実行の計測](inference_recording.md)

## モデルの接続

workloadの1件を画像、人物crop、人物系列などから決める。入力manifestで各行のID・SHA-256と順序を固定し、同一モデル内の3 backendには同じ入力と前後処理、TTAを与える。候補の入力解像度や重み、内部batch、dtype、想定するFP32演算、TensorRT profileは `BenchmarkCase.settings` に記録する。

Meiji 4camの候補比較では `PinnedBenchmarkInputs` で公開済みmanifestと計時protocolの期待SHA-256を渡す。`schedule(index, batch_size)` は計時用ID順を検証して、完全batchのID列と `BenchmarkProtocol` を返す。`frames` は全画面画像と検出＋姿勢の比較、`crops` は人物cropでの比較、`body_sequences` は身体の系列比較に使う。各レコードの実データには `verified_asset(row)` を使う。入力パス、SHA-256、採用したbatchは[現行モデル速度基準](../../knowledge/runs/run-person-model-speed-baseline/README.md)にある。

各backendで `BenchmarkAdapter.load()` から初期化済みの推論関数を返す。例の `load_eager_model`、`preprocess`、`decode` はモデル側の関数を使う。

```python
from pathlib import Path

import numpy as np

from src.utils.inference_benchmark import (
    Backend, Batch, BenchmarkAdapter, BenchmarkCase, Stage, benchmark,
)
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs
from src.utils.io import write_json_atomic

device = "cuda:0"
checkpoint = Path("weights.pth")


def load_pt32():
    model = load_eager_model(checkpoint).float().eval().to(device)

    def infer(rows, stages):
        inputs = stages.stage(Stage.PREPROCESS, lambda: preprocess(rows))
        inputs = stages.stage(Stage.H2D, lambda: inputs.to(device))
        outputs = stages.measure_forward(Backend.PT32, lambda: model(inputs))
        decoded = stages.stage(Stage.POSTPROCESS, lambda: decode(outputs))
        return stages.stage(Stage.D2H, lambda: decoded.cpu())

    return infer


pinned = PinnedBenchmarkInputs(
    fixed_input_dir, published_timing_protocol,
    manifest_sha256=published_manifest_sha256, timing_sha256=published_timing_sha256,
)
size = 1
schedule = pinned.schedule("frames", size)
frames = {row["id"]: np.load(pinned.verified_asset(row), allow_pickle=False)
          for row in pinned.rows("frames")}
batches = [Batch(ids, tuple(frames[key] for key in ids)) for ids in schedule.batch_ids]
schedule.require_batches([batch.ids for batch in batches])
case = BenchmarkCase("detector", Backend.PT32, size, "image", pinned.manifest_sha256, {
    "timing_protocol_sha256": pinned.timing_sha256,
    "timed_input_ids_sha256": schedule.timed_ids_sha256,
})
report = benchmark(
    case, batches,
    {Backend.PT32: BenchmarkAdapter(Backend.PT32, load_pt32, {"model": checkpoint})},
    protocol=schedule.protocol, device=device, dedicated_stream=True,
    check_parity=collect_pt32,
)
write_json_atomic(fresh_output / "detector-pt32-b1.json", report.to_dict())
```

- PT16はモデル指定のFP16 castかautocastをforwardに適用する。TRT16は検証済みのFP16 planを使い、`BenchmarkAdapter.engines` にplanのpathを渡す。既存のTensorRT資産を使う際は `EngineBundle` と `TensorRTRunner` で重み・profile・環境を照合する。詳細NVTXを無効にし、エンジンの演算精度と補助処理のbackendも記録する。[TensorRT資産の契約](../tennis_scene/pipeline/README.md#tensorrt-fp16)
- 複合stageは複数の `stages.measure_forward(..., name="model")` を呼び、間に前処理・転送・復号を差し込める。例えば検出＋poseは検出された全人物を同じ呼び出しで処理し、`item_unit="image"` を使う。系列は `Batch.ids` を系列単位にし、`Batch.items` に有効person-frame数、`processed_items` にpaddingを含む処理数を渡せる。

## 出力のparity

`check_parity` は全固定batchを一度ずつ流した `(Batch, output)` を受ける。例の `collect_pt32` で対象モデル自身のPT32 batch 1出力を入力IDごとに保存する。PT16・TRT16とPT32の別batchでは、同じコード版、重み、モデル設定、固定入力で得た参照を使い、同じ入力IDの出力を共通の `compare_*` 関数へ渡す。参照を読み込めない場合は `ConditionUnavailable(Status.PARITY_UNAVAILABLE, ...)` を返却理由に使う。計測の再実行には新規出力を使う。

| 比較 | `src.utils.inference_parity` の入口 | 渡す出力 |
|---|---|---|
| 人物検出 | `compare_detections(reference, candidate)` | 画像ID別の `DetectionOutput`。元画像pxの `(N, 4)` xyxyと `(N,)` score。画像内でIoU 0.5以上の組を最大件数、次いで最大合計IoUで一対一対応させる。 |
| 2D姿勢 | `compare_coco17(reference, candidate)` | crop ID別の `Coco17Output`。COCO順の `(17, 3)` に元画像pxのx、yとconfidenceを置く。検出と組み合わせる場合は `(N, 17, 3)` を `DetectionOutput.poses` に渡して `compare_detections(..., with_pose=True)` を使う。 |
| 身体復元 | `compare_bodies(reference, candidate)` | 系列ID別の `BodyOutput`。カメラ座標の関節 `(T, J, 3)` とtranslation `(T, 3)` はm、局所回転 `(T, R, 3)` はaxis-angleのrad。同じ関節・回転順で渡す。 |

関節maskを指定する場合は、関節軸までのshapeを持つbool配列を両backendで一致させる。既定では全関節を比較する。比較する関節には有限値が必要で、ID・shape・maskの不一致は `metrics.invalid` と理由を返す。検出の欠落率はPT32の全box数、過剰率は候補backendの全box数が分母。分位点と最大値は、対応したbox・有効関節・フレームの値を固定入力全体で集約する。`PARITY_RULES` の統計量と閾値は `ParityResult.criteria` に出力される。個別モデルへの例外を採用する場合は、共通判定と併記して根拠を比較レポートに残す。

## 計測と集計

- `BenchmarkProtocol` の既定はbatch `(1, 2, 4, 8, 16)`、warm-up 30回、100回×3ブロック。比較を始める前にworkloadごとの共通batch・shape・系列長を決める。掃引は1から始め、各caseで同じサイズの完全batchを渡す。固定入力数に端数が出る場合は、計測用の完全batchを事前に固定し、除いたIDを記録する。入力順はcold startの先頭行、warm-up、定常計測でそれぞれ先頭から開始し、ブロックをまたいで巡回する。
- CUDAのPyTorch FP32 matmulとcuDNN精度を `ieee` に設定し、forwardを `no_grad` で実行する。1条件を新しいqueueプロセスで測ると、CUDA context、allocator、engine context、初回カーネル初期化を条件ごとに揃えられる。ONNX exportとengine buildには個別プロセスで `measure_conversion()` を使い、receiptを `conversion=` で渡せる。既存engineなら、変換の記録は空配列になる。loadと最初の実入力1回（cold start）は個別に `setup_wall_ms` に記録する。[共有training queue](../../.agents/skills/training-queue/SKILL.md)
- `preprocess`、`h2d`、`forward`、`postprocess`、`d2h` は直列に呼び、各操作の完了をCUDA同期してホスト時刻を取る。`stage_wall` とcold startはadapterの戻り後のデバイス同期までを含む。`model_gpu` はforwardごとのCUDA event区間の合計。現行基準と同じ専用streamは `dedicated_stream=True` で使用できる。forward用の補助streamがある場合はevent終端までにjoinする。転送を対応するstageに置き、TTAのforward内処理は条件間で揃える。
- P50/P95は線形補間の標本分位点。batch単位とitem単位を保存する。`items_per_second` は対象の総item数を同期済みの実測時間の総和で割る。各batchのID、stage名、ブロック番号を生サンプルに残す。
- torchのallocator peakはロード直前からparity完了までのallocated / reserved bytes。NVMLはCUDA初期化後のload直前、load後、cold start後、warm-up後、各ブロック後、parity後にデバイス使用量を取得する。最大値はサンプル時点でのデバイス全体の使用量で、ホストの他プロセスの消費も含む。load直前からの差分と、GPU・driver・CUDA・PyTorch・TensorRT・clock・電力を併記する。取得できないclockや電力項目には理由が入る。
- 結果は `ok`、`parity_failed`、`infeasible` の3状態。`parity_failed` は完了した計時の `summary` と生サンプルを保持し、parity指標と不成立理由を記録する。`infeasible` は `summary=null` と、失敗までに完了した反復の生サンプルを保存する。`failure.status`・`phase`・`message` にOOM、変換失敗、非対応op、context生成失敗などの理由が入る。モデル固有の理由は `ConditionUnavailable` で渡せる。指定backendと登録またはforwardが不一致なら実行前に拒否する。

[先行ベンチマークの手順](../../knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)・[バッチ掃引の実測](../../knowledge/runs/run-i2336-vision-batch-sweep/README.md)
