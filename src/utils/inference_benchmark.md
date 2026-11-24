# 固定入力の推論ベンチマーク

`src.utils.inference_benchmark.benchmark` は、固定した入力順・GPU・batchで推論方式を比較する。
シーン実行中のcache・prefetch・並列処理を含む計測には `InferenceRecorder` を使う。
ベンチマークは同じNVMLのデバイス識別を使い、同期済みの1 batchごとの値を集計する。
[シーン実行の計測](inference_recording.md)

## モデルの接続

workloadの1件を画像、人物crop、人物系列などから決める。入力manifestで各行のID・SHA-256と順序を固定し、同一モデル内の3 backendには同じ入力と前後処理、TTAを与える。候補の入力解像度や重み、内部batch、dtype、想定するFP32演算、TensorRT profileは `BenchmarkCase.settings` に記録する。

各backendで `BenchmarkAdapter.load()` から初期化済みの推論関数を返す。例の `load_eager_model`、`preprocess`、`decode` はモデル側の関数を使う。

```python
from pathlib import Path

from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend, Batch, BenchmarkAdapter, BenchmarkCase, Stage, benchmark,
)
from src.utils.io import write_json_atomic

device = "cuda:0"
checkpoint = Path("weights.pth")


def load_pt32():
    model = load_eager_model(checkpoint).float().eval().to(device)

    def infer(rows, stages):
        inputs = stages.stage(Stage.PREPROCESS, lambda: preprocess(rows))
        inputs = stages.stage(Stage.H2D, lambda: inputs.to(device))
        outputs = stages.forward(Backend.PT32, lambda: model(inputs))
        decoded = stages.stage(Stage.POSTPROCESS, lambda: decode(outputs))
        return stages.stage(Stage.D2H, lambda: decoded.cpu())

    return infer


case = BenchmarkCase(
    "detector", Backend.PT32, 1, "image", dual_sha256("input-manifest.json")
)
report = benchmark(
    case,
    [Batch((row.id,), (row,)) for row in pinned_rows],
    {Backend.PT32: BenchmarkAdapter(Backend.PT32, load_pt32, {"model": checkpoint})},
    device=device,
)
write_json_atomic(Path("detector-pt32-b1.json"), report.to_dict())
```

- PT16はモデル指定のFP16 castかautocastをforwardに適用する。TRT16は検証済みのFP16 planを使い、`BenchmarkAdapter.engines` にplanのpathを渡す。既存のTensorRT資産を使う際は `EngineBundle` と `TensorRTRunner` で重み・profile・環境を照合する。詳細NVTXを無効にし、エンジンの演算精度と補助処理のbackendも記録する。[TensorRT資産の契約](../tennis_scene/pipeline/README.md#tensorrt-fp16)
- 複合stageは複数の `stages.forward(..., name="model")` を呼び、間に前処理・転送・復号を差し込める。例えば検出＋poseは検出された全人物を同じ呼び出しで処理し、`item_unit="image"` を使う。系列は `Batch.ids` を系列単位にし、`Batch.items` に有効person-frame数、`processed_items` にpaddingを含む処理数を渡せる。
- `check_parity` は全固定batchを一度ずつ流した `(Batch, output)` のiteratorを受ける。PT32で参照出力をID付きで保存し、PT16/TRT16では同じIDの参照に対する数値差と比較基準を `ParityResult(passed, metrics, reason)` で返す。PT32のbatch間差も同じ口で評価できる。PT16/TRT16の成功にはPT32参照に対する全batchの確認が必要。参照欠測は `ConditionUnavailable(Status.PARITY_UNAVAILABLE, ...)` で理由を記録する。

## 計測と集計

- `BenchmarkProtocol` の既定はbatch `(1, 2, 4, 8, 16)`、warm-up 30回、100回×3ブロック。比較を始める前にworkloadごとの共通batch・shape・系列長を決める。掃引は1から始め、各caseで同じサイズの完全batchを渡す。固定入力数に端数が出る場合は、計測用の完全batchを事前に固定し、除いたIDを記録する。入力順はcold startの先頭行、warm-up、定常計測でそれぞれ先頭から開始し、ブロックをまたいで巡回する。
- CUDAのPyTorch FP32 matmulとcuDNN精度を `ieee` に設定し、forwardを `no_grad` で実行する。1条件を新しいqueueプロセスで測ると、CUDA context、allocator、engine context、初回カーネル初期化を条件ごとに揃えられる。ONNX exportとengine buildには個別プロセスで `measure_conversion()` を使い、receiptを `conversion=` で渡せる。既存engineなら、変換の記録は空配列になる。loadと最初の実入力1回（cold start）は個別に `setup_wall_ms` に記録する。[共有training queue](../../.agents/skills/training-queue/SKILL.md)
- `preprocess`、`h2d`、`forward`、`postprocess`、`d2h` は直列に呼び、各操作の完了をCUDA同期してホスト時刻を取る。`stage_wall` は1 batch呼び出し全体の時間。`model_gpu` はforwardごとのCUDA event区間の合計。forwardは現在のCUDA streamを使い、補助streamがある場合はevent終端までにjoinする。転送を対応するstageに置き、TTAのforward内処理は条件間で揃える。
- P50/P95は線形補間の標本分位点。batch単位とitem単位を保存する。`items_per_second` は対象の総item数を同期済みの実測時間の総和で割る。各batchのID、stage名、ブロック番号を生サンプルに残す。
- torchのallocator peakはロード直前からparity完了までのallocated / reserved bytes。NVMLはCUDA初期化後のload直前、load後、cold start後、warm-up後、各ブロック後、parity後にデバイス使用量を取得する。最大値はサンプル時点でのデバイス全体の使用量で、ホストの他プロセスの消費も含む。load直前からの差分と、GPU・driver・CUDA・PyTorch・TensorRT・clock・電力を併記する。取得できないclockや電力項目には理由が入る。
- OOM、変換失敗、非対応、parity不成立は `status` と `failure` で返る。`ConditionUnavailable` でモデル固有の非対応op、未取得重み、profile範囲外などの根拠を渡す。指定backendと登録またはforwardが不一致なら実行前に拒否する。失敗時は `summary=null` とし、取得済みの生サンプルは診断用に保存する。

[先行ベンチマークの手順](../../knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md)・[バッチ掃引の実測](../../knowledge/runs/run-i2336-vision-batch-sweep/README.md)
