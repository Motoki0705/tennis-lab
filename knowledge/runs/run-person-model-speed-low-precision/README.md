# 現行人物モデルのPT16・TRT16速度再計測

DINO、ViTPose-Hのflipあり・なし、HMR2特徴＋GVHMRの22条件をRTX 5060 Tiで測った。各条件300反復、合計6,600反復。全条件のparity欄には「未判定（PT32計測はスコープ外、ユーザー判断 2026-11-25）」を記録した。 [結果と先行値との照合](../../nodes/tennis_scene/000086-run-person-model-speed-low-precision.md)

- [cases.csv](cases.csv) は速度、VRAM、実行可否、load・cold startの一覧。 [cases.json](cases.json) は環境、入力順のSHA-256、各条件の重み・plan・元レポート・ログのSHA-256を記録する。NVMLはロード直前から各ブロック終端までのデバイス全体のサンプル最大値。
- `measurements/<workload>-<backend>.csv` に6,600反復のモデルGPU時間、wall time、stageごとのwall timeを保存した。0始まりの `batch_index` は、共通の入力indexをbatchごとに区切った順序を指す。ViTPose B=1は固定順の先頭300 cropを計時した。 [固定入力manifest](../run-person-model-speed-baseline/inputs/manifest.json) · [計時protocol](../run-person-model-speed-baseline/timing-protocol.json)
- [timing-comparison.csv](timing-comparison.csv) は、同じworkload・backend・batchの22条件を照合した。初回のparity不成立行も生の反復からGPUとwallのP50/P95、items/sを求めた。分位点は線形補間を使い、items/sは総item数を総秒数で割った。`delta_*` は再計測値から初回値を引き、増減率は初回値を分母にした。 [初回の記録](../run-person-model-speed-baseline/README.md)
- [queue-receipts.json](queue-receipts.json) は各queue jobのID、計測commit、ログと `run.json` のパスとSHA-256を保存する。計測commit `13e508cb78c46663fbe58ad8be03bcf9c96816a6` で3件を実行した。開始時のworktree差分は空で、個々のcaseの終了値は全22件で0だった。

元の各反復・stage呼び出しとログは、メインcheckoutの `.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-low-precision/results/<workload>/<backend>-b<batch>/` にある。`cases.json` の `PERSON_TRT_ROOT` は #2363 の `.worktree-outputs/parent-2278-issue-2363-primary/tensorrt`。`PERSON_POSE_ENGINE_RUN` は初回の `outputs/person-model-speed-baseline-v1`。ViTPose-H flipなしB=1のplanとbuildの記録は初回の資産を参照する。 [共有queueへの投入と集計コマンド](../run-person-model-speed-baseline/README.md#再現) · [初回のbuild receipt](../run-person-model-speed-baseline/vitpose-b1-build.json)
