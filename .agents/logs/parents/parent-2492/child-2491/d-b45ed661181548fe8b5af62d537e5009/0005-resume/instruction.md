## parent CLI が確認した待機結果

全jobが終端した。成果物を検証し、指定された記録を更新する。失敗・取消は原因と残件を確認する。再投入した場合は新しいjob IDを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "queue",
  "jobs": [
    {
      "job_id": "0-8204473201027913525_2026-11-24T13-26-38-972086474Z_2037423_i2491-fasthmr-pt32-pt16-smoke.job",
      "state": "done",
      "exit_code": null,
      "log": "[local path]",
      "repro": "[local path]",
      "output_dir": null
    }
  ]
}

## 親からの補足: 土台の更新（#2498 マージ済み、#2497 が速度基準を公開）

待機明けの自動再開は、土台PRのheadが進んだため親CLIに拒否された。親がこのbranchへ最新headを取り込んだ（このターンの開始時点のHEAD）。

- **#2498（#2495）**: main へマージ済み（`2dd4a5518`）。採点器は `0d4bf0990` で、同じIoUの候補を正準順序（検出スコア降順 → BBox → 関節）で照合するよう修正されている。修正前の採点器で出した姿勢スコア（OKS・PCK）があれば、保存済みのraw予測から再採点して記録を更新する（GPU不要）。値が変わったかどうかも書く。検出Recallは変わらない見込みだが、照合結果を記録する。
- **#2497（#2496）**: head `14ca88f8a` で、固定速度入力と現行モデルの速度基準を公開した。現在、正式レビュー中。
  - 固定入力: `meiji_4cam/person-model-speed-v1`。manifestのsha256は `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62`。正本は `knowledge/runs/run-person-model-speed-baseline/inputs/manifest.json`。実データはメインcheckoutの `.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs/` にあり、読み取り専用として扱う。
  - 基準値: `knowledge/nodes/tennis_scene/000085-run-person-model-speed-baseline.md`。
  - 候補の速度は `src/utils/inference_benchmark.py` で、この固定入力を使って測る。基準と同じbatch・入力順・反復数（warmup 30、3 block×100反復）にそろえる。parityの判定基準は、基準runと同じ値を使う。
  - レビューの結果、harnessのAPIやparityの規約が変わる可能性がある。変わったら親が知らせる。
- 取り込み後のHEADで影響範囲のテストを通してから進める。PRがあるならpushする。
- 待機（`PARENT_RESULT` の waiting）で終える前に、`git fetch origin codex/parent-2492-issue-2496-primary` と `git fetch origin main` を実行し、#2497 の最新headを取り込んでテストを通してから終える。最終行の `PARENT_RESULT` は、1行の正しいJSONにする。

## 親からの補足: #2497 のレビュー結果と、parityの扱い（ユーザー判断 2026-11-25）

#2497 の正式レビューで major が4件出て、#2496 の担当が修正中。

- parityの判定は `src/utils` の共通部品に移る。
- 1反復の計時は、adapterが返った後のデバイス同期までに変わる。
- 条件の状態は `ok`・`parity_failed`（速度は有効、印付き）・`infeasible`（計時できない）の3つに分かれる。
- 固定入力は、期待SHAを検証する共通の入口から読むようになる。

parityの扱いは、ユーザーが次のとおり決めた。

- parityの判定基準は、全候補で共通の1つに固定する。値は、検出がIoU P05 ≥ 0.98・score差P95 ≤ 0.01・欠落／過剰 ≤ 1%、身体がカメラ座標の関節P95 10 mm。姿勢の基準は、#2497 の共通部品の定数に従う。各担当が独自の閾値（例: IoU 0.90・score差0.025）を使ってはならない。
- 検出と姿勢は、parityが不成立でも、疑似GTで実測した精度と速度でトレードオフ図に載せ、「PT32と出力がずれる」の印を付ける。
- 身体復元（GVHMR・FastHMR）は、parityが不成立なら図から外す。
- OOM・変換失敗・非対応op・TensorRT context生成の失敗は、実行不能として理由とともに記録する。

このため、次のように進める。

- **新しい速度計測は、#2497 の修正版が出るまでqueueに投入しない。** すでに投入済み・計測済みの速度は、修正版のharnessで測り直す前提で、暫定値として扱う。
- 精度の計測と採点（疑似GT・PT32/PT16/TRT16）は、これまでどおり進めてよい。
- parityの数値（PT32との比較）は、生の予測から後で共通部品で再判定できるよう、予測を保存しておく。
