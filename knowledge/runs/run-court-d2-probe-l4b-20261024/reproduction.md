# 凍結192枚の再実行

前回のrun・凍結selection・生成/QAのsrc・本番recipeは変更しない。
親は前回の台帳12枚を保存したまま、NHT subprocessへの実際の依頼camera数で
描画予算を数え、実依頼0枚だった前回から全192枚を1回だけ再実行することを許可した。
根拠と上限は [budget.json](budget.json) に固定する。

`render_absolute.py` は共有一時root全体に `Path(temporary).resolve()` を適用してから
前回の `render_probe.run` を呼ぶ。実際のNHT subprocessを呼ぶ直前に依頼camera数を記録し、
成功・失敗ともに再試行しない。CPU専用preflightは同じ一時root helperを使い、
既存 `_NHTRenderer` と `NHTRenderClient` のrequest構築・scene検証・JSON保存を通し、
`subprocess.run` を実行直前で遮断する。L4確保はこの全192camera検査に通った後に行う。

CPU検査の入力は、採用manifestの24ファイルをメインtreeから新しいworktree内の
`tmp/court-d2-probe-l4b/preflight/assets/` へbyteコピーしたもの。元入力は読み取りだけで、
新しい採用資産の作成・再学習・alignment変更は行わない。コピー後も既存verifierを通す。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH="$PWD" .venv/bin/python \
  knowledge/runs/run-court-d2-probe-l4b-20261024/preflight.py \
  --selection knowledge/runs/run-court-d2-support-cpu-20261024/render-selection.json \
  --source-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes \
  --adopted-manifest knowledge/runs/run-court-assets-colab-l4-20261022/assets.json \
  --scratch tmp/court-d2-probe-l4b/preflight \
  --output knowledge/runs/run-court-d2-probe-l4b-20261024/evidence/preflight.json
```

再検査時はscratchとoutputに新規pathを渡す。CPU検査は描画・画質の成功を示さない。
L4のsetup・停止は [前回のresume手順](../run-court-d2-support-cpu-20261024/resume.md)に従い、
renderだけをこのrunのadapterへ置き換える。実際のsession・push済みcommit・argv・
回収先とreceiptは実行後のrun metadataに記録する。
