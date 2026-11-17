# Stage 1 の再現

元入力はrun 75の凍結比較・run 73のRGB、datasetの同じ動画。既存runは読み取り専用。
planningのframe splitはこのbundleの `frame-split.json` に保存した。各protocol、入力inventory、測定点は
`outputs/tennis_scene/evaluate/court-video-shared-stage1-20261117/` に保持する。
各段階はsummary.jsonのheadとsource hashに合わせたcheckoutで実行する。
手指定点・ネット帯・参照Hを引数にしない。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
DRIVER=knowledge/runs/run-court-video-shared-stage1-20261117
RUN=outputs/tennis_scene/evaluate/court-video-shared-stage1-20261117
SPLIT=knowledge/runs/run-court-video-shared-stage1-20261117/frame-split.json
DATASET=data/tennis_multiview/processed/meiji_4cam/dataset
PREVIOUS=/home/kamimura/projects/tennis-lab/.worktree-outputs/court-acceptance-postprocess/tennis_scene/evaluate
```

R1 (`3e8ecf5b9`): `probe.py prepare --output "$RUN" --old-outputs "$PREVIOUS" --dataset "$DATASET" --split "$SPLIT"`、
続けて `probe.py run --output "$RUN" --workers 4` を `$PY "$DRIVER/…"` で実行する。
R2 (`dfff8d751`)は同じ2コマンドのoutputを `$RUN/endpoints` に変える。

R3 (`757f0fc22`): `$PY "$DRIVER/confirm.py" prepare --probe "$RUN/endpoints" --output "$RUN/all-fit" --split "$SPLIT" --dataset "$DATASET" --raw-manifest "$PREVIOUS/court-fixed-support-rgb-20261117/manifest.json" --workers 4`、
次に `$PY "$DRIVER/confirm.py" run --output "$RUN/all-fit" --workers 4`。

R4 (`a9fd35ff4`): `$PY "$DRIVER/reconcile.py" --source "$RUN/all-fit" --output "$RUN/service-order"`。
記録したraw点を再利用し、評価RGBは読み込まない。
最終的な解選択の確認は `$PY "$DRIVER/summarize.py" --source "$RUN" --output "$DRIVER/summary.json"`。
これは各段階の結果を改変せず、R4の有効解を同じ規則で選択した結果も記録する。

full diagnosticと原解はoutputs、small summaryと再現scriptはこのbundleに置く。
outputsを失った場合は各段階の入力hashを揃えて再実行し、推定値を参照から補ってはならない。
