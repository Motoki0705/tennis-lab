# Meiji court crop probe v1（実行準備）

未実行。学習・診断結果や採用判定を表すノードではない。

`probe.py` は固定した3収録の校正clip × cam0/1を、productionと同じ9frame・Court checkpoint・fit条件で比較する。Aはmargin 0.25、Bは0.5、CはAと保存Hの14点bounding boxのunionを20px外へ拡張して画像内へclipする。Cの生成に手動点を使わない。video_000の手動点はfit後の評価専用。cam2は実行対象外。

実行は元repo共通training queueから、worktreeをCWDとして1回だけ投入する。`repro.sh` はqueue環境と新規の絶対outputを要求する。例（投入操作は親が担当）:

```bash
export TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue
bash .agents/skills/training-queue/scripts/training_queue.sh add \
  'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash knowledge/runs/run-slcs-meiji-court-crop-probe-v1/repro.sh --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/meiji_court_crop_probe/s42-001' \
  --name slcs-meiji-court-crop-probe-v1 --provider codex --session SESSION_ID --resource half
```

GPU処理はモデル1回loadのみ。既存dataset/cacheへ書き込まない。入力はcache identityと照合し、実行前後でソース・動画・annotation・cache・checkpointのdual SHAとstatを保存・比較する。runtime.jsonとqueueのrepro captureを合わせて実行状態を追跡する。queueのuncommitted.patch登録、node/group登録、結果画像のレビューは親が行う。

各viewのvariant JSONとraw NPZ、3variant × first/mid/lastのcontact_sheet.jpg、全viewのsummary.jsonを保存する。各cellは上段が元RGB、下段がHから生成した14点と線。raw NPZは検出点であり、Hから生成した点と区別する。fit失敗はrejectedで保存し、条件を緩和しない。Aのraw・score・Hは丸め許容なしでcacheと完全一致を確認し、不一致でも全variant資料を出した後に非ゼロ終了する。fit残差は除外点も含み、GT誤差ではない。fit_returnedは採用や画像評価の合格を意味しない。

CPU確認:

```bash
PYTHONPATH=. .venv/bin/python knowledge/runs/run-slcs-meiji-court-crop-probe-v1/probe.py --help
.venv/bin/ruff check knowledge/runs/run-slcs-meiji-court-crop-probe-v1
.venv/bin/mypy --follow-imports=silent knowledge/runs/run-slcs-meiji-court-crop-probe-v1/probe.py knowledge/runs/run-slcs-meiji-court-crop-probe-v1/test_probe.py
bash -n knowledge/runs/run-slcs-meiji-court-crop-probe-v1/repro.sh
PYTHONPATH=. .venv/bin/pytest -n 0 -q knowledge/runs/run-slcs-meiji-court-crop-probe-v1/test_probe.py
```
