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
renderだけをこのrunのadapterへ置き換えた。実際のsession・push済みcommit・argvは
[run.json](run.json) / [job-requests.json](job-requests.json) に記録する。
adapterのcommitを許可されたbranchへpushした後、その正確な版をColabで使用した。
レンダリング用adapterは実行後に変更していない。

## 実行と保存の確認

全192枚の描画processは正常終了した。workerは周期保存に失敗したため、
失敗statusを保存した後、既存render-probe jobへ `colab.py save` を1回実行した。
これは保存の回復で、描画commandの再提出やNHTの再実行ではない。
保存後statusのcompleted・returncode0・drive_saved_atを確認してから回収jobへ進んだ。
初回失敗statusと保存後statusは別ファイルのまま残している。

collectorは既存JPEG・JSONだけをtarへまとめた。raw NPZ192件は元のDriveへ保持し、
ローカルへ全量回収していない。CUDA・compiler・NHT/Torch版の読み取りjobも保存後、
全jobのterminal/Drive保存とdiffを確認してstopした。停止後のdoctorは割当0件。
通信断で未開始のsetup 1件のcancel/saveと、同じVMへの再接続も記録に残す。

## Drive回収とCPU再集計

実際の回収先は `tmp/court-d2-probe-l4b/review-transfer/` と
`tmp/court-d2-probe-l4b/recovered/`。再取得する場合は、既存先を上書きせず新しいpathを指定する。
Drive接続はskillで選んだ設定を使用し、設定内容を表示しない。

```bash
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download \
  outputs/synthetic_data_generation/profile/court-d2-support-l4/court-d2-probe-l4b-20261024t044312z/recovery \
  tmp/court-d2-probe-l4b/review-transfer-new --verify
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  knowledge/runs/run-court-d2-probe-l4b-20261024/extract_review.py \
  --archive tmp/court-d2-probe-l4b/review-transfer-new/review.tar \
  --output tmp/court-d2-probe-l4b/recovered-new
bash knowledge/runs/run-court-d2-probe-l4b-20261024/repro.sh \
  tmp/court-d2-probe-l4b/recovered-new
```

実回収ではcollector stdoutのtar digest/サイズとDriveから取得したbytesも照合した。
extract_review.pyは全411ファイルの厳密inventory・hash・サイズを検査する。
analyze_saved.pyは凍結selection・実行commit・NHT依頼192cameraの順序・QA bindingと理由を検査する。
repro.shはGPUを無効にし、QA集計・費用・時間JSONを再計算して保存済み結果とbyte照合する。
Colab開始・描画・学習を行わない。

CPU集計scriptはrun bundle内の版を使う。描画時の初期source pinは
[scripts-and-inputs.json](scripts-and-inputs.json)、保存回復の時間と台帳照合を含む
最終postprocessing sourceは [analysis-sources.json](analysis-sources.json) に分けた。
旧runのscriptはそのまま再利用し、旧analyzerのcommit固定部分だけを新adapterで明示的に照合し直す。

## 閲覧成果物

全192 JPEGとoverlay、16contact sheet、8原寸例、各scene48視点/2fpsのスライドショー4本を
CPUで用意した。動画は独立視点のスライドショーで、連続軌道の滑らかさは測らない。
各動画の全48frameをdecodeし、サイズとfpsを確認した。

ローカル入口は `tmp/court-d2-probe-l4b/review/index.html`。
Driveの `review-cpu.tar` は `review/index.html` と `native/` を含む。
正確なDrive path・digest・容量は [cpu-review-archive.json](evidence/cpu-review-archive.json)、
検証付きuploadは [cpu-review-upload.json](evidence/cpu-review-upload.json) を参照する。
raw配列・モデル・正式dataset_versionはこの閲覧archiveに含めない。

生成・学習の次の起動はこの再現手順に含まない。
条件と画質の受入判断は [comparison.md](comparison.md) の推奨を親が検討してから固定する。
