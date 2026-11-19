# 再現と確認

製品数値ソースはbuild-protocolのcommitと全source hashで固定する。sourceのhash一覧はoutputsに保持し、
knowledgeのartifact inventoryから内容を照合できる。前runの凍結raw・LINE・RGB・比較JSONは書き換えない。
このrunはCPUだけで実行し、ニューラルモデルを新たに呼び出さない。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
DRIVER=knowledge/runs/run-court-video-shared-acceptance-20261117
RUN=outputs/tennis_scene/evaluate/court-video-shared-acceptance-20261117
STAGE1=outputs/tennis_scene/evaluate/court-video-shared-stage1-20261117
SPLIT=knowledge/runs/run-court-video-shared-stage1-20261117/frame-split.json
PREVIOUS=/home/kamimura/projects/tennis-lab/.worktree-outputs/court-acceptance-postprocess/tennis_scene/evaluate
DATASET=data/tennis_multiview/processed/meiji_4cam/dataset
```

候補の元比較JSONはrun 75のinventoryに照合済みで、stage1の `candidates/cam*.json` へcrop Hのまま保存してある。
その作成は[stage1の再現手順](../run-court-video-shared-stage1-20261117/reproduce.md)を使う。
`build.py prepare` はRGB・動画hashの来歴と親承認subsetを付け、製品の厳格なrequestを作る。

```bash
$PY "$DRIVER/build.py" prepare --output "$RUN" --stage1 "$STAGE1" --split "$SPLIT" \
  --raw-manifest "$PREVIOUS/court-fixed-support-rgb-20261117/manifest.json" --workers 4
$PY "$DRIVER/build.py" build --output "$RUN" > "$RUN/build.log" 2>&1
$PY "$DRIVER/synthetic.py" --output "$RUN/synthetic" > "$RUN/synthetic.log" 2>&1
```

buildは実際の `TennisSceneOrchestrator.prepare_video_court()` を使う。製品が保存したcamera内容hash・
bundle digestと、入力request・RGB hashを保持する。完了後の同じbuildはcamera cacheを再利用できる。
単一clipで読む場合は `court_kp.mode=video_shared` と、`action=load`、同じstore・dataset/video scope・
`build-summary.json` のartifact digestを明示する。入力manifest・内容hash・必要なRGBを失った場合に
別cameraや旧clip推定へ切り替えてはならない。

評価はbuildが終了してから実行する。参照を読むのはこの段階だけで、推定値を書き換えない。

```bash
$PY "$DRIVER/evaluate.py" --output "$RUN" --dataset "$DATASET" --split "$SPLIT" \
  --reference knowledge/runs/run-court-white-line-reference-20261116/reference/references.json \
  --legacy "$PREVIOUS/court-legacy-default-20261117" > "$RUN/evaluation.log" 2>&1
```

`review/index.html` をローカルbrowserで開く。camera/clip/先頭・heldout frameを選び、
共通Hと参照Hを切り替え、100/200/400%で原画を確認できる。各cropは元解像度PNGで、
線付きと原画を別保存する。frame 0は目視用で、数値driftは承認済みのheldout 1frameで検査する。
`cases.csv` は全276件の参照誤差、drift、旧clip推定との差、実court componentの出力／明示失敗を持つ。
旧clip推定が不採用、またはframe 0が参照作成frameの場合は差をnullとし、もっとも近いraw Hで補わない。

既定replayはrun 75のdriverと旧比較コードを新しい出力先へコピーして実行する。
過去のprotocolは変更しない。標準driver同様、開始と終了で数値ソースと設定を照合する。

```bash
$PY "$DRIVER/replay.py" --output "$RUN/legacy-replay" --prior-output "$PREVIOUS" \
  --workers 8 > "$RUN/replay.log" 2>&1
$PY "$DRIVER/verify_frozen_inputs.py" --prior-output "$PREVIOUS" \
  --output "$RUN/input-integrity.json" > "$RUN/input-integrity.log" 2>&1
$PY "$DRIVER/capture_evidence.py" --output "$RUN" --dataset "$DATASET"
```

`capture_evidence.py`は全replayと入力照合の成功を確認してから、small summary・CSV・概観図だけをknowledgeへコピーする。
大きな予測JSON・native画像・full protocolはoutputsのまま、inventoryを残す。
browserの確認には `check_gallery.py --chrome <既存のchrome-headless-shell> --output "$RUN"` を使った。
専用の一時profileを使いGPUを無効にし、276件・4camera・69clip・元解像度・拡大・warning・連続切替を確認する。
この確認は画像の人間承認を代替しない。

未完了の出力先を上書きせず、別名へ保存してから新しいrun先を使う。
`pre-heldout-interrupted-build` は診断記録と初期値境界の補正前、`ci-import-interrupted` は
設定カタログの型解決修正前の中断記録。後者の153画像は完全一致だったが全件受入には数えない。
`cpu-thread-interrupted`はCPUプロセス化前のbuild中断記録で、数値条件は変更していない。
停止理由・終了状態・protocolを残し、最終runと混ぜない。
