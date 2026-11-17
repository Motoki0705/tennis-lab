# 再現・保存先

実装は `edf33290d5e22383dd3d66e4e277ec2538c81f62`。このrun bundleも必要。
共通Hはrun 77のstoreを**loadだけ**する。存在しない場合に共通fitをやり直す手順ではない。
`immutable-inputs.json` のbundle・camera文書・数値source、`frame-protocol.json` の選定、
`frozen-profile.json` の設定が入力の正本。初期値・fit・参照frameは数値診断から除外済み。

以下は実行したコマンド。再実行時には既存の評価先を上書きせず、`RUN`を新しい空directoryへ変え、
3つのprotocolと復元済み人工fixtureをコピーする。`frozen-profile.json` は結果に合わせて書き直さない。
`evidence.py` は数値sourceのhashを照合し、mypyコメント1行だけの差は凍結sourceとのAST一致で検証する。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
DRIVER=knowledge/runs/run-court-video-drift-lines-20261117
RUN=outputs/tennis_scene/evaluate/court-video-drift-lines-20261117
```

開発集合は別seed 219178。開発結果を記録してからprofileを凍結した。
開発の再実行が凍結profileを更新することはない。

```bash
$PY "$DRIVER/artificial.py" development --output "$RUN" > "$RUN/development.log" 2>&1
$PY "$DRIVER/artificial.py" recover --output "$RUN" > "$RUN/recover.log" 2>&1
$PY "$DRIVER/artificial.py" heldout --output "$RUN" > "$RUN/heldout.log" 2>&1
```

`recover` はrun 77で保存されなかった人工共通Hだけを旧コードから復元し、旧40 RGB hashと精度数値へ照合する。
本番4cameraのHは扱わない。既に保存した `recovered-synthetic-fixtures.json` を使えば再構成は不要。
heldoutは8視点の静止/5px/11pxで、24件。参照Hや人工GTは診断器へ渡さず、共有fit行列だけを渡す。

```bash
$PY "$DRIVER/evaluate.py" --output "$RUN" --workers 4 > "$RUN/evaluate.log" 2>&1
$PY "$DRIVER/review.py" --output "$RUN" --workers 4 > "$RUN/review.log" 2>&1
```

実データは `data/tennis_multiview/processed/meiji_4cam/dataset`。
全69clip×4cameraの媒体hashとsource frameを照合し、1,380 frameを計測する。
新診断は `drift-store/`、元Hはrun 77の `store/` にあり、別々の内容hashを持つ。
結果には実行時刻を含むため、別実行のbundle hashが同じになるという契約ではない。
元Hは同じhashのままで、clip consumerには指定した診断camera digestも入る。

受入結果は `evaluation-summary.json`。`review/index.html` でcamera・clip・frameを選び、
表からframeへ移動し、原寸・200%/400%・線なし表示とraw/overlay cropを確認できる。
先頭frameは目視専用。5つの測定frameのうち最悪判定をclip判定に使う。
人工実験と同じく、時間平均や共通Hの置換はしていない。

初回全件のBGR checksumをRGBへ直すため、全件をもう一度測定した。
修正前summaryは `evaluation-summary-bgr.json`、旧bundleも同じdrift-storeに保持する。
`check_results.py` はこのrunの固定pathで新旧1,380件の数値・判定一致と、実court component全276件を検査する。
`causes.py` は旧bundleの失敗437件を再抽出し、記録された整合subsetだけの条件付き変位を求める。
これは事後の原因分析であり、採否・正式成果物・閾値を変更しない。

```bash
$PY "$DRIVER/check_results.py" > "$RUN/check-results.log" 2>&1
$PY "$DRIVER/causes.py" --output "$RUN" > "$RUN/causes.log" 2>&1
$PY "$DRIVER/figures.py"
$PY "$DRIVER/check_gallery.py" \
  --chrome /home/kamimura/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell \
  --output "$RUN" > "$RUN/browser.log" 2>&1
$PY "$DRIVER/capture.py"
```

browserは `--disable-gpu --use-gl=disabled` と専用の一時profileで起動し、検査後終了した。
276行・4camera・69clip・6frame選択・5行の判定表・1920×1080 canvas・拡大・warning・先頭frameの目視表示・
連続切替後の対応画像・最悪frame・表のクリックを検査した。人間の品質確認を代替しない。

`capture.py` は小さなJSON・camera別CSV・図をknowledgeへ保存し、大きな予測・元画像・全inventoryをoutputsに残す。
コピー対象と上限はknowledge READMEに従う。ファイル削除はしていない。
既定clipの旧数値sourceは不変なので、run 77の3,545画像/8,530領域の完全一致replayを再利用する。
`default-replay-reuse.json` にgeometry・estimator・inference・componentのbyte一致と既定YAMLの比較を保存した。

テストは `verification.log`（143）、`configuration-verification.log`（464）、
`reader-and-motion-final.log`（13）、`multi-clip-final.log`（2）、`runtime-reference-final.log`（1）を参照。
同じnodeを複数suiteで使っており、件数を合計して独立ケース数とはしない。

## 人間の判断による方針変更の再集計

上記は元の数値測定・fail方針の履歴であり、実装commit `edf33290d` に対応する。
現在の方針適用は `56656531c6dd1721d3967790dacbebad07ce0abd` と同じソースを使う。
H、RGB測定、診断bundleを再作成せず、保存済みのrun 78の結果を既定 `warn` と明示 `fail` で比較する。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=.venv/bin/python
DRIVER=knowledge/runs/run-court-video-drift-lines-20261117
RUN=outputs/tennis_scene/evaluate/court-video-drift-lines-20261117
$PY "$DRIVER/reassess_policy.py" --previous "$RUN" --output "$RUN/policy" > "$RUN/policy/reassessment.log" 2>&1
$PY "$DRIVER/check_policy_gallery.py" \
  --chrome /home/kamimura/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell \
  --output "$RUN/policy" > "$RUN/policy/browser.log" 2>&1
```

再実行は空の別outputへ行う。既存の `policy/summary.json` とCSV、判断の出典を保持する。
`reassess_policy.py` はHydraから明示load設定を読み、YAML既定の `warn` を確認する。
276件のcourt component出力がcamera内で同一であること、strict方針で出力できた値との一致、
元Hとdriftのbytes/hash不変、frameごとの数値・理由不変を検査する。元動画をdecodeしない。
人間の確認範囲と引用は親からの指示（2026-11-17、turn c4c7bed22d114a2da39b7a77252fca41）に基づく。

現在の確認ページは `policy/review/index.html`。既存raw/cropへの相対参照を使うため、元のreviewディレクトリも保持する。
元のH・参照H・cropは不変で、適用判定と理由・frame・方針を更新した。旧失敗ラベルの全画面JPEGは現在のページからリンクしない。
`policy/tests.log`（75ケース）、`policy/related-tests.log`（508ケース）、故障注入と復元2ケースの記録を参照する。
このrunの現在の受入と歴史的な厳格判定は、nodeの別の節で区別している。
