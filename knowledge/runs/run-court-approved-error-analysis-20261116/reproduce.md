# 承認記録・誤差分析・改善計画の再現

Issue #2190 / PR #2200。production sourceの基準は`7b59c4730dfa934fb8e2323ae0553b96f737f412`。
新しいモデル推論・学習・重み変更・productionコードの変更はない。
GPUは禁止中で、training queueへの追加も禁止。以下はCPUの既存測定と再現方法の記録であり、GPU起動指示ではない。

親note:

- 承認: https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6917118209
- GPU停止・CPU再開: https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6918621524

## 正本と入力

- `knowledge/runs/run-court-white-line-reference-20261116/reference/references.json`は承認済み。
  `preapproval-reference.json`は7b59c4730時点のbytes、`freeze.json`はその時点の固定記録。
  承認によるH変更はなく、前後のgeometry hashは`approval.json`にある。
- `outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116/`の凍結raw、旧/新geometry、
  `meiji4-evaluation/meiji_4cam/`の276評価ファイルを使った。後の`scene-evaluation`へ切り替えない。
- rawはscene448画像・5433領域、Court val2211、synthetic明示test886。取得済みのNPZとmanifestを使う。
- 元RGBを使ったpilot/確認図は、中断前に既存動画のframe0をCPU decodeし、凍結rawのRGB hashと照合した。
  その結果は中断前に完了済み。再開後は保存結果の集計だけを行った。
- 大きなraw/候補配列、全trace、RGB/元解像度図はoutputsに保持。`source-identity.json`に入力manifest、
  source、保持ファイル集合のhashとサイズがある。入力が無ければ停止し、新しいrawを取り直さない。

## scriptの配置

scriptは当時のoutputsを基準に入出力を決める。Git内のbundleをそのまま実行場所にしない。
再現時は元の結果を上書きせず、同じ`evaluate/`配下の別名ディレクトリへscriptと必要な表をコピーする。
`common.py`は兄弟の以前のrunを読む。作業ディレクトリはこのworktreeのrepo root、Pythonは`.venv/bin/python`。
sourceは`source-identity.json`と一致させる。後日のproduction実装を混ぜて再現したとは扱わない。

## 実行したCPU分析

`RUN`は実際の実行場所。以下は実施履歴のコマンドで、GPUを使用しない。
再開ターンで実行し直したのは保存結果の`extra_summaries.py`だけで、完了したfitやraw取得を重複させていない。

```bash
RUN=outputs/tennis_scene/evaluate/court-error-analysis-20261116
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$RUN/record_approval.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python "$RUN/classify_candidates.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python "$RUN/regression_factorial.py" --workers 6
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$RUN/analyze_raw.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python "$RUN/occlusion_review.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python "$RUN/replay_sensitivity.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$RUN/line_repeat_probe.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python "$RUN/regression_paired.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python "$RUN/summarize_analysis.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$RUN/observation_ablation.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$RUN/rgb_pilot.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python "$RUN/render_analysis.py"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python "$RUN/extra_summaries.py"
```

初回の中断前コマンドは`CUDA_VISIBLE_DEVICES`を指定していなかったが、CPU演算のみのscriptだった。
ここでは禁止を明示するため空値を併記する。`line_repeat_probe.py`の初回は同じ処理をPython heredocで実行し、
内容をscriptとして保存した。`record_approval.py`は承認済みreferenceと集計metadataを更新するもので、
数値評価を再実行しない。derived reportの承認statusも更新した。作成時のrepro script/commandsは履歴として保持する。

`occlusion-review/annotations.json`は、20画像の原画contact sheetをagentが目視したラベル。
confidenceや予測の成否から生成していないため、script実行で自動再生成できる教師ではない。
全280点のうち、明確な人物遮蔽8点、塗線を確認124点、不確定143点、画像外5点。
不確定や未確認clipを可視/遮蔽へ補完しない。

## 比較の意味

- `candidate-analysis`は既存の選択・採否を一切変更せず、承認済みGTで形状分類と救済不能理由を集計した。
  oracle平行移動は分類の説明だけ。推論へ適用せず、新しいgateにもしていない。
- 最初のfactorialは221画像・1176 solve。全val喪失114、synthetic喪失45/獲得30と、両datasetの成立維持control16ずつ。
  O=境界/重複の除外、S=広がりを重視する点選択、V=seedで見えた線の固定。
- `regression_paired.py`は同じLINE evidenceを条件間で共有する追加対照。配列のhashを実行前後に検査する。
  観測除外が空ならO0/O1は等価なので、重複solveはせず明記している。
  距離場の配列自体は保存しておらず、別実行でのbitwise一致は保証しない。今後の比較では配列も保持する。
- 最初のfactorialを削除/上書きしていない。pairedでも元の全statusは再現しない。元runに距離場とseed traceが無いため、
  historical transition未再現12件は未確定のまま。新しく得た値で元の45喪失/30獲得を置き換えない。
- `observation_ablation.py`はOが関わる42画像で、境界/重複を分けた4条件。各画像内は同じLINE evidence。
- `rgb_pilot.py`は参照作成clipと重ならない32画像で既存RGB fitterをモデルHから開始した診断。
  単frameなので支持frame数を明示的に1とし、それ以外は既存CFGを使う。結果を見て設定を調整していない。
  GT・手指定maskをfitに使わず、失敗時の別Hへの切替もない。fit成功を採用としていない。
  同camera・同系統のfitterに由来する評価の依存性は残り、独立受入結果ではない。
- `report/video-selection.json`は既存の採用出力から最低costを選ぶだけ。4cameraの成否を276独立試行として数えない。

## 検証と今後

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest -o addopts='' --confcutdir="$RUN" "$RUN/test_analysis.py" --durations=0
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$RUN/verify_evidence.py"
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py
.venv/bin/pre-commit run
```

数値分類3ケースは3 passed / 3.63秒。prototypeの精度や回帰についての全実験はunit testとは別の記録。
初回の整合性checkは0件のFNがCounterに格納されていないためKeyErrorになった。省略された0件を扱うよう
checkを直し、最終照合は通過した。数値・分類・閾値は変更していない（`verification-correction.json`）。
元の実装・設定・testは変更していない。通常CIはCPU版torchを要求し、GPU用workflow_dispatchは呼び出さない。
push後の同じHEADの結果はPR #2200へ記録する。

GPU再開後のjobはnode本文の**計画のみ**。既存checkpoint・入力・条件、既存取得時間460.5秒を根拠とする見積もりと、
未計測のmodel load/新CPU処理を区別した。semantic LINE/SEGの配列は未保存で、効果は未確認。
独立な新注釈は評価専用として親の判断へ返し、今回人へ依頼していない。merge・issue closeは行わない。
