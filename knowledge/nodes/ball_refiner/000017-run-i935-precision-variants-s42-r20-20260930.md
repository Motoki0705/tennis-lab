---
id: run-i935-precision-variants-s42-r20-20260930
type: run
task: ball_refiner
sequence: 17
recorded_at: '2026-09-30'
title: 候補精度の消失をCPU診断し平均parameterizationと学習長を比較
provider: codex
issue: 935
date: '2026-09-30'
status: failed
config: {seed: 42, detector: mixed-e9, context: detector_only}
metrics: {cpu_meiji_detector_within8_frames: 14182, cpu_meiji_refiner_error_median_px_on_detector_within8: 19.737589086433406}
artifacts:
  run_dir: knowledge/runs/run-i935-precision-variants-s42-r20-20260930
  log: knowledge/runs/run-i935-precision-variants-s42-r20-20260930/queue.log
  resource_usage: knowledge/runs/run-i935-precision-variants-s42-r20-20260930/resource_usage.json
repro:
  commit: 0d3fd3f8a2498da33edbaf04656c1dee81185c3b
  branch: campaign930/i935-12-precision-diagnosis
parents: [run-i935-detector-only-mixed-e9-s42-r18-20260930]
relations: [{to: run-i935-detector-only-ft-e13-s42-r4-20260928, rel: compares}]
papers: []
tags: []
---

## CPU診断と仮説

[全診断表・曲線](../../runs/run-i935-precision-variants-s42-r20-20260930/diagnosis-tables.md)、
[母数・source/camera/half・hashのJSON](../../runs/run-i935-precision-variants-s42-r20-20260930/diagnosis.json)、
[再現script](../../runs/run-i935-precision-variants-s42-r20-20260930/diagnose.py)を保存。
GPUを使わず、r18のobserved条件の280 paired NPZとbest更新時の216 selection NPZ、
cache manifest・学習曲線等の計501入力のhashを記録した。新旧ともsourceの同じframe/PTS/教師を確認した。
元mediaは再読込せず、人工gapの再診断もしていない。人工gapの既存比較は親ノード000016を参照。

検出器が8px以内でも新refinerはMeiji中央値19.74px、TrackNet16.20px、chat26.38px。
この群でrefiner−detectorのx差中央値はそれぞれ+18.46/+16.51/+24.57px、y差は+0.87/−0.23/+0.38px。
典型frameの劣化は大誤検出の裾とは別に生じており、絶対座標headの共通する右方向の偏りが主要因。
この偏りは同じ重みの推論結果の説明であり、最適化上の根本原因を確定したものではない。

sigmaは最大weight成分のsource画素共分散の長軸1σで、Meiji中央値17.81px。
最近候補距離中央値21.94px、≤2pxは0.7%に留まり、成分平均は候補peak上にほぼ乗っていない。
zは同成分のMahalanobis半径。Meiji p50/p90/p95=1.42/7.34/13.65で、
単一2D Gaussianの期待1.177/2.146/2.448より裾が大きい。
top成分の50/90/95%楕円coverageは37.9/76.4/80.3%。この値を全GMMのHDR較正と混同しない。
TrackNet/chatもz中央値は1.48/1.58、中心50%楕円は26.7/28.0%しか含まず、
単にsigmaが一様に過大なだけではない。sigmaを縮めるだけの案は根拠が弱い。

## 平均を生成する経路

現行[`model.py`](../../../src/tasks/ball_refiner/refiner_2d/model.py)は各候補のuv・score・
5×5 probability patch・境界maskを53次元から128次元へ埋め込み、候補集合attentionで
1frameを1tokenへ集約する。33frameの時間attentionを通した後に4成分×6量と存在logitを出す。
[`model_io.py`](../../../src/tasks/ball_refiner/refiner_2d/model_io.py)が2量をsigmoidに通し、
sourceのx/(W−1),y/(H−1)という**絶対座標**にする。候補へのskip接続やpeak選択はない。

保存証拠は288×512入力に対する72×128 native格子、K=8、NMS5、patch5。
1920×1080では1格子約15.1×15.2 source px。候補uvは上流のlog-parabolic subpixel補正済みで、
整数格子へ戻していない。patchだけが整数cell中心のnative値で再サンプルしない。
refinerも連続floatの平均を返し丸めないので、格子自体が15pxの精度下限を強制してはいない。
sigma下限0.001は1920×1080で各軸1.919/1.079px（相関を含む長軸下限とは異なる）。
現在の17.81pxの長軸中央値は設定下限より大きく、床を下げることを今回の施策にはしない。

## 学習不足と旧pilotとの差

新pilotのtrain joint NLLは250stepの−1.31から3,000stepの−4.93へ低下しており、
収束済みとは言えない。一方val選択NLLはepoch8で−1.42へ跳ね、bestはepoch10の−4.877、
最後は−4.568。単調なunderfitではなく最適化の揺れが強い。
保存されたMeiji選択側のdetector正解群ではepoch4/7のx偏り−14.15/−17.16pxが
epoch10では+18.30pxに反転する。固定の座標変換誤差だけでは説明できない。
学習不足・dropout・一定LRによる偏りの揺れの寄与は今回のCPU資料だけでは分離できない。
trainはgap/dropoutありjoint項、valはepoch末evalの位置項なので、その差を正確な汎化gapとはしない。
曲線はJSONLから描画した。TensorBoardログはこのrunnerでは生成していない。

TrackNetのdetector中央値は2.88→3.39pxという小幅な退行に対し、refinerは11.06→16.44px。
chatのdetectorは6.33→6.67pxだがrefinerは20.20→29.81px。新detectorが正しい同一frameに
限定してもrefinerの差が残る（表参照）。候補の良否だけでは説明できず、headの偏りが主な説明。
新detectorの裾改善はrefinerのMeiji/chatの裾改善に繋がる一方、両runのbestはMeiji選択側の
observed/gap NLLだけで選ぶため他sourceの典型位置を保証しない。選択規則は変更しない。
1seed・異なる入力証拠からの2回学習なので、入力変化と学習確率性の因果寄与の割合までは未確定。

## 次の実験

run20 directiveがrun19のcourt-only先行提案を上書きする。
同じepoch9 cache、seed42、source比、窓、損失、optimizer、val選択規則のまま、
長期化と候補を保持する平均parameterizationの最大3案を事前固定して比較する。
CPU診断は完了している。GPU実験は以下の監視障害で失敗し、3案の結果は得られていない。
pipeline default、BallGMM2D契約、NLL目的を変更せず、court/person/poseとtestを使わない。

## GPU投入前の3案固定

正本は[plan.json](../../runs/run-i935-precision-variants-s42-r20-20260930/plan.json)、
[実行command](../../runs/run-i935-precision-variants-s42-r20-20260930/command.txt)、
[CPU preflight](../../runs/run-i935-precision-variants-s42-r20-20260930/preflight.json)。
全678入力のhashを固定し、実行前後に照合する。全案がr18と同じdata manifestを生成し、
4,910窓・source内訳・18選択clip・gap・教師が完全一致した。r18実checkpointの310frameをCPU再推論し、
保存分布との既定許容差内の一致も確認した。出力root以外の変更因子は以下だけ。

| 案 | r18からの因子 | 期待と反証 |
|---|---|---|
| absolute_12k | 12→48epoch、3,000→12,000step（同じ250step/epoch） | 平均の学習不足なら偏り/中央値が減る。一定LRの揺れなら延長だけでは解消しない可能性。 |
| anchored_3k | 候補残差平均parameterizationのみ、3,000step | 3成分がscore上位候補を基準に±0.02uvの学習offsetを持ち、1成分は自由な平均。典型frameの候補精度を保つ。裾/欠損が退行する可能性も表で確認。 |
| anchored_12k | 上記parameterization＋12,000step | absolute_12kとは方式だけ、anchored_3kとは長さだけの比較。両因子の併用で典型位置とgap NLLを両立できるか確認。 |

全案はscratchからseed42、AdamW lr3e-4/weight_decay.01、batch32、source等比率、
33frame/stride16、gap確率.5/長さ1,4,8,16、compile inductor/defaultを維持。
同じMeiji選択側18clipのobserved/gap位置NLLの等重みでbestを選び、厳密改善時だけ更新するため同率は早いepoch。
test・較正側・TrackNet/chatをcheckpoint選択へ使わない。
anchored headの数学・完全な設定schema・無効候補時の自由平均・端点処理は
[README](../../../src/tasks/ball_refiner/README.md#候補残差平均の明示的な実験設定)を正本とする。
既存checkpointを変換したり、設定省略時に新方式を選んだりしない。
新e9のrecall@3/@8はMeiji87.72/90.85%、TrackNet98.76/99.02%、chat87.45/89.04%。
上位3候補外も含めて扱うため自由成分を残すが、この選択が最良とは未検証。

評価は70val clip×2条件×3案の420 NPZ。r19と同じ2,048サンプル・seed1729・HDR50/90/95%、
observed/source/camera/Meiji halfごとのp50/p90/p95、位置NLL・HDR coverageと面積、存在NLLを保存する。
候補と教師はcache/storeから読み、r18 pilotとe9 detectorの**保存済み**同一frame指標を参照列にする。
detector再推論はない。unknownはN/A、不在は存在NLLだけ、推定位置はobservedから分離。
比較表は各案の評価JSONを検査してから生成し、途中状態を完了として公開しない。

1queue job/resource=allの見積もりは**25–45分・peak VRAM2–4 GB・新規出力2 GB以内**。
r18実測132.97秒/3,000stepから27,000stepの学習を約1,197秒と見積もり、
cache/HDR評価・新compile cache・共有CPUの余裕を加えた。これは実測前の推定。
wall時間はtimeout3585秒＋TERM後15秒KILLで最大1時間、PyTorch allocator6GiB capと
device全体7.5GB停止監視（1秒poll）でgrant8GBへ余裕を持たせる。
poll間の瞬間peakを完全保証する測定ではない。RAM8GiB以上で起動し、6GiB未満で停止。
CPU数はtorch/OMP/MKL/OpenBLAS2、compile subprocess2、data loader0。
新規出力は専用compiler cacheも含め4.5GBで停止監視し、worktree追加なしで全5GB予算内に収める。
失敗・OOM・timeout時も再投入/精度変更/CPU fallback/学習延長を行わない。旧資産は全て保持する。

通常検証はCPU計44件成功（診断3、モデル/anchor26、学習/参照比較15、全てpytest -n4）。
CPU fullgraph capture、新anchorの勾配・初期peak保持・全gap・AMP・同score集合順序・設定欠落拒否・保存復元、
実tiny cacheの学習→best復元→比較→bundle exportを検査した。ruff/mypyも成功。
GPU比較結果は得られていない。実runtime/VRAMと障害は以下に記録する。

## run21で回収した監視障害

job `1790738050136194711_2100742_i935-precision-variants-s42-r20-20260930` は
**failed / exit_code=1、116.4639秒**。
[元queue job](../../runs/run-i935-precision-variants-s42-r20-20260930/queue.job)、
[ログ](../../runs/run-i935-precision-variants-s42-r20-20260930/queue.log)、
[資源記録](../../runs/run-i935-precision-variants-s42-r20-20260930/resource_usage.json)、
[repro metadata](../../runs/run-i935-precision-variants-s42-r20-20260930/run.json)を保存した。
共有queueのfailed/job/log/reproは削除・書換えしていない。
監視90回、device-used最大1,410,334,720 bytes（1.41 GB）、
CUDA allocated/reserved最大247,112,704 / 262,144,000 bytes。
停止時compiler cacheは148,215,476 bytes。上限超過やモデル精度による失敗ではない。

ログの最後の学習行は最初の`absolute_12k`のepoch0/step250。
watchdogは10回ごとのdisk検査で`output_size()`を呼ぶが、元実装の`Path.rglob()`は
ループ本体のtryより先に次の要素を取得する。そこで子ディレクトリが消えると
`FileNotFoundError`が漏れ、監視がSIGTERMを送り、Inductor中の主threadが停止する。
**同じPython 3.11で走査中に子ディレクトリを削除するテストが元実装で同じ例外になった**。
資源記録の停止理由も`FileNotFoundError(2, 'No such file or directory')`で一致する。
元ログは監視threadのtraceback/消失pathを保存していないため、消失した個別cache directory名は
確定できない。compile cacheの一時directory更新と整合し、モデル障害を示す証拠はない。

[部分出力一覧・hash](../../runs/run-i935-precision-variants-s42-r20-20260930/partial_outputs.json)の
全出力を元の場所に保持した。学習側はconfig/data_manifest/run_stateの3ファイルだけで、
checkpoint・学習曲線・val NPZ・variant比較は未生成。reportとcompiler cacheを含む回収時の
実サイズは148,535,536 bytes（停止後のcompiler終了処理による増分を含む）。
TensorBoard出力はこのrunnerでは生成しない。途中のbatch NLLをvalidation結果として扱わない。

修正とorchestratorが明示承認した同一3案の1job retryは
[run21](000018-run-i935-precision-variants-s42-r21-20260930.md)に分離する。
ここにある元plan・command・script・config・既存のCPU診断は失敗時の再現資料として保持する。
