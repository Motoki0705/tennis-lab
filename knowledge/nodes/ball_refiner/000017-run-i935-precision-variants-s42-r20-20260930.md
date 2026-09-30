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
status: planned
config: {seed: 42, detector: mixed-e9, context: detector_only}
metrics: {cpu_meiji_detector_within8_frames: 14182, cpu_meiji_refiner_error_median_px_on_detector_within8: 19.737589086433406}
artifacts: {run_dir: knowledge/runs/run-i935-precision-variants-s42-r20-20260930}
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
plannedはGPU実験の状態。ここまでのCPU診断は完了している。
pipeline default、BallGMM2D契約、NLL目的を変更せず、court/person/poseとtestを使わない。
