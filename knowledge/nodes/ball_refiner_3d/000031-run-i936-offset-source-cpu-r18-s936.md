---
id: run-i936-offset-source-cpu-r18-s936
type: run
task: ball_refiner_3d
sequence: 31
recorded_at: '2026-10-01'
title: 保存入力・出力のoffsetと窓内translation損失のCPU診断
issue: 936
provider: codex
status: done
date: '2026-10-01'
config: {validation_rallies: 16, primary_update: 20000, oracle_minimum_three_visible_free_frames: 3}
metrics: {good_input_frames: 3031, good_input_reprojection_p50_px: 1.48652, c_flow_good_input_reprojection_p50_px: 10.10192, repro3_flow_good_input_reprojection_p50_px: 8.93291, c_flow_negative_translation_direction_windows: 43}
artifacts:
  run_dir: knowledge/runs/run-i936-offset-source-cpu-r18-s936
parents: [run-i936-repro3-512-physics10-r17-s936, run-i936-reprojection-gap-r17-s936]
relations: [{to: run-i936-condition-readout-r11-s936, rel: extends}]
papers: []
tags: [cpu, diagnostic]
---

## 結論と制限

[結果前に固定したCPU診断](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5922681411)を実施した。
**良好観測での約10pxのずれは、入力の全混合平均に同じずれがあることやfloat32の座標刻みだけでは説明できない。**
全16val/6,383frameの(c)とrepro3、両armの平均・全sampleを保存値から調べた。入力全125成分を保持し、
成分・seed・checkpointの選別、予測の再生成、追加val/test、実Meiji、GPUを使っていない。
oracle診断の位置補正を予測成果物にせず、残差出力モデル・新しい性能・正式合格とは扱わない。

## 入力に同じoffsetがあるか

事前固定した「3camera可視」4,831frameと、その中で入力混合平均の各camera誤差が全て5px以下の3,031frameを比較した。
後者は診断だけのGT依存層別であり、学習/正式評価からframeを除外する規則ではない。

| 方法 | 3camera再投影p50 px | 良好入力層の再投影p50 px | 良好入力層の3D誤差p50 m | 同3D RMSE m |
|---|---:|---:|---:|---:|
| 全混合平均 | 2.109 | 1.487 | .059 | .093 |
| RTS | 1.886 | 1.369 | .062 | .720 |
| (c) flow平均 | 11.067 | 10.102 | .279 | .475 |
| (c) flow全sample | 11.079 | 10.081 | .280 | .475 |
| repro3 flow平均 | 9.984 | 8.933 | .262 | .438 |
| repro3 flow全sample | 9.991 | 8.939 | .261 | .439 |
| (c) regression | 8.829 | 7.872 | .245 | .446 |
| repro3 regression | 8.143 | 7.172 | .338 | .477 |

良好入力層の全混合平均の符号付きXYZ誤差は[.0008, .0117, .0019]mだが、(c) flowは[-.1858, -.0311, -.0326]m。
入力誤差との軸別相関は.042/.054/.044、平均cosine .037と弱い。回帰のX平均誤差は逆符号の+.067m。
同じ入力bank/scaleの一様なバイアスがそのまま伝わった、という説明を支持しない。
全3camera層での相関は.5〜.6あり、大外れや周辺frameからの影響は否定していない。
RTSも良好入力層に裾誤差があり、中央値の良さだけを完全な解決とはしない。

(c) flowの5frame trend/residual p50は10.975/.172px、repro3は10.079/.304px。
再投影係数増加はtrendを少し下げる一方で細かな残差を増やし、全体の粗さ悪化と整合する。
全数値/符号付き統計は[analysis.json](../../runs/run-i936-offset-source-cpu-r18-s936/results/analysis.json)、
分布は[ECDF図](../../runs/run-i936-offset-source-cpu-r18-s936/results/input-output-ecdf.png)。

## 座標表現と出力解像度

実際の正規化は全XYZを11.885mで割る同一scale。全16入力の真/推定camera行列も一致した。
保存された224平均/sample系列で、float32のround trip最大座標差1.91e-6m、3camera可視での再投影差最大.000222px。
次のfloat32表現値への1ULP変位も3.81e-6m/.000935pxで、約10pxとの差を説明しない。
`diffusion/model.py`は時間長Tを保つTransformerと各frameの連続なLinear(width,3)絶対座標headで、
粗いheatmap/binや時刻downsamplingはない。これは数値単位・量子化に関する検査であり、
非線形なpooling、LayerNorm、幅128の表現容量や時間混合で精度を失う可能性は残る。

## 現在の損失はずれの修正を好むか

元の56所有窓のうち3camera可視かつfree-flightが3frame以上ある45窓を対象とし、不適格11窓は件数とsupportを保存。
各sampleのそのsupportでの平均GT誤差を一定translationとして窓の全座標から引き、**全成分の元のStudent-t NLL**、x0、physicsを採点した。
窓内の一定translationは加速度を変えない。計算はfloat64で、元float32入力のcamera/covariance値と実時間を保持する。
窓間のseamはこの局所損失診断に含めない。正解から作るtranslationをモデルが推定できることは未証明。

| 方法 | NLLが下がる窓/sample | 修正方向へのNLL微分<0 | 重み付き位置loss微分<0 | NLL平均 before→after |
|---|---:|---:|---:|---|
| (c) flow平均 | 36/45 | 43/45 | 43/45 | 10.139→9.240 |
| (c) flow全sample | 144/180 | 170/180 | 171/180 | 10.140→9.241 |
| repro3 flow平均 | 35/45 | 41/45 | 41/45 | 9.798→9.026 |
| repro3 flow全sample | 140/180 | 164/180 | 164/180 | 9.798→9.027 |
| (c) regression | 33/45 | 43/45 | 43/45 | 9.671→9.217 |
| repro3 regression | 31/45 | 43/45 | 43/45 | 9.346→9.028 |

表のNLL平均は45窓（sampleは180）の単純平均で、学習ログや全frame平均と同じ集計ではない。
physics差は全630窓/sampleケースで最大1.51e-14。x0平均も減った。
630件全てのNLL方向微分を中央差分で検証し、最大絶対差1.96e-7、固定許容差内だった。
[窓ごとの原数値](../../runs/run-i936-offset-source-cpu-r18-s936/results/oracle-windows.json)には改善しない窓も残した。

したがって「約10pxのoffsetを現在の損失が常に好む」「physicsが一定translationの修正を必ず妨げる」とは説明できない。
ただしこれは**出力空間の微分**であり、network parameter勾配、学習flow-time、訓練分布上の最適化や一般化を証明しない。
Student-t NLLとGT画素誤差の目的差も残るため、損失形そのものを無条件に正しいとはしない。

## 選んだ次の検証

**GPUは投入せず、現(c)20kの両encoderについて、pool直後のtokenから全混合平均を読むCPU read-outを次に行う。**
旧bank/64trainで成功した[旧診断](000017-run-i936-condition-readout-r11-s936.md)は、現bank/512train/physics1e-3のencoderの証明ではない。
まず入力情報がtokenに残るかを測れば、2D tokenや損失を同時に変更する前にconditioning encoderとその後段を切り分けられる。
現encoderでも復元できるなら、次は時間Transformer/絶対座標headと学習上の精度配分を対象にする。
復元できなければ、全成分符号化を保つ明示的なmoment特徴の追加を単因子候補として提案する。出力への残差加算はしない。

このrunではread-out自体は実行しない。次のCPU検証は512train全frameで全混合平均だけを教師にし、
切片付きfloat64 SVD、ridgeなし、同16val、両armの現20k重み固定。元node000017と同じval復元RMSE<=.10mを診断基準とし、
camera層と3camera p50/p95も全て報告する。合成GTをfit教師にせず、係数/閾値/seedのval調整・rank不足fallbackをしない。
各arm10分上限を見積り、CPU1process/native1、RAM2〜3GB/空き6GiB下限、出力100MB以下。
費用は旧64trainの42.76秒からの概算で、本番性能の合格には使わない。

## 検証と資源

元(c)/repro3の全体再投影mean/p50/p95/件数・RMSE計40項目と一致。入力16NPZを含む106hashを前後照合。
全125成分、同16valのみ。全sampleの精度検査、camera parity、方向微分は[audit.json](../../runs/run-i936-offset-source-cpu-r18-s936/audit.json)。
主診断5.347秒、peak RSS.947GB、最低空きRAM24.148GB。結果約1MB、GPU0、新dataset0。
`analyze.py`/`audit.py`は当該bundleに保存。親checkoutをPYTHONPATHにしてCPU/native1/CUDA無効で実行する。
