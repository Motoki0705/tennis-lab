---
id: group-geometric-residual-v2
type: group
task: plcs
sequence: 111
recorded_at: '2026-09-21'
title: Court14校正・持続誤検出・損失と入力conditioningの比較
members:
- run-plcs-residual-v2-balanced-s42
- run-plcs-residual-v2-legacy-s42
- run-blcs-residual-v2-balanced-s42
- run-blcs-residual-v2-legacy-s42
- run-residual-v2-conditioning-cpu
- run-blcs-residual-v2-balanced-s42-resume
- run-plcs-residual-v2-asinh-legacy-s42
- run-blcs-residual-v2-asinh-legacy-s42
parents:
- group-geometric-residual-v1
papers: []
tags:
- triangulation-residual-v2
- court-calibration
- persistent-errors
- loss-ablation
- conditioning
status: done
artifacts:
  report: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/report.md
  dashboard: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/index.html
  summary: /mnt/c/Users/kamim/.codex/visualizations/2026/09/20/01a0bf91-5494-74c0-8793-3424866618ed/geometric-residual-v2/summary.json
---

## まとめ

Court14再推定、四隅＋両端中央フェンス付近の6候補、秒単位の持続誤検出を実装し、3つのscoutで方式案を独立に検討した。既存ACCAD/物理worldとsplitを再利用する。全20000 sceneのepoch0入力走査は失敗0。

同じv2データ・モデル・seed42で各taskのlegacy/raw、balanced/raw、legacy/asinhを比較した。候補間もvalidation平均3D誤差で選択し、PLCSはlegacy/raw、BLCSはbalanced/rawを実験recipeの既定とする。いずれも30epochのepoch29が各runの最良だった。production checkpointは更新しない。

| task | 初期test平均 m | legacy/raw | balanced/raw | legacy/asinh |
|---|---:|---:|---:|---:|
| PLCS | 0.155524 | 0.119929 | 0.131759 | 0.128067 |
| BLCS | 0.844004 | 0.826194 | 0.812402 | 0.812641 |

PLCSは従来損失が新損失より良く、新損失bundleの一律採用は支持されない。BLCSの新損失はmean/p95を改善したが、point中央値は0.079263→0.080593mへ微悪化。asinhはBLCSのlegacy対照には有効だったが、PLCSには悪化し、両taskの最良validation構成も更新しなかった。CPU embedding診断の改善を精度改善の証明とはしない。

全6条件のscene/GT/初期3D/frameとinit mask/severity/familyは保存NPZで完全一致。PLCSの選択構成は改善scene率55.0%、BLCSは56.4%。最大補正1%を除いた平均gainもそれぞれ+0.022812m/+0.023910mで、裾1%だけの改善ではない。一方、改善point率は39.5%/40.1%で、cleanを含む多数点の微小悪化は残る。誤差成分別の詳細は各runのdiagnostic_metrics.jsonを参照する。

MeijiのPLCSは同じraw-valid四肢edgeで1m超28→2件、最大骨長3.038→約1.1mへ減ったが、frame320の前腕は約0.92mで依然長すぎる。選択PLCSの再投影平均3.252870→3.739684px、選択BLCSは2.301307→2.305067px。BLCS実補正平均は約2mmで、実用的な位置改善は確認できない。実clipに独立3D正解はなく、骨格異常低減や再投影を絶対3D精度へ読み替えない。

BLCS balancedはepoch23後にnative workerが中断し、同epoch24全8000件のfresh単一process走査は成功した。原因は未確定。checkpointからoptimizer/scheduler/best historyを保ちworker0で完走したが、無中断と後半batch順/RNG進行の完全一致は保証しない。新runtimeのspawn/OpenCV1threadはsampleのバイト一致・epoch共有を確認し、conditioningの両30epochは完走した。

単一seed、未校正の合成誤差仮定、ACCADのsubject重複、実clip/別会場の独立3D正解不足が制限。既定選択はこの実験family内の判断であり、v1や既存deployの異なる評価条件とは直接順位付けしない。次はcamera-onlyの観測可能性、低誤差点の不要補正、長い欠測と実測誤差分布、複数seed/会場を検証する。
