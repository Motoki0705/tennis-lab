---
id: group-court-align-kp14-sigma-ablation
type: group
title: KP14 Ground-UV alignment σアブレーション
members:
- run-court-align-kp14-ablation-sigma-075
- run-court-align-kp14-ablation-sigma-100
- run-court-align-kp14-ablation-sigma-150
- run-court-align-kp14-ablation-sigma-200
parents: []
relations: []
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- cnn
- sigma-ablation
---

## まとめ

4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42、architecture、loss、data を固定し、KP Gaussian target の σ と output directory だけを変えた 4-run 比較である。vote radius 3 px と center-vote mask も固定したため、観測差は KP target 幅の sweep として解釈できる。小予算 pilot は未学習であり、この group と σ 選択から除外した。

表の headline test 値は各 run の `metrics.json` 完全値を小数第7位に丸め（末尾ゼロは省略）て表示する。test loss だけは各 run の `diagnostic_metrics.json` にある `loss` を同じく小数第7位に丸めた表示である。

| σ | F1 | count acc | KP px | center px | rotation ° | scale | translation px | test loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.9954038 | 0.986328 | 1.329173 | 0.283171 | **0.048468** | **0.0008015** | **0.057766** | 0.0284996 |
| 1.0 | 0.9927963 | 0.978516 | 1.475431 | 0.279737 | 0.054825 | 0.0008419 | 0.064545 | 0.0207129 |
| 1.5 | **0.9973684** | **0.992188** | 1.079006 | 0.268355 | 0.073458 | 0.0009741 | 0.075331 | 0.0135679 |
| 2.0 | **0.9973684** | **0.992188** | **0.820975** | **0.248675** | 0.082812 | 0.0010853 | 0.080233 | **0.0105472** |

観測として、σ=2.0 は KP、center、test loss が最良で、F1 と count accuracy も σ=1.5 と同率首位だった。σ=0.75 は Sim(2) rotation・scale・translation residual が最良だった。σ=1.0 はいずれの採用指標でも首位にならず、localization には 2.0、pose residual には 0.75 という用途別候補の間で、本比較では支配的な選択肢ではない。

次の real/noisy heatmap inference には σ=2.0 を localization baseline として推奨する。ただし absolute KP 精度ではなく pose residual を重視する経路の代替として σ=0.75 を保持し、実入力上でトレードオフを再評価する。
