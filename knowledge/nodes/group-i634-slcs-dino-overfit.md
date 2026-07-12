---
id: group-i634-slcs-dino-overfit
type: group
title: SLCS単一clip DINO有無比較 (#634)
issue: 634
members:
- run-i634-slcs-overfit-no-dino
- run-i634-slcs-overfit-dino
parents: []
tags: [slcs, overfit, dino, ablation]
---

## まとめ

単一clip・13 windowをtrain/val/testで共有するmemorization条件では、DINOありrunがDINOなしrunよりplayer位置2.8%、yaw 21.2%、ball位置7.2%改善した。SLCSのend-to-end学習経路とDINO cross-attentionは実データで動作している。

ただし、これは同一window上の過学習比較であり汎化性能ではない。player位置は約0.47 mまで記憶できた一方、ball位置は約1.95 mに留まった。次段階ではrecording非重複splitと、ball smoothness / DINO samplingのablationが必要。
