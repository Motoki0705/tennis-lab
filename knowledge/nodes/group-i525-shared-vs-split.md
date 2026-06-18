---
id: group-i525-shared-vs-split
type: group
title: 共有 trunk vs 分離 trunk（パラメータ数対照）(#525)
issue: 525
members:
  - run-i525-shared-match-dim
  - run-i525-shared-match-layers
tags: [plcs, canonical, shared-trunk]
---

## まとめ

「分離 trunk の優位はパラメータ数増のためでは？」を検証する param-matched 共有 trunk 対照。次元拡張で
合わせた `match_dim` は角度のみ分離型に匹敵（位置は崩れる）、層数増で合わせた `match_layers` は学習崩壊。
**パラメータ数だけでは説明できず、増やし方とアーキテクチャの寄与がある**のが暫定結論。
（キュー drain 後に `i525_shared_6l` を追加予定。）
