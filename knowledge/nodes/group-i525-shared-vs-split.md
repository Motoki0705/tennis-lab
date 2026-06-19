---
id: group-i525-shared-vs-split
type: group
title: 共有 trunk vs 分離 trunk（パラメータ数対照）(#525)
issue: 525
members:
  - run-i525-shared-6l
  - run-i525-shared-match-dim
  - run-i525-shared-match-layers
tags: [plcs, canonical, shared-trunk]
---

## まとめ

「分離 trunk の優位はパラメータ数増のためでは？」を検証する param-matched 共有 trunk 対照。基準は
単一 6 層共有 `shared_6l`（39.3M, `15.27° / 0.836m`）＝分離 EX10(78M) の半分。

- `match_dim`（幅で倍化, 79M）: `12.22° / 0.848m` … 角度は分離型 (9.98°) に肉薄するが**位置は不変**。
- `match_layers`（深さで倍化, 78M）: `29.80° / 1.617m` … **基準 6L より悪化**（学習崩壊）。

**位置は共有 trunk ではパラメータ容量によらず約 0.84m で頭打ち**で、深さ増はむしろ悪化。
→ 分離 trunk の優位はパラメータ数ではなく**アーキテクチャ（タスク分離）に起因**するのが結論。
回転は容量（幅）で改善するが位置は分離が必須、という非対称性も得られた。
