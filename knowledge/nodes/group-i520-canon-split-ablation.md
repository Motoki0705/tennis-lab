---
id: group-i520-canon-split-ablation
type: group
title: canonical pose パス分離アブレーション (#520)
issue: 520
members:
  - run-i520-canon-none
  - run-i520-canon-rot
  - run-i520-canon-pos
  - run-i520-canon-both
tags: [plcs, canonical, split-trunk]
---

## まとめ

canonical pose のパス分離を none / rot / pos / both の 4 パターンで比較。両分離 (`canon_both`) が
角度最良 (15.9°) だが位置は悪化、位置単独分離 (`canon_pos`) は逆効果。**分離は角度に効くが位置との
トレードオフがある**というのが #520 の結論。
