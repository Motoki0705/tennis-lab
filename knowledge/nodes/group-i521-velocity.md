---
id: group-i521-velocity
type: group
title: 角速度 (temporal) canonical 損失 (#521)
issue: 521
members:
  - run-i521-base-vel
  - run-i521-ex10-vel
  - run-i521-canonboth-vel
tags: [plcs, canonical, velocity]
---

## まとめ

関節角度の角速度損失 `canonical_rot_vel` を3アーキ（baseline 共有 / ex10 分離 auxpos / #520 最良
canon_both）へ導入した実験。

- `base_vel`: 角度は良好 (11.3°) だが共有 trunk のため位置破綻 (0.79m)。
- `ex10_vel`: 位置補助パスの分離で位置を回収しつつ角度 13.2° / 位置 0.270m を維持（**バランス最良**）。
- `canonboth_vel`: canonical-trunk 構成では位置を 0.364→0.298m に改善するが、角度・位置とも ex10 系に劣る。

velocity 損失は静的精度を一変させないが構成依存で位置に効く。**velocity 損失 + auxpos 分離 (ex10) が
現状の有力構成。** 「固まり（時間的に動かない）」緩和の定量評価には予測角速度 vs GT の専用評価が必要。
