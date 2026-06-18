---
id: group-i521-velocity
type: group
title: 角速度 (temporal) canonical 損失 (#521)
issue: 521
members:
  - run-i521-base-vel
  - run-i521-ex10-vel
tags: [plcs, canonical, velocity]
---

## まとめ

関節角度の角速度損失 `canonical_rot_vel` の導入実験。velocity 損失は角度を大幅改善 (17.8°→11.3°)
する一方で位置を悪化させるが、位置補助パスの分離 (`ex10`) で位置を回収しつつ角度 13.2° を維持できる。
**velocity 損失 + auxpos 分離が現状の有力構成。**
