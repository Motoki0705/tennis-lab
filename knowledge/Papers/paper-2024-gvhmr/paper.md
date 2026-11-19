---
id: paper-2024-gvhmr
type: paper
title: World-Grounded Human Motion Recovery via Gravity-View Coordinates
year: 2024
authors:
- Zehong Shen
- Huaijin Pi
- Yan Xia
- Zhi Cen
- Sida Peng
- Zechen Hu
- Hujun Bao
- Ruizhen Hu
- Xiaowei Zhou
tasks:
- plcs
- tennis_scene
source: https://arxiv.org/abs/2409.06662v1
summary: 重力とカメラの視線方向で定義するGravity-View座標系に人体姿勢を推定し、カメラ回転を用いて世界座標へ戻す研究。
reason: GVHMRをPLCSのmotion sourceに採用する研究上の根拠を確認し、実動画への適用可能性を検討するため。
external_ids:
  arxiv: '2409.06662'
---

## このプロジェクトとの関係

PLCSのmotion sourceとして用いるGVHMRの出典。今回の整理で、既存のGVHMRモーション生成試験・学習記録に背景研究として参照を追加した。元の実験時点で論文の手法を比較検証したという意味ではない。

## 検証したい仮説・適用限界

GVHMR由来の動作を混ぜることが実動画でのPLCS精度改善に繋がるかは、固定split・同一budgetでの比較が必要。論文の世界座標系と、このrepoのコート座標系との整合は別途検証する。
