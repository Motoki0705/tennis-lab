---
id: group-i536-parameff-frontier
type: group
title: '縮小 split 効率フロンティア: 深さ vs 幅 vs 長期学習 (#536)'
issue: 536
members:
- run-i525-parameff
- run-i541-parameff-deeppose
- run-i541-parameff-medcap
- run-i541-parameff-longtrain
parents: []
tags:
- plcs
- canonical
- split-trunk
- parameter-efficiency
- efficiency-frontier
---

## 考察 / Findings

#525 → #536 の系: split-trunk の**パラメータ効率の上限**探索。縮小 split(eff: 256幅/3層, 9.9M)を起点に、容量を「深さ」「幅」で足す/学習を延ばす、の各方向で 200(一部400)ep 収束させ、EX10(split 78M, 9.98°/0.238m)との効率フロンティア上の位置づけを見る。

| run | 変更軸 | params | ep | 回転° | 位置m | 備考 |
|---|---|---|---|---|---|---|
| run-i525-parameff | 基準(256幅/3層) | 9.9M | 200 | 15.55 | 0.569 | 縮小 split 基準 |
| **run-i541-parameff-deeppose** | **深さ 3→6**(幅256維持) | 19.5M | 200 | **9.55** | **0.202** | 本群最良・効率フロンティア更新 |
| run-i541-parameff-medcap | 幅 256→384(4層) | 28.9M | 200 | 17.75 | 0.571 | 幅増しは無効 |
| run-i541-parameff-longtrain | epoch 200→400(eff) | 9.9M | 400 | 24.15 | 0.655 | 長期学習は逆効果 |
| (参考) run-i518-exp10 (EX10) | 78M split | 78M | 200 | 9.98 | 0.238 | 基準 |

**結論**:
1. **回転は『幅』ではなく『深さ』依存**。幅を据え置き深さだけ 3→6 にした deeppose が、回転 9.55°(≒EX10)・位置 0.202m(<EX10)を **EX10 の ~1/4(19.5M)** で達成。#525 の「回転=容量(幅)依存」は「回転=**深さ**依存」へ精緻化。
2. **幅増し(medcap)は失敗**。28.9M を幅に振っても 17.75° と eff 基準並み。同予算帯で深さ ≫ 幅。
3. **長期学習(longtrain)は逆効果**。under-capacity な eff を 400ep 回すと 200ep より悪化(過学習)。効率は epoch でなく容量配分で得る。
4. **#535 deep16 と独立に同結論**: 別 issue・別容量帯で揃って「深さが回転の主レバー」。

効率フロンティア最良 = **deeppose(19.5M, 9.55°/0.202m)**。次手: 深さ 6→8、幅 256→192 など下限再探索。
