---
id: group-geometric-residual-v1
type: group
task: plcs
sequence: 105
recorded_at: '2026-09-21'
title: PLCS・BLCSの三角測量残差モデル初回検証
members:
- run-plcs-triangulation-residual-accad-v1
- run-blcs-triangulation-residual-physics-v1
parents: []
papers: []
tags:
- plcs
- blcs
- triangulation-residual
---

## まとめ

PLCS/BLCSの入力を観測2D・再投影2D・差分・Court14・三角測量3D・cameraに統一し、confidence/maskとともに残差学習するprofileを実装した。PLCSはhip中点root residualとcourt-oriented relative pose residual、BLCSはball XYZ residualを出力する。

| 評価 | PLCS 初期値→補正後 | BLCS 初期値→補正後 |
|---|---:|---:|
| 合成test 平均3D誤差 | 0.421834→0.337163m | 0.595380→0.589038m |
| 合成test 改善率 | 20.07% | 1.07% |
| 実clip 平均再投影誤差 | 3.252870→5.938328px | 2.301307→2.324268px |

合成側でPLCSは改善したが、指定実映像での改善は確認できていない。PLCSのraw-validな同一関節で測った1m超の四肢骨長は28件/26 person-frameのまま。BLCSの実補正はほぼ一定の5mmだった。
実clipには独立3D正解がなく、再投影誤差を真の3D誤差として扱わない。checkpointは合成validationだけで選び、実clipへ合わせた選択をしなかった。

PLCSはACCAD source-motion-disjoint splitで30epoch、best epoch29。BLCSはphysics scene splitで13epoch、best epoch6。両者とも共有GPU queueを経由し、best checkpointのtestと実clip全1010frame推論を実行した。詳細な構成・slice別メトリクス・解釈は各run nodeを正本とする。

- [PLCS run](000104-run-plcs-triangulation-residual-accad-v1.md)
- [BLCS run](../blcs/000034-run-blcs-triangulation-residual-physics-v1.md)
- [実装・実行方法](../../../src/tasks/base/triangulation_residual/README.md)

学習前validatorは指定1回、試行1回、完了1回。判定はCHANGES_REQUIRED。VAL-01（BLCSの初期化不能なcamera集合）とVAL-02（weights-only checkpoint読込）を採用修正した。修正後は通常テスト125件、型/設定監査、実際のcheckpoint保存・再読込・test、全20000sceneのepoch0入力走査を確認した。最終修正後の独立再評価はしていない。

比較動画・NPZ・推論用checkpoint・評価JSONは会話artifactのgeometric-residual/{plcs,blcs}へ保存。推論用checkpointはoptimizer状態を除き、元best checkpointとstate_dict完全一致およびstrict reloadを検証した。学習再開用の元checkpointはoutputs/{plcs,blcs}/triangulation_residual_v1_20260921に保持する。
