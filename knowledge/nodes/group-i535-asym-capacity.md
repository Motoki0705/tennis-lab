---
id: group-i535-asym-capacity
type: group
title: '非対称容量フロンティア: 幅 vs 深さ (#535)'
issue: 535
members:
- run-i525-asym
- run-i535-asym-deep16-rerun
- run-i535-asym-wide-rerun
- run-i540-asym-wide
- run-i540-asym-deep16
parents: []
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- capacity-frontier
---

## まとめ

#525 → #535 の系: split-trunk の容量を**非対称に配分**(rotation trunk を深く/広く)して回転を伸ばせるかの探索。EX10(対称 split 78.1M, 9.98°/0.238m)と比較する。

> ⚠️ **【重要・訂正 2026-06-21】当初の「深さが回転の主レバー」という結論は撤回。** 当初の deep16/wide
> ランは作業ディレクトリ取り違え(main ツリー実行)で `rot_num_task_layers` が **no-op** になり、
> 非対称化が効いていなかった(deep16→対称 EX10 再学習 78.1M、wide→対称 768 172M)。`exp/i525-asym`
> worktree で再実行した有効値が下表の "(再)"。詳細は #535 コメント / [[plcs-asym-depth-worktree-only]]。

| run | 容量配分 | params | 回転° | 位置m | 状態 |
|---|---|---|---|---|---|
| (参考) run-i518-exp10 (EX10) | 対称6層 / 幅512 | 78.1M | 9.98 | 0.238 | 基準 |
| run-i525-asym | rot深10 / 幅512 | 103.8M | 19.94 | 0.700 | 有効(worktree 実行) |
| **run-i535-asym-deep16-rerun (再)** | **rot深16** / 幅512 | **142.3M** | **10.41** | **0.252** | 有効・full収束 |
| **run-i535-asym-wide-rerun (再)** | rot深10 / **幅768** | **228.7M** | **60.56** | **0.894** | 有効だが未収束(batch2) |
| ~~run-i540-asym-deep16~~ | (意図 rot16) | ~~78.1M~~ | ~~8.40~~ | ~~0.207~~ | **無効(no-op=EX10)** |
| ~~run-i540-asym-wide~~ | (意図 rot10/768) | ~~172M~~ | ~~12.27~~ | ~~0.368~~ | **無効(no-op=対称768)** |

**訂正後の結論**:
1. **非対称容量(深さ/幅)は固定小データ・200ep では EX10 を改善しない**。rot 深さ(幅512): 6層=9.98° / 10層=19.94° / 16層=10.41° と**非単調**で、いずれも EX10 を超えない。deep16 は train 8.16° まで収束しても val/test 10.41° ＝ 余剰容量を小データで活かせず軽い過学習。
2. **wide(228.7M)は学習破綻(60.56°)**。ただし batch=2(VRAM 制約)＋過大容量での**最適化失敗/未収束**(train すら 40°、ep92 早期終了)であり、「幅が無効」の証拠ではない。容量の公平判定にはデータ拡充と適正バッチが要る。
3. 総じて **EX10(78.1M, 対称6層)が依然最良**。容量を非対称に振っても本データ規模では割に合わない。
4. 「深さが効くか」の独立シグナルは #536(`run-i541-parameff-deeppose`: width256 で depth 3→6, 9.55°)に残るが、これは under-depth→適正への改善であり、512幅で 6→16 に深める話とは別。

次手: 容量が効くかの**公平判定**を #539(chunked・データリッチ・勾配累積で effective batch を回復)で行う。本群の固定小データ結論はその前提条件として確定。
