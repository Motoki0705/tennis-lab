---
id: group-i535-asym-capacity
type: group
title: '非対称容量フロンティア: 幅 vs 深さ (#535)'
issue: 535
members:
- run-i525-asym
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

## 考察 / Findings

#525 → #535 の系: split-trunk の容量を**非対称に配分**して回転を伸ばせるかの探索。rotation trunk へ容量を「深さ」または「幅」で投下し、200ep で収束させて EX10(split 78M, 9.98°/0.238m)と比較する。

| run | 容量配分 | params | 回転° | 位置m | 備考 |
|---|---|---|---|---|---|
| run-i525-asym | rot深10 / 幅512 | ~103M | 19.94 | 0.700 | #535 当初: 深化は負(未収束疑い) |
| run-i540-asym-wide | rot深10 / **幅768** | 172M | 12.27 | 0.368 | 幅は効くが高コスト・resume必須 |
| **run-i540-asym-deep16** | **rot深16** / 幅512 | 78.1M | **8.40** | **0.207** | EX10 超え。本群最良 |
| (参考) run-i518-exp10 (EX10) | 対称6層 / 幅512 | 78M | 9.98 | 0.238 | 基準 |

**結論**:
1. **回転の主レバーは深さ**。同等予算(78M)で rotation trunk を 16 層まで深めた deep16 が EX10 を両指標で上回り、本群唯一の EX10 超え。
2. **幅は割に合わない**。768 幅(172M)の wide は 12.27° に留まり、resume を要してなお 78M の deep16 に負ける。
3. #535 当初の負(rot=10, 103M で 19.94°)は「深化が無効」ではなく、**過大容量(103M)の未収束**が主因と再解釈される。深さは学習可能なサイズに収まる限り有効。

次手: deep16 を新ベースラインに、rotation 深さ(12/16/20)の最適点 sweep と、幅を 512→384 に絞っても深さで回転を保てるか(#536 deeppose と接続)を検証する。
