---
id: group-i576-camtoken-posw
type: group
title: camtoken readout分離 × position_weight ablation (#576)
issue: 576
members:
- run-i576-camtoken-s0-h12
- run-i576-camtoken-posw4
- run-i576-camtoken-posw8
parents: []
relations:
- to: run-i545-s6-h0-auxoff-posw8
  rel: compares
tags:
- plcs
- shared-trunk
- readout-split
- camtoken
- position-weight
- chunked
---

## まとめ

共有 trunk + readout 分離（pose←cam0 / rot←cam1, `multiview_axial_camtoken`, S=0/H=12）上で `loss.position_weight` を 1→4→8 と振った ablation。共通プロトコルは canonical_rot / chunked_multiview_sequence_bs8 / batch=8 / seq_len[64,256] / 200ep / early-stop OFF。

| run | posw | 位置m (mean/med) | 回転° (mean/med) | 位置@0.5m | 角@15° |
|---|---|---|---|---|---|
| [[run-i576-camtoken-s0-h12]] | 1 | 0.564 / 0.471 | 10.18 / 7.55 | 0.542 | 0.797 |
| [[run-i576-camtoken-posw4]] | 4 | 0.353 / 0.295 | **8.45 / 6.02** | 0.789 | **0.854** |
| [[run-i576-camtoken-posw8]] | 8 | **0.313 / 0.249** | 8.62 / 6.98 | **0.831** | 0.841 |

参考（separate-trunk posw8, #545）: [[run-i545-s6-h0-auxoff-posw8]] **0.166m / 8.46°**（位置ベスト）。

### 結論

- **position_weight は camtoken の位置を大きく回復させる。** posw1→4 で位置 0.564→0.353m（-0.21m）かつ回転も 10.18→8.45° と同時改善、posw4→8 で位置 0.353→0.313m と更に改善。readout 分離で rotation が別トークンに逃げているため、posw 増で position を伸ばしても回転を崩さない。
- **位置のリターンは逓減**（posw1→4 で -0.21m、posw4→8 で -0.04m）。回転は posw4 が底（8.45°）で posw8 はわずかに悪化（8.62°）。トレードオフ点は posw4〜8。
- **回転は separate-trunk に並んだが、位置は ~0.15m 差が残る**（camtoken posw8 0.313m vs separate posw8 0.166m）。安価な共有 trunk で回転 8.5°級は達成できるが、位置最良は依然 trunk 分離。i576 当初の「位置には trunk 分離が必須」は **posw で大幅に緩和されるが完全には埋まらない**、と更新。

### スイートスポット

- 位置重視: posw8（0.313m / 8.62° / 位置@0.5m 0.831）。
- 回転重視: posw4（8.45° / 6.02° / 0.353m）。

### 次手

- 残る位置 gap は共有 trunk 内の表現競合由来と見られる。readout 分離 × 浅い共有（S5/H2）、または separate-trunk への readout 分離併用で位置 0.2m 切りを狙うのが本筋。
- posw>8 は逓減のため優先度低。
