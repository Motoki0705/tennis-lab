---
id: group-i551-dinov3-rope
type: group
title: DINOv3+3軸RoPE 時系列検出器 baseline (#551)
issue: 551
members:
- run-i551-dinov3-rope-tracknet-train-retry1
- run-i551-dinov3-rope-tracknet-t3-train
parents: []
tags:
- ball_detection
- dinov3
- rope
- tracknet
- temporal-detector
---

## まとめ

Issue #551 で実装した DINOv3 ViT-B/16 + 3 軸 RoPE bidirectional Transformer 検出器を tracknet 上で学習した最初の baseline 群（`num_frames` を 1 と 3 で振った 2 本。失敗した初回 run は対象外）。

| run | num_frames | test/f1 | precision | recall | mean_dist_px | test/loss |
|-----|-----------|---------|-----------|--------|--------------|-----------|
| [[run-i551-dinov3-rope-tracknet-train-retry1]] | 1 | 0.080 | 0.079 | 0.082 | 2.75 | 0.00087 |
| [[run-i551-dinov3-rope-tracknet-t3-train]] | 3 | 0.094 | 0.084 | 0.107 | 2.57 | 0.00093 |

結論:
- モデル契約（`T=1`→`T>1` を同一パラメータで学習可能）は機能し、forward/backward・shape は問題なく学習が回る。
- ただし両 run とも test/f1 が 0.08–0.09 と非常に低く、loss 収束と検出品質が乖離。frozen backbone・浅い decoder(4層)・20ep・focal_bce の背景支配が共通ボトルネックと推定。
- `num_frames=3` は T=1 比で recall(0.082→0.107) と位置精度(2.75→2.57px) を小幅改善し、時系列文脈が効くことを示すが効果は限定的。

次の方向性:
- num_frames を増やす前に、backbone 部分解凍（`last_n_blocks>0`/`full`）・学習エポック増・loss/peak_threshold/sigma 調整で絶対水準を底上げするのが費用対効果が高い。
- Issue #553 の評価パイプライン整備後に pred_test を取得し、同一プロトコルで定量比較する（本群は eval pipeline 整備前のため pred_test.npz 無し）。
