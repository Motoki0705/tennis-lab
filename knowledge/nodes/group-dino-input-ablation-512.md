---
id: group-dino-input-ablation-512
type: group
title: DINO入力チャネル変換 512-step アブレーション
members:
- run-dino-input-abl512-repeat-rgb
- run-dino-input-abl512-learnable-1x1
- run-dino-input-abl512-red-only
parents: []
tags:
- court-alignment
- dino
- detr
- lora
- input-ablation
---

## まとめ

公式DINO Swin-L 4-scaleをLoRA fine-tuneし、1ch court-line heatmapから複数courtのAABB・長辺scale・180度周期の軸方向を推論する512-step比較。入力以外はseed 42、800×800、train/val/test 512/128/128、identity augmentationで固定した。

検出F1とtotal lossは`red_only`、軸角・corner精度は`repeat_rgb`が最良だった。`learnable_1x1`は開始時に`repeat_rgb`と同一写像であるにもかかわらず短期性能が低く、512 stepでは追加自由度の恩恵を確認できなかった。

全条件で予測court総数194はGT総数194と一致したが、32 px corner gateのTPは26–43に留まった。中心位置よりscale・orientation・extentが主要ボトルネックであり、1 epochの結果だけで入力方式を確定しない。次は`red_only`と`repeat_rgb`を複数seed・長期予算で比較し、最終alignment用途に合わせてF1とcorner/angleを同時評価する。
