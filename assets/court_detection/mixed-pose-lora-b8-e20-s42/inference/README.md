# Mixed-source dense+pose LoRA inference

`court_mixed_pose_lora_b8_e20_s42`の学習終了時checkpointを、TennisCourtDetector由来の実画像とSynthetic Court V3の合成画像へ適用した推論可視化です。

- checkpoint: `outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/last.ckpt`
- source: `data/court/images/-0M6ixK7aIU_*.png`の先頭8枚
- rendering: 2 fps、8 frames、入力画像と各headの予測を左右に表示
- evaluation metrics / convergence: [`knowledge/nodes/run-court-mixed-pose-lora-b8-e20-s42.md`](../../../../knowledge/nodes/run-court-mixed-pose-lora-b8-e20-s42.md)

## Keypoints

左は予測keypoint、右は全channelの最大heatmapです。KP decoderはchannelごとに複数peakを許容します。

![Keypoint inference](kp.gif)

## Court-cell segmentation

7クラスのcourt-cell予測を色分けしています。

![Segmentation inference](seg.gif)

## Court-line segmentation

line headの確率mapを白黒で表示しています。

![Line inference](line.gif)

## Synthetic Court V3

Synthetic Court V3のtest splitに保存された`rgb.npy` 8枚へ、同じcheckpointと各headを適用します。

### Keypoints / heatmap

![Synthetic keypoint inference](synthetic-kp.gif)

### Court-cell segmentation

![Synthetic segmentation inference](synthetic-seg.gif)

### Court-line segmentation

![Synthetic line inference](synthetic-line.gif)

## Reproduction

各headは次の共通overrideに`visualization=kp|seg|line`と対応する`visualization.save`を指定して生成します。GPU実行時はrepositoryのtraining queueを使用します。

```bash
python -m src.tasks.court_detection.scripts.visualize \
  visualization=kp \
  'visualization.image_source=court/images/-0M6ixK7aIU_*.png' \
  'visualization.checkpoint=court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/last.ckpt' \
  visualization.max_frames=8 \
  visualization.fps=2 \
  visualization.save=court_detection/mixed-pose-lora-b8-e20-s42/inference/kp.gif \
  run.device=cuda
```

合成データでは`image_source`を再帰globへ変更します。

```bash
python -m src.tasks.court_detection.scripts.visualize \
  visualization=kp \
  'visualization.image_source=synthetic_data_generation/scenes/B00/datasets/court/samples/test/**/rgb.npy' \
  'visualization.checkpoint=court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/last.ckpt' \
  visualization.max_frames=8 \
  visualization.fps=2 \
  visualization.save=court_detection/mixed-pose-lora-b8-e20-s42/inference/synthetic-kp.gif \
  run.device=cuda
```
