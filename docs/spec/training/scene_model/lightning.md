# SceneModelLightningModule 仕様

`src/training/scene_model/lightning.py`: SceneModel の学習を司る LightningModule。

## 目的
- モデル組立、損失計算、最適化、スケジューリング、ロギング、可視化。

## 入出力と形状
- `forward(frames)` → `Mapping[str, Tensor]`
  - 代表（BBoxHead）: `bbox_center[b,t,q,2]`, `bbox_size[b,t,q,2]`, `exist_conf[b,t,q,1]`, `cls_logits[b,t,q,c]`
- `training_step(batch: SceneBatch, idx)`
  - `batch.frames: [B,T,3,H,W]`
  - `batch.targets: list[list[TargetFrame]]`
  - `batch.padding_mask: Bool[B,T]`

## 擬似コード

```
__init__(cfg):
  self.model       = build_scene_model(cfg.model, cfg.debug)
  self.head_adapter= HeadAdapter(cfg.model)
  self.denoiser    = DinoDenoiser(cfg.training.denoiser)
  self.metric_state= (matched:0, total:0)
  # 可視化・正規化用の mean/std をバッファ登録
  # バックボーン初期凍結
  save_hyperparameters(cfg)

training_step(batch):
  outputs  = self.model(batch.frames)
  dn_state = self.denoiser.make_noise(batch.targets)
  loss_dict= self.head_adapter.compute_loss(outputs, batch.targets, batch.padding_mask, dn_state)
  log_each('train/<key>', loss_dict[key])
  return loss_dict['total']

validation_step(batch):
  outputs  = self.model(batch.frames)
  dn_state = self.denoiser.make_noise(batch.targets)
  loss_dict= self.head_adapter.compute_loss(...)
  log_each('val/<key>', ...); metric_state.update(loss_dict)
  if should_log_images(batch_idx):
    grid = build_visual_grid(batch, outputs)
    if grid: tensorboard.add_image('val/bbox_predictions', grid, global_step)

on_validation_epoch_end():
  score = metric_state.matched / max(metric_state.total, eps)
  log('val/idf1', score); metric_state.reset()

configure_optimizers():
  adamw = AdamW(params, lr, weight_decay, betas)
  if scheduler:
    cosine = CosineAnnealingLR(adamw, T_max=max_epochs, eta_min=lr*min_ratio)
    return {optimizer: adamw, lr_scheduler: {scheduler: cosine, interval:'epoch'}}
  else:
    return adamw

on_train_epoch_start(): maybe_unfreeze_backbone(current_epoch)
```

## バックボーン凍結/解凍
- `cfg.model.backbone.freeze: true` で初期凍結。
- `cfg.model.backbone.unfreeze_after: int` 以上のエポックで `requires_grad=True` に戻す。

## 検証時の可視化
- 先頭バッチのみ、`viz.max_frames` 枚まで。
- `exist_conf`>閾値のボックス（なければ上位 `viz.max_boxes`）を `[cx,cy,w,h]→[x0,y0,x1,y1]` で描画。
- 画像はデノーマライズ（データセット正規化の mean/std 使用）。

## メトリクス
- `metric_state` は `num_matches/num_targets` を蓄積し、簡易 IDF1 風スコア `matched/total` を `val/idf1` として記録。

