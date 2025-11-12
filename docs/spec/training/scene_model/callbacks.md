# ロガー/コールバック 仕様

`src/training/scene_model/callbacks.py`: TensorBoard ロガーと代表的なコールバックを設定から構築。

## 目的
- 実験名・保存先を `cfg.logging` から決定し、ロガーとコールバックを返す。

## 擬似コード

```
def build_logger(logging_cfg, experiment_name=None):
  cfg = logging_cfg.logger
  return TensorBoardLogger(
    save_dir = cfg.save_dir or 'runs',
    name     = experiment_name or cfg.name or 'scene_model',
    version  = cfg.version,
    default_hp_metric = cfg.default_hp_metric or False,
  )

def build_callbacks(logging_cfg):
  cfg = logging_cfg.callbacks
  cbs = []
  if cfg.checkpoint:
    cbs += [ModelCheckpoint(
      monitor=cfg.checkpoint.monitor or 'val/total_loss',
      mode=cfg.checkpoint.mode or 'min',
      save_top_k=cfg.checkpoint.save_top_k or 1,
      save_last=cfg.checkpoint.save_last or True,
      every_n_epochs=cfg.checkpoint.every_n_epochs (optional),
      filename=cfg.checkpoint.filename (optional),
    )]
  if cfg.lr_monitor:
    cbs += [LearningRateMonitor(logging_interval=cfg.lr_monitor.logging_interval or 'epoch')]
  return cbs
```

## 主な設定キー
- `logging.logger.save_dir/name/version/default_hp_metric`
- `logging.callbacks.checkpoint.{monitor,mode,save_top_k,save_last,every_n_epochs,filename}`
- `logging.callbacks.lr_monitor.logging_interval`

