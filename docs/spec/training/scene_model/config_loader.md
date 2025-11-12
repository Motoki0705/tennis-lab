# ConfigLoader / 設定ロード 仕様

`src/training/utils/config.py`: YAML 設定の階層マージと、Trainer/Logger/Callbacks/Module/DataModule の生成。

## 目的
- 断片化された YAML を一つの `DictConfig` に統合し、学習オブジェクトをファクトリで構築。

## 設定ロードの流れ

```
def load_cfg(path, overrides=None):
  base = OmegaConf.load(path)
  include_map = base['includes']              # {key: relative_path}

  merged = {}
  for key, rel in include_map.items():
    part = OmegaConf.load(path.parent / rel)  # 相対パスを解決
    merged = merge(merged, {key: part})

  # includes を除いた本体をマージ
  merged = merge(merged, base - {'includes'})

  # 環境変数で最小構成に（テスト/デバッグ用）
  if ENV.CFG_DEBUG_MINIMAL in {'1','true','yes','on'}:
    merged = merge(merged, {debug: {minimal: True}})

  # 末尾の上書き（dotlist）
  if overrides:
    merged = merge(merged, OmegaConf.from_dotlist(overrides))
  return merged
```

## オブジェクト生成（ConfigLoader）

```
class ConfigLoader:
  cfg: DictConfig

  def build_datamodule():
    return DancetrackDataModule(cfg.dataset, cfg.debug)

  def build_lit_module():
    return SceneModelLightningModule(cfg)

  def build_callbacks():
    return build_callbacks(cfg.logging)

  def build_logger():
    return build_logger(cfg.logging, cfg.experiment_name)

  def build_trainer(logger=None, callbacks=None):
    trainer_cfg = cfg.training.trainer
    pl_logger   = logger if logger is not None else build_logger()
    cbs         = list(callbacks) if callbacks is not None else build_callbacks()
    return Trainer(logger=pl_logger, callbacks=cbs, **trainer_cfg)
```

## 備考
- `CFG_DEBUG_MINIMAL` 有効時はモデル構成も `build_scene_model` で軽量化（最小設定）される。
- `overrides` は `hydra` の dotlist 形式に準拠（例: `training.optimizer.lr=1e-4`）。

