# AGENTS.md

This file defines implementation-time conventions for this repository.

## 1) Domain overview (what goes where)
- `WASB` (`src/wasb`): detect 2D ball position on the image.
- `PLCS` (`src/plcs`): infer 3D player position on the court from 2D skeletons.
- `BLCS` (`src/blcs`): infer 3D ball trajectory on the court from 2D ball positions.
- `third_party/`: external modules (e.g., GVHMR for SMPL/pose). Keep vendor code isolated.
- `data/`: datasets and inputs. Document any local-only data.
- `outputs/`: model checkpoints, artifacts, and generated results.

### Typical data flow
1) Video input -> 2D detections (WASB, court keypoints, skeletons)
2) 2D detections -> court-space 3D (PLCS / BLCS)
3) Optional SMPL/mesh from `third_party/GVHMR` -> fused 3D scene

## 2) Data & outputs policy
- Do not commit large data or model artifacts to git.
- If local data is required, add a short note in docs (or README) describing:
  - expected path under `data/`
  - how to obtain it
  - any licensing/usage constraints
- Use placeholder files or small samples when needed for tests.

## 3) Implementation conventions
- Keep changes within the existing repo structure; do not add new top-level directories.
- Favor explicit, typed interfaces for public functions and module boundaries.
- Use clear error messages for invalid inputs and boundary cases; avoid silent failures.
- Docstrings are required for public APIs and non-trivial logic; keep comments concise and purpose-driven.
- Minimize global state and side effects; keep I/O at the edges of modules.

## 4) Training configuration injection pattern
**Standard pattern**: Pass the entire Hydra `DictConfig` directly to modules and access values internally.

### LightningModule pattern
```python
from omegaconf import DictConfig

class MyLightningModule(BaseLightningModule):
    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)
        
        # Access config sections as needed
        model_config = self.config.get("model", {})
        loss_config = self.config.get("loss", {})
        
        # Build components using config values
        self.model = MyModel.from_config(model_config)
        self.loss_fn = MyLoss(loss_config.get("weight", 1.0))
```

### DataModule pattern
```python
from omegaconf import DictConfig

class MyDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}
        
        # Extract data configuration
        data_cfg = self.config.get("data", {})
        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 4)
```

### TrainingRunner pattern
```python
class MyTrainingRunner(BaseTrainingRunner):
    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        # Pass config directly - no decomposition
        return MyDataModule(config)
    
    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        # Pass config directly - no decomposition
        return MyLightningModule(config)
```

### Rationale
- **Consistency**: All training modules (BLCS, PLCS, WASB, court_detection, etc.) follow the same pattern.
- **Flexibility**: Modules can access any config section they need without runner intervention.
- **Maintainability**: Reduces duplication of config extraction logic across runners.
- **Type safety**: DictConfig provides runtime type hints and validation from Hydra.

### Migration from dict decomposition
If you encounter legacy code using `OmegaConf.to_container()` to decompose config into separate dicts:
1. Update the module's `__init__()` to accept `config: DictConfig | None = None`
2. Extract config sections internally using `self.config.get("section", {})`
3. Update the runner to pass `config` directly instead of decomposed dicts
4. Remove `OmegaConf.to_container()` calls from the runner
