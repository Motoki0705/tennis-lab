# Configuration Quick Reference

Quick guide for understanding and using configuration files in Tennis-Lab.

## Which Config Structure Should I Use?

### Standard Tasks (WASB, PLCS, BLCS, Court Detection)

```yaml
# Example: src/wasb/configs/train.yaml
defaults:
  - run: default      # ← Contains runtime settings
  - model: hrcnet
  - data: default
  - training: default

# Access in code: config.run.seed, config.run.output_dir
```

**Files to reference**:
- `src/court_detection/configs/train.yaml` (simplest example)
- `src/wasb/configs/train.yaml` (more complex)

### MAE Pre-training (Exception)

```yaml
# Example: src/mae/configs/train.yaml
defaults:
  - _self_
  - model: base
  - data: cached_batches
  - training: default

seed: 42              # ← Top-level, not config.run.seed
trainer:              # ← Replaces config.run.gpus, etc.
  devices: auto
  accelerator: auto

# Access in code: config.seed, config.trainer.devices
```

**Why different?** MAE uses Hydra working directory management and directly exposes PyTorch Lightning Trainer parameters.

---

## Common Config Access Patterns

### Standard Structure
```python
# In a standard task runner (WASB, PLCS, etc.)
output_dir = config.run.output_dir
seed = config.run.seed
gpus = config.run.gpus
max_epochs = config.training.max_epochs
batch_size = config.data.batch_size
```

### MAE Structure
```python
# In MAE runner
output_dir = Path.cwd()  # Hydra-managed
seed = config.seed
devices = config.trainer.devices
max_epochs = config.training.max_epochs
batch_size = config.data.batch_size
```

---

## Creating a New Task

### Option 1: Standard Structure (Recommended)

1. Copy config structure from existing task:
   ```bash
   cp -r src/court_detection/configs src/my_task/configs
   ```

2. Create runner extending `BaseTrainingRunner`:
   ```python
   from src.base.training.runner import BaseTrainingRunner
   
   class MyTaskRunner(BaseTrainingRunner):
       def build_datamodule(self, config):
           # ...
       
       def build_lightning_module(self, config, datamodule, **kwargs):
           # ...
   ```

3. Config will have `config.run.*` automatically.

### Option 2: MAE-style Structure (Only if needed)

**Use only if**:
- You need Hydra working directory management
- Your task is self-supervised pre-training similar to MAE
- You need extensive PyTorch Lightning Trainer customization

**Steps**:
1. Copy `src/mae/configs/train.yaml`
2. Override all config access methods in your runner (see `MAETrainingRunner`)
3. Document why your task uses this exception structure

---

## Config Organization

```
src/<task>/configs/
├── train.yaml          # Main config entry point
├── run/
│   └── default.yaml    # Runtime settings (standard tasks only)
├── model/
│   ├── model_a.yaml
│   └── model_b.yaml
├── data/
│   └── default.yaml
└── training/
    └── default.yaml
```

**MAE exception**: No `run/` directory; runtime settings in top-level `train.yaml`.

---

## Troubleshooting

### Error: `'DictConfig' object has no attribute 'run'`

**Cause**: You're using a MAE-style config with code expecting standard structure.

**Solution**: 
- If writing a standard task, add `run: default` to defaults and create `configs/run/default.yaml`
- If extending MAE, override config access methods like `MAETrainingRunner` does

### Error: Can't find output directory

**Standard tasks**: Check `config.run.output_dir` is set
**MAE**: Outputs go to Hydra working directory (usually `outputs/<timestamp>`)

---

## Further Reading

- **docs/config_architecture.md** - Full policy and rationale
- **AGENTS.md** - Overall implementation conventions
- **src/base/training/runner.py** - Base runner expecting standard structure
- **src/mae/training/runner.py** - MAE exception implementation
