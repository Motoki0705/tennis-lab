# Configuration Architecture

This document describes the configuration structure used across different modules in Tennis-Lab.

## Standard Configuration Structure

Most task modules (WASB, PLCS, BLCS, Court Detection) follow a **nested configuration schema** with `config.run.*` at the top level:

```yaml
# Standard structure (e.g., src/wasb/configs/train.yaml)
defaults:
  - run: default        # Runtime configuration
  - model: <model_name>
  - data: default
  - training: default

# config.run contains runtime/environment settings
# (defined in configs/run/default.yaml)
run:
  seed: 42
  gpus: 1
  output_dir: outputs/<task_name>
  dry_run: false
  fast_dev_run: false
  resume: null

# config.model, config.data, config.training are task-specific
model:
  # Model architecture settings

data:
  # Dataset and dataloader settings

training:
  max_epochs: 100
  learning_rate: 1.0e-4
  gradient_clip_val: 1.0
  # ...
```

### Runner Implementation (Standard)

The `BaseTrainingRunner` expects this structure and accesses config as:

```python
# src/base/training/runner.py
def prepare_output_dir(self, config: Any) -> Path:
    return Path(self._ensure_absolute(str(config.run.output_dir)))

def seed_everything(self, config: Any) -> None:
    seed = getattr(config.run, "seed", None)
    # ...

def select_devices(self, config: Any) -> tuple[str, int]:
    gpus = int(getattr(config.run, "gpus", 0))
    # ...
```

---

## MAE Configuration Exception

**MAE (Masked Autoencoder pre-training)** uses a **flattened top-level schema** without `config.run.*`:

```yaml
# MAE structure (src/mae/configs/train.yaml)
defaults:
  - _self_
  - model: base
  - data: cached_batches
  - training: default

# Top-level runtime settings (no 'run' prefix)
seed: 42

# Trainer settings (replaces run.gpus, run.fast_dev_run)
trainer:
  accelerator: auto
  devices: auto
  fast_dev_run: false
  # ... PyTorch Lightning Trainer args

# Task-specific settings
training:
  max_epochs: 400

checkpoint:
  monitor: val/loss
  # ...

data:
  # ...
```

### Why MAE is Different

MAE has unique requirements that justify its exception status:

1. **Hydra-managed output**: MAE relies on Hydra's automatic working directory (`hydra.run.dir`), which is more natural for long pre-training runs. Other tasks explicitly set `config.run.output_dir`.

2. **PyTorch Lightning Trainer alignment**: MAE config exposes `trainer.*` keys that directly map to `pl.Trainer` arguments (e.g., `trainer.devices`, `trainer.accelerator`), reducing indirection.

3. **No test phase**: MAE is a pre-training task without a separate test/evaluation phase, so it doesn't need the full `run.*` lifecycle settings.

4. **Different execution pattern**: MAE uses epoch-based caching callbacks and special data processing, distinct from supervised task workflows.

### MAETrainingRunner Override

`MAETrainingRunner` cleanly adapts the base interface by overriding config access methods:

```python
# src/mae/training/runner.py (lines 25-28, 31-39)
class MAETrainingRunner(BaseTrainingRunner):
    """Training runner for MAE pre-training task.

    MAE config uses top-level keys (seed, trainer, training) instead of
    nested config.run structure. This runner overrides base methods to
    adapt to MAE's config layout.
    """

    def _get_seed(self, config: Any) -> int | None:
        return config.get("seed", 42)

    def _get_gpus(self, config: Any) -> int:
        trainer_cfg = config.get("trainer", {})
        devices = trainer_cfg.get("devices", "auto")
        # ...

    def prepare_output_dir(self, config: Any) -> Path:
        # MAE uses Hydra-managed output directory (cwd)
        return Path.cwd()

    # ... other overrides to translate MAE config to base expectations
```

---

## Exception Policy

### When to Use Standard Structure

Use the **nested `config.run.*` structure** for:
- New supervised learning tasks (detection, classification, regression)
- Tasks requiring explicit output directory management
- Tasks with separate train/validation/test phases
- Fine-tuning or task-specific training

### When to Use MAE-style Structure

Use the **flattened MAE-style structure** for:
- Self-supervised pre-training tasks (if closely related to MAE)
- Tasks explicitly designed around Hydra's working directory management
- Tasks requiring deep PyTorch Lightning Trainer customization

**Guideline**: Default to the **standard structure** unless you have specific requirements that align with MAE's design rationale.

---

## Interoperability

### Using MAE as a Backbone

When using MAE-pretrained weights in downstream tasks (WASB, PLCS, etc.), you only need the checkpoint file:

```yaml
# Standard task config (e.g., WASB)
model:
  backbone_checkpoint: outputs/mae/checkpoints/mae-epoch-399.ckpt
```

The downstream task uses the **standard config structure**; only the checkpoint loading needs to reference MAE's output.

### Config Validation

Third parties can identify the config structure by checking for `config.run`:

```python
# Determine config type
has_run_section = hasattr(config, "run")

if has_run_section:
    # Standard structure (WASB, PLCS, BLCS, Court Detection)
    output_dir = config.run.output_dir
else:
    # MAE-style structure
    output_dir = Path.cwd()  # Hydra-managed
```

**See also**: `docs/examples/config_detection_example.py` for a complete example
showing how to write code that works with both config structures.

---

## Migration Considerations

If future refactoring aims to unify MAE with the standard structure:

1. **Preserve backward compatibility**: Support loading old configs
2. **Update scripts**: Migrate `src/mae/scripts/train.py` to new structure
3. **Document breaking changes**: Provide migration guide for existing MAE users
4. **Test checkpoints**: Ensure pretrained MAE checkpoints work with new configs

**Current recommendation**: Keep MAE as an exception. The override approach is clean, maintainable, and avoids breaking existing workflows.

---

## See Also

- `src/base/training/runner.py` - Base runner implementation
- `src/mae/training/runner.py` - MAE exception implementation
- `src/wasb/training/runner.py` - Standard structure example
- `AGENTS.md` - Overall implementation conventions
