# Common Dataset Utilities

This module provides shared infrastructure for dataset generation and augmentation across PLCS, BLCS, and other modules.

## Architecture

The shared dataset utilities follow the DRY (Don't Repeat Yourself) principle, extracting common functionality from individual modules into reusable components.

```
src/utils/dataset/
├── __init__.py
├── writer.py          # BaseDatasetWriter - NPZ dataset generation
├── augmentation.py    # Common augmentation functions
└── README.md          # This file
```

## Components

### BaseDatasetWriter (`writer.py`)

Abstract base class for NPZ-based dataset writers (PLCS/BLCS).

**Purpose**: Eliminate code duplication between PLCS and BLCS dataset writers (~200 lines saved).

**Common Functionality**:
- Directory management (`output_dir`, `scenes_dir`)
- Train/val/test split generation (`save_split_info()`)
- Metadata JSON generation (`save_meta_json()`)
- Dataset statistics tracking (`save_dataset_info()`)

**Usage**:
```python
from src.utils.dataset.writer import BaseDatasetWriter
from pathlib import Path

class MyDatasetWriter(BaseDatasetWriter):
    def save_scene(self, scene_data) -> Path:
        """Implement module-specific scene saving logic."""
        filename = f"{scene_data.id}.npz"
        filepath = self.scenes_dir / filename

        # Save NPZ with module-specific data
        np.savez_compressed(filepath, ...)

        # Track for meta.json
        self.scene_records.append({
            "file": filename,
            # ...other metadata
        })

        return filepath

# Use the writer
writer = MyDatasetWriter("outputs/my_dataset")
writer.save_scene(scene)
writer.save_split_info(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
writer.save_meta_json(config={})
```

**Methods Inherited**:
- `__init__(output_dir)`: Initialize directory structure
- `save_split_info(train_ratio, val_ratio, test_ratio, seed)`: Generate train/val/test splits
- `save_meta_json(config)`: Save dataset metadata with statistics
- `save_dataset_info(stats)`: Save additional statistics

**Abstract Methods** (must implement):
- `save_scene(scene_data) -> Path`: Save a single scene to NPZ

### Augmentation Functions (`augmentation.py`)

Common data augmentation utilities for keypoints and visibility masks.

**Purpose**: Share augmentation logic across PLCS/BLCS datasets (~60 lines saved).

**Functions**:

#### `add_gaussian_noise(tensor, noise_std) -> Tensor`
Add Gaussian noise to any tensor.

```python
from src.utils.dataset.augmentation import add_gaussian_noise

# Add noise to keypoints
noisy_kp = add_gaussian_noise(keypoints, noise_std=0.01)
```

#### `random_visibility_dropout(visibility, drop_prob) -> Tensor`
Randomly drop visibility flags for augmentation.

```python
from src.utils.dataset.augmentation import random_visibility_dropout

# Randomly hide some keypoints
augmented_vis = random_visibility_dropout(visibility, drop_prob=0.05)
```

#### `augment_keypoints(keypoints, visibility, noise_std, visibility_drop_prob) -> tuple[Tensor, Tensor]`
Combined convenience function for keypoint augmentation.

```python
from src.utils.dataset.augmentation import augment_keypoints

# Apply both noise and dropout
aug_kp, aug_vis = augment_keypoints(
    keypoints, visibility,
    noise_std=0.01,
    visibility_drop_prob=0.05
)
```

**Tensor Shape Flexibility**:
All functions work with arbitrary tensor shapes:
- Frame-level: `(N, 2)` keypoints, `(N,)` visibility
- Sequence-level: `(T, N, 2)` keypoints, `(T, N)` visibility
- Batched: `(B, T, N, 2)` keypoints, `(B, T, N)` visibility

## Design Principles

### 1. Single Responsibility
Each component has one clear purpose:
- `BaseDatasetWriter`: Dataset file management
- `augmentation.py`: Data augmentation

### 2. Don't Repeat Yourself (DRY)
Common code is extracted to base classes/functions:
- ✅ Before: PLCS/BLCS each had ~200 lines of identical code
- ✅ After: Shared in `BaseDatasetWriter` (~180 lines total)

### 3. Open/Closed Principle
Base classes are open for extension, closed for modification:
- Extend `BaseDatasetWriter` for new modules
- Don't modify base class for module-specific logic

### 4. Type Safety
All functions have complete type hints:
- mypy strict mode enabled (`disallow_untyped_defs = true`)
- Generic types support flexible shapes

## Integration with Existing Modules

### PLCS Integration

**Before**:
```python
class PLCSDatasetWriter:
    def __init__(self, output_dir):
        # 20 lines of boilerplate

    def save_split_info(...):
        # 50 lines duplicated from BLCS

    def save_meta_json(...):
        # 50 lines duplicated from BLCS
```

**After**:
```python
from src.utils.dataset.writer import BaseDatasetWriter

class PLCSDatasetWriter(BaseDatasetWriter):
    # Only implement save_scene()
    # Inherit all common functionality
```

### BLCS Integration

Same pattern as PLCS - inherits from `BaseDatasetWriter`.

### WASB Integration

WASB uses external annotations (CSV files), so it doesn't use `BaseDatasetWriter`. However, it can use augmentation functions if needed.

## Future Extensions

### Adding New Modules

To add a new dataset module:

1. **Create types.py** with TypedDict schemas:
```python
# src/newmodule/data/types.py
class NewModuleBatch(TypedDict):
    input: torch.Tensor
    target: torch.Tensor
```

2. **Create dataset writer**:
```python
# src/newmodule/generate_dataset/io/dataset_io.py
from src.utils.dataset.writer import BaseDatasetWriter

class NewModuleDatasetWriter(BaseDatasetWriter):
    def save_scene(self, scene) -> Path:
        # Implement NPZ saving logic
        pass
```

3. **Use augmentation functions** in dataset classes:
```python
# src/newmodule/data/dataset.py
from src.utils.dataset.augmentation import augment_keypoints

class NewModuleDataset(Dataset):
    def __getitem__(self, idx):
        kp, vis = augment_keypoints(...)
        return {"input": kp, "target": vis}
```

### Adding New Augmentations

To add new augmentation functions:

```python
# src/utils/dataset/augmentation.py

def new_augmentation(data: Tensor, param: float) -> Tensor:
    """New augmentation with clear docstring.

    Args:
        data: Input tensor of shape (...).
        param: Augmentation parameter.

    Returns:
        Augmented tensor of same shape.
    """
    # Implementation
    return augmented_data
```

**Best Practices**:
- Keep functions pure (no side effects)
- Support arbitrary tensor shapes
- Add comprehensive type hints
- Write clear docstrings with shape annotations

## Migration Guide

### For Existing Code

If you have existing dataset code and want to use base module:

1. **Identify common patterns** (e.g., repeated augmentation logic)
2. **Extract to base module** if used by 2+ modules
3. **Update imports** in existing files
4. **Run tests** to ensure no regressions
5. **Remove duplicate code**

### Backward Compatibility

All changes maintain backward compatibility:
- Existing dataset classes work unchanged
- New modules can adopt incrementally
- No breaking changes to public APIs

## Testing

Base module components are tested indirectly through:
- PLCS dataset generation and training
- BLCS dataset generation and training
- Integration tests in each module

For direct testing:
```bash
# Run type checking
uv run mypy src/utils/dataset

# Run linting
uv run ruff check src/utils/dataset
```

## Maintenance

### Adding Features

1. Discuss with team if feature is truly common (2+ modules)
2. Implement in base module with tests
3. Update documentation
4. Migrate existing modules incrementally

### Removing Features

1. Check all usages across modules
2. Deprecate first (if public API)
3. Remove after migration period
4. Update documentation

## References

- PLCS dataset: `src/tasks/plcs/data/`
- BLCS dataset: `src/tasks/blcs/data/`
- WASB dataset: `src/tasks/wasb/data/`
- Type definitions: `src/{module}/data/types.py`
