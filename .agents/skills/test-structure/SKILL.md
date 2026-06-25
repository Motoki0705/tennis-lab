---
name: test-structure
description: Always refer to this skill when writing, placing, or organizing tests in this repository, which is a Hydra config + PyTorch Lightning based tennis video analysis project with the following structure: src/tasks/{ball_detection,blcs,court_detection,plcs}, src/tennis_scene, and src/utils. Use this skill for all decisions such as where to create a new test_*.py file, where the test for a given implementation file should live, how to structure the tests/ directory, and how to handle tests for ML models, lightning_modules, and rendering. This skill should be triggered for requests such as "write tests," "where should the test for this module go?", "I want to organize tests," or "test structure," even when this skill is not explicitly specified.
---

# Test Structure (tests/ ↔ src/ Mirroring Policy)

This document serves as the decision criteria for placing and creating tests in this project. We **do not mechanicaly create `test_*.py` for every single module**. The core principle is to **mirror only the unit tests to src/ and fill them starting from high-value code**.

## Core Principle: Separate into Three Axes

```
tests/
├── conftest.py        # Global shared fixtures (seed fixing, tmp_path, dummy tensor generation, etc.)
├── unit/              # ← Only this directory mirrors src/
├── integration/       # Feature/Task level. Do not split on a per-file basis
└── e2e/               # Verifies entry points equivalent to scripts/ at the command level
```

- **unit/** … Pure logic with clear inputs and outputs. Reflects the structure of `src/` as-is.
- **integration/** … Interaction between multiple modules, smoke tests running 1-step training/inference with minimal data. Does not correspond 1:1 with a single source file.
- **e2e/** … End-to-end pipeline or script execution. Verifies entry points such as `tennis_scene/scripts/run_pipeline.py`.

## Mirroring Rules (Limited to unit/)

1. Since `src/` is the layout root, drop it **one layer down** on the test side. `src/utils/geometry/angles.py` → `tests/unit/utils/geometry/test_angles.py`.
2. Prefix file names with **`test_`**.
3. Maintain the directory hierarchy exactly as it is (e.g., `tasks/<task>/data/components/web/parser/` remains unchanged).

**Examples (Always follow this format):**

```
src/utils/geometry/angles.py
→ tests/unit/utils/geometry/test_angles.py

src/utils/projection/camera_projector.py
→ tests/unit/utils/projection/test_camera_projector.py

src/tasks/ball_detection/data/components/web/parser/kaggle.py
→ tests/unit/tasks/ball_detection/data/components/web/parser/test_kaggle.py

src/tasks/plcs/data/targets.py
→ tests/unit/tasks/plcs/data/test_targets.py
```

## Exclusions from Mirroring

The following are **not included** in the Python `unit/` mirror:

- **`configs/**/*.yaml` (Hydra config)** — While abundant, these are not implementation code. If testing is needed, it belongs to a different axis, such as "Can the config be composed and validated?" and should be handled like `tests/integration/<task>/test_config_compose.py`. Do not mirror.
- **`src/tasks/blcs/generate_dataset/webui/` (Next.js / TypeScript)** — Part of a different ecosystem. Do not place in Python's `tests/`. If testing is required, place a separate JS testing infrastructure within the webui directory.
- **`__init__.py`** — Do not test if it is empty.

## Priority of Implementation (Ordered by High ROI)

Do not mechanically fill tests from end to end; write them in this order, starting from where they add the most value:

1. **`utils/`** is the top priority. `geometry/` (angles, court_pose, keypoints, matrices), `projection/camera_projector`, `data/` (heatmaps, splits), `video/` (windows, batching, transforms), and `tensor_utils` consist of pure logic with explicit inputs/outputs, offering the highest cost-performance ratio. If these break, it ripples through all tasks, so solidify them first.
2. **`tasks/base/`** comes next. Since it is an abstraction that all tasks (`ball_detection`, `blcs`, `court_detection`, `plcs`) depend on, treat it **one level more thoroughly** than individual tasks.
3. Pure logic for each task (`ball_detection` / `blcs` / `court_detection` / `plcs`). Since their internal structures (`data` / `models` / `training` / `inference` / `visualization` / `scripts`) are largely identical, the test side will follow a recurring pattern. Specific examples:
   - Parser systems (`ball_detection/.../parser/{kaggle,roboflow,racketvision}.py`) — Clear inputs/outputs, making them easy to write.
   - `evaluation/` (metrics, contracts, adapters)
   - `data/targets.py`, `generate_dataset/sampling/motion_sampler.py`

## ML-Specific Judgments: What to Unit Test vs. What to Smoke Test

**Do not create `test_*.py` for every module.** The location changes depending on the nature of the code:

- **Pure functions / Shape transformations** → `unit/`. For example, verifying a model's "input shape → output shape" can easily be unit tested on CPU. These can go into `unit/tasks/<task>/models/test_*.py`.
- **`lightning_module.py`'s `training_step`, actual model forwards (heavy computation), and `rendering/` drawing** → **Not suited for unit tests**. Isolate these into `integration/` as a "smoke test running 1 step on minimal data." Mark them with `@pytest.mark.gpu` or `@pytest.mark.slow` so they can be selectively run in CI.
- **`scripts/` (entry points for each task)** — Rather than individual mirroring, delegate these to `e2e/` for command-line level verification.

**Decision Flow (When given an implementation file):**
1. Is it a config YAML or webui TS? → Excluded from mirroring.
2. Is it a pure function / lightweight shape transformation? → `tests/unit/<relative_src_path>/test_<name>.py`.
3. Is it heavy training, inference, or rendering? → `tests/integration/<task>/test_<feature>_smoke.py` (with markers).
4. Is it a script / entire pipeline? → `tests/e2e/<area>/test_<name>.py`.

## Placement of conftest.py

- `tests/conftest.py` … Global shared fixtures such as seed fixing, utilizing `tmp_path`, and dummy tensor generation.
- Directly under each task (e.g., `tests/unit/tasks/ball_detection/conftest.py`) … Consolidate task-specific fixtures like dummy datasets here. Scoping them by task keeps things organized.

## Expected Structure (Reference Skeleton)


```

tests/
├── conftest.py
├── unit/
│   ├── utils/
│   │   ├── geometry/{test_angles,test_court_pose,test_keypoints,test_matrices}.py
│   │   ├── projection/test_camera_projector.py
│   │   ├── video/{test_windows,test_batching,test_transforms}.py
│   │   ├── data/{test_heatmaps,test_splits}.py
│   │   └── test_tensor_utils.py
│   └── tasks/
│       ├── base/data/test_chunk_manager.py ...
│       ├── ball_detection/
│       │   ├── conftest.py
│       │   ├── data/components/web/parser/{test_kaggle,test_roboflow,test_racketvision}.py
│       │   └── evaluation/{test_metrics,test_contracts,test_adapters}.py
│       ├── blcs/ ...
│       ├── court_detection/ ...
│       └── plcs/
│           ├── data/test_targets.py
│           └── generate_dataset/sampling/test_motion_sampler.py
├── integration/
│   ├── tasks/ball_detection/test_train_smoke.py   # 1-step training with minimal data
│   └── tennis_scene/{test_dependency_graph,test_orchestrator}.py
└── e2e/
└── tennis_scene/test_run_pipeline.py

```

## Summary in One Sentence

While `unit/` mirrors `src/`, only populate it with code that is truly worth testing; configs and webui are excluded, `utils/` and `base/` are the starting points, heavy forwards/training_steps/rendering are isolated to `integration/` smoke tests with markers, and scripts are offloaded to `e2e/`.
