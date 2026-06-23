# `src/utils` — shared utilities

**Before writing a helper function in a task module, check whether it already
lives here.** This package is the single home for cross-cutting, domain-agnostic
logic (path/device/seed handling, IO, tensor/geometry math, heatmaps, rendering,
schemas, video). Re-implementing any of the below inside `src/tasks/...` or
`src/tennis_scene/...` re-introduces the WET duplication that this package
exists to prevent — see `docs/refactoring/utils-duplication-survey.md` for the
history.

Rule of thumb: if a function has **no dependency on a specific task's domain
types**, it belongs here (or should be promoted here), not in the task module.

## "I need to…" quick reference

| Need | Use | Import |
|------|-----|--------|
| Resolve a repo-relative path | `resolve_project_path()`, `PROJECT_ROOT` | `from src.utils.paths import resolve_project_path, PROJECT_ROOT` |
| Pick a torch device (`"auto"`/fallback) | `resolve_device()` | `from src.utils.device import resolve_device` |
| Lightning `(accelerator, devices)` | `select_accelerator()` | `from src.utils.device import select_accelerator` |
| Seed RNGs in a script | `seed_everything()` | `from src.utils.seeding import seed_everything` |
| Per-sample dataloader RNG | `make_sample_rng()` | `from src.utils.seeding import make_sample_rng` |
| `mkdir(parents=True, exist_ok=True)` | `ensure_dir()` | `from src.utils.io import ensure_dir` |
| Write/read JSON (creates dirs) | `save_json()` / `load_json()` | `from src.utils.io import save_json, load_json` |
| Clone a `dict[str, Tensor]` sample | `clone_tensor_dict()` | `from src.utils.tensor_utils import clone_tensor_dict` |
| Tensor → numpy (bf16-safe) | `to_numpy()` | `from src.utils.tensor_utils import to_numpy` |
| Masked mean / padding mask | `masked_mean()`, `normalize_padding_mask()` | `from src.utils.tensor_utils import masked_mean, normalize_padding_mask` |
| Gaussian heatmaps / decode | `generate_gaussian_heatmaps()`, `heatmaps_to_argmax/soft_argmax/peaks/pixel_coords()` | `from src.utils.data.heatmaps import ...` |
| ImageNet normalize/denormalize | `normalize_tensor_images_imagenet()` / `denormalize_tensor_images_imagenet()` / `normalize_frames_imagenet()` | `from src.utils.data.augmentation import ...` |
| Parse a config range | `parse_float_range()` / `parse_int_range()` | `from src.utils.data.augmentation import ...` |
| Load a scene directory | `load_scene_payload()` | `from src.utils.data.scene_io import load_scene_payload` |
| Wrapped angle / angular error | `angular_error()`, `wrapped_angle_diff()`, `normalize_vector()`, `signed_angle_around_axis()` | `from src.utils.geometry.angles import ...` |
| COCO-17 joint/torsion/twist/bone | `compute_joint_angles/torsion_angles/torso_twist/bone_lengths()` | `from src.utils.geometry.skeleton import ...` |
| Court ↔ world pose | `canonical_pose_to_world_pose()`, `world_pose_to_canonical_pose()` | `from src.utils.geometry.court_pose import ...` |
| Rotation matrices / SMPL transform | `rotation_matrix_y/z()`, `axis_angle_to_rotation_matrix()`, `apply_plcs_transform[_batch]()` | `from src.utils.geometry.matrices import ...` |
| Pixel ↔ normalized keypoints | `normalize_keypoints()` / `denormalize_keypoints()` | `from src.utils.geometry.keypoints import ...` |
| Camera projection | `CameraProjector`, `project_points()`, `make_look_at_camera()` | `from src.utils.projection import ...` |
| Render court / skeleton / ball | `CourtRenderer`, `SkeletonRenderer`, `BallRenderer` | `from src.utils.rendering import ...` |
| Court / player schema constants | court & COCO/SMPL keypoint definitions | `from src.utils.schema.court import ...` / `from src.utils.schema.player import ...` |
| Read/stream video frames | `probe_video_info()`, `OpenCVVideoFrameReader`, `iter_temporal_windows/batches()` | `from src.utils.video import ...` |
| Transformer / MoE / RoPE blocks | `TransformerBlock`, `MoELayer`, RoPE helpers, `MLPHead` | `from src.utils.models import ...` |

## Modules

### Top-level helpers
- **`paths.py`** — `PROJECT_ROOT` (repo root) and `resolve_project_path()`. Always
  resolve repo-relative paths through this instead of a hand-counted
  `Path(__file__).parents[N]`. (Note: Hydra's `to_absolute_path` resolves against
  the *launch* cwd, not the repo root — keep using it where that is intended.)
- **`device.py`** — `resolve_device()` (handles `"auto"` + CUDA fallback) and
  `select_accelerator()` for Lightning.
- **`seeding.py`** — `seed_everything()` (Python/NumPy/Torch) and worker-aware
  `make_sample_rng()`. Training entry points needing full Lightning determinism
  should still use `lightning.pytorch.seed_everything`.
- **`io.py`** — `ensure_dir()`, `save_json()`, `load_json()`.
- **`tensor_utils.py`** — `clone_tensor_dict()`, `to_numpy()` (detaches, moves to
  CPU, upcasts bf16/fp16), `masked_mean()`, `normalize_padding_mask()`.

### `data/`
- **`heatmaps.py`** — `generate_gaussian_heatmap[s]()`, `heatmaps_to_argmax()`,
  `heatmaps_to_soft_argmax()`, `heatmaps_to_peaks()`, `heatmaps_to_pixel_coords()`.
- **`augmentation.py`** — UV/visibility augmentation primitives, ImageNet
  `normalize_/denormalize_tensor_images_imagenet()`, `normalize_frames_imagenet()`
  (numpy HWC), `parse_float_range()` / `parse_int_range()`.
- **`scene_io.py`** — `load_scene_payload()` for the `*.npy` + `scalars.json` +
  `meta.json` scene-directory layout.

### `geometry/`
- **`angles.py`** (torch) — `angular_error`, `wrapped_angle_diff`,
  `normalize_vector`, `signed_angle_around_axis`.
- **`skeleton.py`** (torch) — COCO-17 `compute_joint_angles`,
  `compute_torsion_angles`, `compute_torso_twist`, `compute_bone_lengths`.
- **`court_pose.py`** (torch) — `court_position_to_world_translation`,
  `canonical_pose_to_world_pose`, `world_pose_to_canonical_pose`.
- **`matrices.py`** (numpy) — `rotation_matrix_y` (scalar), `rotation_matrix_z`
  (batched), `axis_angle_to_rotation_matrix`, `apply_plcs_transform[_batch]`.
- **`keypoints.py`** (numpy) — `normalize_keypoints`, `denormalize_keypoints`.

### Other packages (pre-existing)
- **`models/`** — DeepSeek-style Transformer blocks, MoE, RoPE, attention,
  `MLPHead`, domain token embeddings, DINOv3 backbone loading.
- **`projection/`** — pinhole `Camera`, `CameraProjector`, `make_look_at_camera`,
  `project_points`.
- **`rendering/`** — `CourtRenderer`, `SkeletonRenderer`, `BallRenderer`.
- **`schema/`** — canonical court geometry and COCO-17 / SMPL-H pose schemas
  (names, indices, skeletons, angle triplets, coordinate scales).
- **`video/`** — OpenCV streaming: `probe_video_info`, `read_video_frame`,
  `OpenCVVideoFrameReader`, `iter_temporal_windows/batches`, `PrefetchIterator`,
  `BgrToTensorTransform`, `normalize_tensor_imagenet`.

## Adding a new utility

1. **Search first.** Grep this README's table and the relevant module. The most
   common duplication is re-deriving something already here.
2. **Put it next to the closest existing module** (tensor math →
   `tensor_utils.py`, heatmap decode → `data/heatmaps.py`, angle math →
   `geometry/angles.py`). Create a new module only when nothing fits.
3. **Keep it domain-agnostic.** If it needs a task's types/config, it probably
   belongs in that task — or the task-specific part should be a thin wrapper over
   a generic core that lives here.
4. **Export it** from the module's `__all__` (and the package `__init__`/this
   README if broadly useful), and add a unit test in
   `tests/test_utils_extraction.py`.
5. **Migrate, don't fork.** When you find an existing local copy, replace it with
   an import (delegate or re-export to preserve public import paths) rather than
   leaving both.
