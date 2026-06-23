# Shared-utility duplication survey

**Date:** 2026-06-23
**Scope:** `src/` (`src/tasks/`, `src/tennis_scene/`, `src/utils/`)
**Trigger:** `_resolve_project_path` in `src/tasks/court_detection/models/dinov3_detr.py:36` — a private path-resolution helper that looks generic enough to belong in `src/utils/`, raising the question of how much other generic logic is re-implemented locally across tasks.

> This is a **read-only census and recommendation report**. No code is moved here. It is meant to drive follow-up extraction PRs, one cluster at a time.

---

## TL;DR

The codebase already has a healthy shared package (`src/utils/`: `data/`, `models/`, `projection/`, `rendering/`, `schema/`, `video/`, `tensor_utils.py`). But a layer of **small, cross-cutting helpers** has not been promoted into it and is now re-implemented locally — sometimes byte-for-byte — across `ball_detection`, `court_detection`, `blcs`, `plcs`, `base`, and `tennis_scene`.

The highest-value gaps, roughly in priority order:

| # | Cluster | Evidence | Proposed home |
|---|---------|----------|---------------|
| 1 | **Project-root / path resolution** (the motivating example) | `_resolve_project_path` + ~85 `Path(__file__).parents[...]` / `to_absolute_path` sites | `src/utils/paths.py` (new) |
| 2 | **Device resolution** | canonical in `base/inference/predictor.py:93`; **3 independent copies** in plcs scripts (signature drift: returns `str` vs `torch.device`) | `src/utils/device.py` (new) |
| 3 | **Deterministic seeding** | `_seed_everything` **verbatim ×2** (blcs + plcs `generate_dataset.py`) + `base` runner method | `src/utils/seeding.py` (new) |
| 4 | **Duplicate already inside `src/utils`** | `normalize_tensor_imagenet` (video) vs `normalize_tensor_images_imagenet` (data) are **byte-identical** | collapse to one |
| 5 | **`_clone_sample`** | **byte-identical** in `blcs` and `plcs` augmentation | `src/utils/tensor_utils.py` |
| 6 | **Rotation / angle geometry** | `angular_error`, wrapped-angle-diff, rotation matrices, skeleton angles scattered across plcs + tennis_scene + analysis scripts | `src/utils/geometry/` (new) |
| 7 | **JSON-result save boilerplate** | `mkdir(parents=True) + json.dump(indent=2)` copy-pasted in **6** pipeline `Result.save()` methods | `src/utils/io.py` (new) |
| 8 | **Heatmap → pixel-coord denorm** | `heatmaps_to_argmax` + manual `*(W-1)` scaling re-done in 4+ sites | extend `src/utils/data/heatmaps.py` |
| 9 | **Tensor→numpy / ensure-dir idioms** | `_to_numpy` ×3, `.detach().cpu().numpy()` inline; `mkdir(parents=True, exist_ok=True)` ×54 | `tensor_utils.py` / `io.py` |

---

## Method

Three Sonnet sub-agents surveyed the **main working tree** in parallel, then key claims were spot-checked directly:

1. **Pattern census** — grep-driven sweep of all of `src/` for cross-cutting idioms (path resolution, seeding, device, config/checkpoint loading, filesystem IO, tensor conversion, logging).
2. **Task-local helpers** — deep read of `src/tasks/{ball_detection,base,blcs,court_detection,plcs}` for module-level `_helper` functions that are domain-agnostic.
3. **Utils inventory + scene/tools** — catalogued the existing `src/utils/` public surface and surveyed `src/tennis_scene/` for overlaps and promotion candidates.

**Excluded:** `third_party/` (vendored), `__pycache__`, `.venv`. The top-level `tools/` directory is **untracked local tooling** (colab CLI per project notes) — not committed repo code, so it is out of scope for a shared-utility survey.

Verified directly (byte-for-byte or via grep counts): `_clone_sample` duplication, the two ImageNet-normalize functions, `_resolve_device` copies, `_seed_everything` copies, and the path-resolution / ensure-dir / `to_numpy` occurrence counts.

---

## The motivating example

```python
# src/tasks/court_detection/models/dinov3_detr.py:29,36
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    return resolved.resolve()
```

This "resolve a relative path against the repo root" need recurs throughout the codebase, expressed inconsistently:

- `src/tennis_scene/pipeline/components/gvhmr.py:143,334` — `Path(__file__).parents[3]`
- `src/tennis_scene/pipeline/orchestrator.py:309` — `cwd=str(Path(__file__).parents[3])`
- `src/tasks/blcs/generate_dataset/config.py:118` — `Path(__file__).resolve().parents[1] / "configs"`
- `src/tasks/base/training/runner.py:387` — `_ensure_absolute` wrapping Hydra's `to_absolute_path`
- ~85 total sites across `src/` use some form of `Path(__file__).parents[...]` or `to_absolute_path`.

The brittle part is the hard-coded `parents[N]` depth: each new file must recount its distance to the repo root. A single `PROJECT_ROOT` constant + `resolve_project_path()` in `src/utils/paths.py` removes that footgun.

---

## A. Cross-cutting generic patterns re-implemented locally

| Pattern | Occurrences (verified) | Reusable? | Suggested home |
|---|---|---|---|
| Project-root / path resolution | `_resolve_project_path` (court_detection) + `_ensure_absolute` (base runner:387) + ~85 inline `parents[...]`/`to_absolute_path` sites | **yes** | `src/utils/paths.py` (new) |
| Device resolution | `base/inference/predictor.py:93` `_resolve_device` (canonical) + plcs copies: `scripts/analysis/visualize_rotation_error_samples.py:46`, `scripts/generate_dataset.py:41`, `scripts/analysis/analyze_loss_dominance.py:340` | **yes** | `src/utils/device.py` (new) |
| Deterministic seeding | `plcs/scripts/generate_dataset.py:34` & `blcs/scripts/generate_dataset.py:46` (`_seed_everything`, **identical**) + `base/training/runner.py:173` (`seed_everything` method) | **yes** | `src/utils/seeding.py` (new) |
| Tensor → numpy | `base/training/lightning_module.py:150` (`_to_numpy`), `plcs/visualization/adapters/render_inputs.py:23` (`_to_numpy`), `court_detection/generate_dataset/annotation_session.py:436` (`to_numpy`), plus inline `.detach().cpu().numpy()` | **yes** | `src/utils/tensor_utils.py` (exists) |
| Ensure-dir (`mkdir(parents=True, exist_ok=True)` / `makedirs`) | **54 inline sites**, no wrapper | marginal | `src/utils/io.py` (new) `ensure_dir()` |
| JSON read/write helpers | private json save/load re-implemented across several modules; 6 pipeline `Result.save()` methods share the same `mkdir + json.dump(indent=2)` body | **yes** | `src/utils/io.py` (new) |
| Config loading (`OmegaConf.load` / `yaml.safe_load`) | present but contextually distinct per call-site | **no** | leave as-is |
| Checkpoint / state_dict munging | mostly localized; `src/utils` already has a `models/dino_backbone` path; `dinov3_detr` does inline `torch.load` + key-unwrap | low | leave / extend `src/utils/models` |
| Logging setup | 3 `basicConfig` calls (all entrypoints) + ~25 idiomatic `getLogger(__name__)` | **no** | leave as-is (idiomatic) |

**Top duplication offenders (by count/severity):**
1. Inline `mkdir(parents=True, exist_ok=True)` — 54 sites (idiom, no helper).
2. Path/project-root resolution — ~85 sites, inconsistent depth handling.
3. `_resolve_device` — 1 canonical + 3 drifting copies.
4. `_seed_everything` — 2 verbatim copies.
5. `_to_numpy` / tensor→numpy — 3 helper copies + inline.

---

## B. Duplication already inside `src/utils/`

`src/utils/` itself contains a verbatim duplicate. Both functions are byte-identical (same body, only the name and default-arg style differ):

- `src/utils/video/transforms.py:35` — `normalize_tensor_imagenet(images, *, mean, std)`
- `src/utils/data/augmentation.py:59` — `normalize_tensor_images_imagenet(images, *, mean, std)`

**Recommendation:** keep one canonical implementation and have the other delegate to it (or re-export), so the two consumer trees (`video/` streaming and `data/` augmentation) don't drift.

---

## C. Task-local helpers worth promoting

Grouped by subpackage. "Dup?" = whether an equivalent already exists elsewhere.

### `ball_detection`
| file:line | helper | what it does | dup? | suggested home |
|---|---|---|---|---|
| `data/augmentation.py:44` | `denormalize_tensor_images_imagenet` | invert ImageNet norm on `(...,3,H,W)` | symmetric to `normalize_*` already in utils | `src/utils/data/augmentation.py` |
| `data/augmentation.py:32` | `normalize_frames_imagenet` | ImageNet norm for a list of HWC numpy frames | numpy sibling of utils tensor version | `src/utils/data/augmentation.py` |
| `data/augmentation.py:574,586` | `_parse_ratio_range` / `_parse_int_range` | parse/validate 2-element ranges from config | **yes** — `utils...parse_float_range` exists; `base...:291` `_validate_range` is another copy | extend `src/utils/data/augmentation.py` |
| `data/augmentation.py:750` | `make_sample_rng` | worker-aware deterministic per-sample RNG | generic dataloader concern | `src/utils/data/` (seeding) |
| `data/augmentation.py:63` | `_resolve_border_mode` | config string → `cv2` border constant | reusable for any cv2 augmentation | `src/utils/data/augmentation.py` |

### `base`
| file:line | helper | what it does | dup? | suggested home |
|---|---|---|---|---|
| `data/scene_dataset.py:27` | `_load_scene_payload` | load `.npy` + `scalars.json` + `meta.json` from a scene dir | the project-wide scene-dataset format, shared by blcs/plcs | `src/utils/data/scene_io.py` (new) |
| `data/scene_dataset.py:291` | `_validate_range` | validate `(lo, hi)` ordered/positive | **yes** — see ball_detection copies | shared range-validation util |
| `training/runner.py:322` | `select_devices` | config → `("gpu", N)` / `("cpu", 1)` | same intent as `_resolve_device` | `src/utils/device.py` |

### `court_detection`
| file:line | helper | what it does | dup? | suggested home |
|---|---|---|---|---|
| `models/dinov3_detr.py:36` | `_resolve_project_path` | **the motivating example** | yes (see §A) | `src/utils/paths.py` |
| `training/metrics.py:123` | `_heatmaps_to_pixel_coords` | argmax heatmap → denormalized pixel coords | **yes** — ball_detection re-scales manually in `inference/predictor.py:114`, `scripts/eval.py:326`, `visualization/adapters/render_inputs.py` | extend `src/utils/data/heatmaps.py` |
| `inference/preprocess.py:16` | `preprocess_court_image` | short-side resize→snap-to-8→norm→batch→device | **yes** — `data/augmentation.py:119,297` re-do the resize math | `src/utils/data` or `video/transforms` |
| `data/augmentation.py:35,40` | `_pil_to_tensor_image` / `_mask_pil_to_tensor` | standard PIL→tensor conversions | shared by 3 court datasets | `src/utils/data/augmentation.py` |

### `blcs` / `plcs`
| file:line | helper | what it does | dup? | suggested home |
|---|---|---|---|---|
| `blcs/data/augmentation.py:27` & `plcs/data/augmentation.py:30` | `_clone_sample` | deep-clone tensor dict | **byte-identical** | `src/utils/tensor_utils.py` `clone_tensor_dict()` |
| `blcs/training/losses.py:15` & `plcs/training/losses.py:118` | `trajectory_position_loss` / `position_loss` | smooth-L1 position loss (±mask) | same op, masked vs maskless | shared `src/utils/training/losses.py` |
| `plcs/training/losses.py:159,187` | `angular_error`, `_wrapped_angle_diff` | wrapped angular error in radians | **yes** — `plcs/training/metrics.py:126-127` inline; numpy `_angular_error_deg` re-derived in `scripts/analysis/visualize_rotation_error_samples.py:75` & `analyze_loss_dominance.py` | `src/utils/geometry/rotation.py` (new) |
| `plcs/training/losses.py:182` | `_normalize_vector` | safe L2-normalize | generic | `src/utils/tensor_utils.py` |
| `plcs/training/losses.py:193,226,277` | `compute_joint_angles`, `compute_torsion_angles`, `signed_angle_around_axis` | skeleton/3D geometry | imported by analysis scripts already | `src/utils/geometry/skeleton.py` (new) |
| `plcs/utils/pose_geometry.py:14-63` | `court_position_to_world_translation`, `canonical_pose_to_world_pose`, … | normalized-court ↔ world-meter pose conversions | scale constants come from `src/utils/schema/court.py`; blcs projection denorms with same scale | `src/utils/geometry/court_pose.py` (new) |
| `plcs/training/metrics.py:14` | `_flatten_valid` | masked-gather over padded `(B,T,D)` | generic | `src/utils/tensor_utils.py` |

**Highest-value extractions (shortlist):** `_clone_sample`, `_resolve_project_path`, `_resolve_device`, `_seed_everything`, `_heatmaps_to_pixel_coords`, the `angular_error` / wrapped-angle cluster, and the `compute_*` skeleton-geometry functions — these are either byte-identical duplicates or already imported across module boundaries.

---

## D. `src/tennis_scene/` → `src/utils/` promotion candidates

| file:line | helper | what it does | status | note |
|---|---|---|---|---|
| `utils/transforms.py:18` | `rotation_matrix_y(yaw)` | scalar-yaw Y-axis rotation (numpy) | promote | unify with the renderer's `_rotation_matrix_z` |
| `rendering/tennis_scene_renderer.py:151,174` | `_axis_angle_to_matrix`, `_rotation_matrix_z` | batch Rodrigues / Z-axis rotation | promote | co-located but inconsistent rotation helpers → `src/utils/geometry/rotations.py` |
| `utils/transforms.py:40,65` | `apply_plcs_transform[_batch]` | local→court rigid transform of SMPL verts | promote | renderer duplicates this via einsum at `:241-246` |
| `utils/transforms.py:96,118` | `normalize_keypoints` / `denormalize_keypoints` | pixel ↔ UV | promote | zero domain coupling |
| `pipeline/components/{blcs,plcs,court_kp,gvhmr,ball_detection,player_association}.py` | `Result.save()` | `mkdir(parents=True) + json.dump(indent=2)` | promote | **6 verbatim copies** → `src/utils/io.py` `save_json()` |
| `pipeline/components/gvhmr.py:143,334`, `orchestrator.py:309` | project-root via `Path(__file__).parents[3]` | locate repo root for subprocess cwd / 3rd-party import | promote | same gap as §A |

---

## Existing canonical surface (`src/utils/`) — do **not** re-implement these

| Module | Provides |
|---|---|
| `tensor_utils.py` | `masked_mean`, `normalize_padding_mask` |
| `data/augmentation` | UV/visibility augmentation, `normalize_tensor_images_imagenet`, `parse_float_range`, `dilate_temporal_mask`, … |
| `data/heatmaps` | `generate_gaussian_heatmap[s]`, `heatmaps_to_argmax`, `heatmaps_to_soft_argmax`, `heatmaps_to_peaks` |
| `models/` | Transformer blocks, MoE, RoPE, attention, `MLPHead`, domain token embeddings |
| `projection/` | `Camera`, `CameraProjector`, `make_look_at_camera`, `project_points` |
| `rendering/` | `CourtRenderer`, `SkeletonRenderer`, `BallRenderer` |
| `schema/` | court geometry (`CourtKP*`, scales) and player pose schema (COCO-17 / SMPL-H joints, skeletons, angle triplets) |
| `video/` | OpenCV streaming: `probe_video_info`, `read_video_frame`, `OpenCVVideoFrameReader`, `iter_temporal_windows`, `iter_temporal_batches`, `PrefetchIterator`, `normalize_tensor_imagenet`, `BgrToTensorTransform` |

New extractions should land **next to the closest existing module** (e.g. tensor helpers in `tensor_utils.py`, heatmap denorm in `data/heatmaps.py`) and create new modules (`paths.py`, `device.py`, `seeding.py`, `io.py`, `geometry/`) only where no natural home exists.

---

## Recommended sequencing for follow-up PRs

Ordered by value-to-risk (each is an isolated, behavior-preserving move):

1. **Zero-risk dedup:** `_clone_sample` → `tensor_utils.clone_tensor_dict`; collapse the two ImageNet-normalize functions. (Pure deletion of duplicates.)
2. **`src/utils/paths.py`:** `PROJECT_ROOT` + `resolve_project_path()`; migrate `_resolve_project_path`, `_ensure_absolute`, and the tennis_scene `parents[3]` sites.
3. **`src/utils/device.py`:** promote `BasePredictor._resolve_device` to a free function; retire the 3 plcs script copies and `select_devices`.
4. **`src/utils/seeding.py`:** single `seed_everything()`; retire the 2 `_seed_everything` copies and fold in `make_sample_rng`.
5. **`src/utils/io.py`:** `ensure_dir()`, `save_json()`, `load_json()`; migrate the 6 `Result.save()` methods and high-traffic mkdir sites.
6. **`heatmaps_to_pixel_coords`** in `data/heatmaps.py`; retire court/ball manual denorm sites.
7. **`src/utils/geometry/`** (`rotation.py`, `skeleton.py`, `court_pose.py`): consolidate the angular-error / rotation-matrix / skeleton-angle / pose-conversion cluster. (Largest; do last, with focused tests.)

Each step should be its own PR with before/after equality tests so the moves stay provably behavior-preserving.
