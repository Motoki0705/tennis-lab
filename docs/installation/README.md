# Standard multiview pipeline installation

[日本語](README.ja.md)

Generate SceneResult v2 from synchronized videos recorded by 3–5 fixed cameras.
This is the canonical guide for environment setup, asset placement and running the published sample.
For input constraints and the storage format, see [tennis_scene](../../src/tennis_scene/README.md);
for terms of use and attribution, see [LICENSES.md](../../LICENSES.md#code-license-scope).
Obtain each asset under its upstream terms before running the commands below.

<a id="requirements"></a>

## Requirements

- Linux x86-64, or WSL2 with access to an NVIDIA GPU. Python 3.11 is pinned in `.python-version` and synced with uv.
- An NVIDIA GPU. The tested configuration uses an **RTX 5060 Ti / 16 GB VRAM**. During the published sample run
  in an independent clone with a fresh venv, `nvidia-smi` measurements taken about every 5 seconds reached
  a **device-wide maximum of 4,928 MiB**. This includes the initial 1,215 MiB used by other processes;
  the first approximately 22 seconds and brief peaks between samples were not measured.
  The minimum VRAM needed on smaller GPUs has not been verified. The existing WSL2 host had a 24 GB host RAM limit;
  peak RAM usage was not measured. Run without sharing the GPU with other workloads.
- **CUDA toolkit 13.x `nvcc`** and a compatible driver. The lock uses `torch 2.14.1+cu130`.
  The tested configuration uses toolkit **13.0.88** and driver **610.88**. CUDA 13.x requires at least
  driver series 580; CUDA 13.0 GA requires Linux driver 580.65.06 or later
  ([NVIDIA's versioned release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver)).
  The “CUDA Version” shown by `nvidia-smi` does not confirm that the toolkit is installed. Also check `nvcc --version`.
  The extension build stops if the CUDA major versions of torch and the toolkit differ. Using 13.0 matches the locked cu130 build.
- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), a C++ compiler, a CUDA-compatible host compiler,
  `ffmpeg` (for MP4 output) and Ninja. On Ubuntu, install `build-essential`, `ninja-build` and `ffmpeg` at the OS level.
- The workspace used **about 20.87 GB (19.43 GiB)** of disk space, including the independent clone, venv,
  isolated uv/HF caches, weights, sample, CUDA build, component cache, scene and full-length visualization
  (about 19.94 GB when preparation finished). This excludes the CUDA toolkit, OS packages and the workspace
  for separately downloading third-party assets. Space requirements vary with caches and the number of runs;
  allow room for those and additional outputs. See the [installation verification record](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)
  for the measurement scope.

CPU pipeline execution is outside the scope of this guide. Asset checks, extension import/dispatch validation
and visualization run on the CPU.

<a id="clone-and-sync"></a>

## Clone and sync

```bash
git clone https://github.com/Motoki0705/tennis-lab.git
cd tennis-lab
uv sync --locked
git submodule update --init third_party/DINO third_party/dinov3
```

`uv sync --locked` creates a fresh `.venv` including the dev and gpu groups.
`third_party/GVHMR` and `third_party/nht` are not needed for the standard pipeline.
Use the submodule versions pinned by this checkout's gitlinks; do not add `--remote`.
Run all subsequent commands from the repository root.

<a id="dino-cuda-extension"></a>

## DINO CUDA extension

Check your GPU's compute capability in [NVIDIA's list](https://developer.nvidia.com/cuda-gpus)
and set `TORCH_CUDA_ARCH_LIST` explicitly. **12.0 below is an example for the RTX 5060 Ti**.
For another GPU, replace it with a value supported by that GPU and CUDA 13.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=2
"$CUDA_HOME/bin/nvcc" --version
nvidia-smi
bash scripts/build_dino_extension.sh "$PWD" "$PWD/outputs/dino_extension"
export PYTHONPATH="$PWD/outputs/dino_extension/lib"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -c \
  'from src.submodules.models import validate_dino_extension; print(validate_dino_extension())'
```

Adjust `CUDA_HOME` for your environment, for example to `/usr/local/cuda`. `PYTHONPATH` is also required
for preflight and pipeline execution in the same shell. Set it again when using another shell.

The script creates `TENNIS_LAB_DINO_OPS_BUILD_CONFIG` under the canonical path contract and builds
DINO and the shared CUDA extension through the root `setup.py`. It writes to
`outputs/dino_extension/{lib,temp,cache}` and `build.json`, without changing sources in submodules
or the shared venv. Compatibility fixes apply only to the generated DINO sources.
The build sets `CUDA_VISIBLE_DEVICES=''` and does not use the GPU. `validate_dino_extension()` also
checks forward/backward dispatch with CPU tensors; it does not numerically validate CUDA kernels.
If you sync to a different torch version, rebuild into a new output directory and explicitly use its `lib`.

<a id="required-assets"></a>

## Required assets

Place all assets as **real files** inside the repository. The path contract rejects symlinks outside the checkpoint root.
Preflight checks known weights and the sample against pinned SHA-256 values; it does not recompress or convert files,
or automatically switch to another release. The authoritative details on license scope are in
[third-party weights](../../LICENSES.md#third-party-weights) and [published artifacts](../../LICENSES.md#published-artifacts).

<!-- checkpoint-assets:start -->
| Asset | Destination (relative to `ckpt/`) | Source | License / terms |
| --- | --- | --- | --- |
| court | `court_detection/multiscale_depth3/b863df1f01f0.ckpt` | [Source](https://huggingface.co/Motoki098/tennis-lab-models) | DINOv3 License |
| ball | `ball_detection/i935-mixed-ft-s42-epoch09.ckpt` | [Source](https://huggingface.co/Motoki098/tennis-lab-models) | CC BY-NC 4.0 |
| detector | `dino/checkpoint0029_4scale_swin.pth` | [Source](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_) | DINO / Apache-2.0¹ |
| vitpose | `vitpose/vitpose-h-multi-coco.pth` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | ViTPose / Apache-2.0¹ |
| hmr2 | `hmr2/epoch=10-step=25000.ckpt` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | HMR2 / MIT¹; separate SMPL terms apply |
| gvhmr | `gvhmr/gvhmr_siga24_release.ckpt` | [Source](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) | GVHMR non-commercial terms |
| body_model | `body_models/smplx/SMPLX_NEUTRAL.npz` | [Source](https://smpl-x.is.tue.mpg.de/) | SMPL-X Model License / registration and acceptance |
| aflink | `person_tracking/AFLink_epoch20.pth` | [Source](https://drive.google.com/file/d/1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9/view) | AFLink weight terms unconfirmed |
| tracking_encoder | `player_association/person_vit_clip_reid.pth` | [Source](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view) | CLIP-ReID / MIT¹ |
| smpl_faces | `body_models/smplh/neutral/model.npz` | [Source](https://mano.is.tue.mpg.de/) | MANO / SMPL+H License / registration and acceptance |
| dinov3 | `dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | [Source](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | DINOv3 License / application required |
| smpl_hmr2_mean_params | `body_models/gvhmr/smpl_mean_params.npz` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/network/hmr2/configs/smpl_mean_params.npz) | GVHMR / SMPL / SMPL-X terms |
| smpl_smplx_to_smpl | `body_models/gvhmr/smplx2smpl_sparse.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx2smpl_sparse.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smpl_coco17_regressor | `body_models/gvhmr/smpl_coco17_J_regressor.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smplx_verts437 | `body_models/gvhmr/smplx_verts437.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smplx_verts437.pt) | GVHMR / SMPL / SMPL-X terms |
| smpl_smpl_neutral_joint_regressor | `body_models/gvhmr/smpl_neutral_J_regressor.pt` | [Source](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt) | GVHMR / SMPL / SMPL-X terms |
<!-- checkpoint-assets:end -->

¹ These are the licenses stated for the upstream repository's code. Permission for the code is not treated
as a blanket grant of all rights to training data, body models or separately distributed weights.
AFLink remains in the standard path under a decision to continue using it with its terms unconfirmed.
The basis and constraints are in the [StrongSORT NOTICE](../../src/tasks/person_tracking/strongsort_NOTICE.md).

### Project weights and the published sample

Use the [weight acquisition CLI](../../src/utils/README.md#published-detector-weights) and
[sample acquisition CLI](../../src/utils/README.md#published-multiview-sample).
Existing files are checked; mismatches are not overwritten.

```bash
.venv/bin/python -m src.utils.model_weights --checkpoint-root "$PWD/ckpt"
.venv/bin/python -m src.utils.sample_dataset --data-root "$PWD/data"
```

The sample is `video_000/clip_000` from `Motoki098/tennis-lab-multiview-sample`:
1920×1080, 59.94006 fps, 1,010 frames and 3 cameras. It includes upscaling and letterboxing of cam0.
See the published dataset card for synchronization and provenance. Its terms are **CC BY-NC-ND 4.0**.
The videos, manifest and LICENSE are placed as follows.

<!-- sample-assets:start -->
| Destination | Source / terms |
| --- | --- |
| `data/samples/cam0.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam1.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/cam2.mp4` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/clip.json` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
| `data/samples/LICENSE` | [Published dataset](https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample) / CC BY-NC-ND 4.0 |
<!-- sample-assets:end -->

### Third-party weights and body models

- **DINO**: Download `checkpoint0029_4scale_swin.pth` from the 4-scale Swin-L Drive link in the
  [official model zoo](https://github.com/IDEA-Research/DINO/tree/d84a491d41898b3befd8294d1cf2614661fc0953#model-zoo).
  The initial Swin weights used for training are not needed.
- **ViTPose-H, HMR2 and GVHMR**: Download the three files in the table from the
  [official Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) linked by the
  [official GVHMR INSTALL](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md).
  The GVHMR conda environment, YOLO and DPVO are not needed.
- **CLIP-ReID**: Select **ViT-CLIP-ReID / Market / model** in the
  [official README](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs).
  The filename is `Market1501_clipreid_ViT-B-16_60.pth`. Place it at
  `ckpt/player_association/person_vit_clip_reid.pth` without changing its contents.
  The [verification record](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)
  documents that the official distribution and the previously used version are identical.
- **AFLink**: Download `AFLink_epoch20.pth` from the Drive link in the
  [official StrongSORT README](https://github.com/dyhBUPT/StrongSORT#datamodel-preparation).
  Its unconfirmed terms are described in the NOTICE above.
- **DINOv3 ViT-B/16**: Apply through [Meta's official site](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
  and obtain `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
  The published court checkpoint includes the backbone, but **the current model construction also requires this standalone file**.
- **SMPL-X**: Register and accept the terms at the [official site](https://smpl-x.is.tue.mpg.de/),
  then place `SMPLX_NEUTRAL.npz` from the NPZ body models. The male/female SMPL-X models, SMPL itself
  and the SMPL-H PKL files are not needed for this standard run.
- **SMPL-H**: Register and accept the terms at [MANO / SMPL+H](https://mano.is.tue.mpg.de/),
  then obtain the Extended SMPL+H models for AMASS. Place the extracted `neutral/model.npz` at the location in the table.
  Visualization reads its mesh faces. Do not convert PKL to NPZ.
- **Five SMPL auxiliary assets**: Follow the canonical [src/submodules guide](../../src/submodules/README.md#smpl補助資産の取得)
  for acquisition, verification and placement. Obtain them from the pinned official GVHMR commit and
  use `install_smpl_assets` to place them at the table's destinations. This guide does not duplicate those steps.

For Google Drive folders, download only the relevant files and place them under the names in the table.
Models requiring registration or an application are not redistributed by this repository or supplied from other mirrors.

<a id="preflight"></a>

## Preflight

```bash
export PYTHONPATH="$PWD/outputs/dino_extension/lib"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.tennis_scene.scripts.preflight
```

Require `status: ok` and exit code 0. Missing files, content mismatches, read failures and extension incompatibilities
are **reported together** in JSON; any failure produces exit code 1. Explicitly fix the acquired files and rerun.
The check covers every asset from `enabled_model_assets()`, the DINO and DINOv3 sources, standalone DINOv3 weights,
SMPL-H and the regressor for visualization, and input videos (plus the manifest and LICENSE for the published sample).
It does not download files, substitute models, run inference or use the GPU. Source checks verify that required files exist.
This is a check for installing the published standard path listed above, rather than a general command for evaluating other model weights.
Resolve every failure before starting the pipeline.

<a id="run-the-sample"></a>

## Run the sample

```bash
PYTHONPATH="$PWD/outputs/dino_extension/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  output_directory=tennis_scene/generate/clean_install output_name=scene \
  cache.directory=tennis_scene/cache/clean_install
```

On this repository's shared GPU, run through the [training queue](../../.agents/skills/training-queue/SKILL.md).
In the clean installation verification on the RTX 5060 Ti, the time from pipeline start to saving the scene was
**30 minutes 27 seconds** ([measurement record](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)).
This is a reference measurement for one clip of the published sample. It excludes queue waiting, process startup,
asset acquisition, building and visualization.

The run saves `outputs/tennis_scene/generate/clean_install/scene.npz` and `scene.metadata.json`.
Check that the metadata has `scene_schema_version=2` and `status=ok`.
Also use the canonical reader to check masks, reason codes and coordinate systems.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
from pathlib import Path
from src.tennis_scene.archive import load_scene_result
scene = load_scene_result(Path('outputs/tennis_scene/generate/clean_install/scene.npz'))
assert scene.metadata['scene_schema_version'] == 2 and scene.metadata['status'] == 'ok'
assert scene.num_frames == 1010
print(scene.num_frames, scene.fps, scene.metadata['camera_ids'])
PY
```

The output cache belongs to this input, configuration and code. Use a fresh clone and empty output locations
for a new installation verification. Completion confirms that processing and storage contracts hold;
it does not guarantee 3D accuracy against ground truth. Compared with an earlier run using the same inputs
and inference code, the numbers of valid ball 3D and player root/SMPL frames differed.
The cause is unresolved; this verification does not guarantee byte-identical outputs.
See the measurement record above for details.

<a id="visualize"></a>

## Visualize

Check that `ffmpeg -version` succeeds, then render using SMPL-H and the regressor from the same checkpoint root.

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venv/bin/python -m src.tennis_scene.scripts.visualization \
  input=tennis_scene/generate/clean_install/scene.npz \
  output=tennis_scene/visualize/clean_install/scene.mp4 \
  style.player_representation=smpl
```

Play the MP4 and check the court, both players, the ball and missing intervals.
CPU rendering in the clean installation verification took **6 minutes 3 seconds** (timed by the parent session,
including startup) and produced a 1200×800 MP4 with 1,010 frames lasting about 16.85 seconds.
The parent session checked the frame at 9 seconds and confirmed that the court, both players' SMPL meshes,
the ball and the minimap were rendered. See the [review UI](../../src/tennis_scene/visualization/review/README.md)
for comparison with each component's inputs.

<a id="rehearse-clean-install"></a>

## Rehearse a clean install

[scripts/prepare_clean_install.py](../../scripts/prepare_clean_install.py) runs this sequence in an independent
clone and venv, recording each step's time, logs, copy sources, verification results and disk usage in `preparation.json`.
**For this local verification only**, third-party assets acquired by the user through registration, applications
or Drive are checked and copied from an explicitly specified checkpoint root, then checked again after copying.
Project weights and the sample are fetched from the Hub. The script does not link an existing venv, data or ckpt,
and does not copy weights without checking them. For your own installation, obtain assets from the official sources above.

```bash
python3 scripts/prepare_clean_install.py \
  --workspace /absolute/path/to/new-clean-install \
  --repository https://github.com/Motoki0705/tennis-lab.git --revision <commit> \
  --local-checkpoint-root /absolute/path/to/user-acquired/ckpt \
  --cuda-home /usr/local/cuda-13.0 --cuda-arch 12.0
```

To test an unpublished branch, specify `--repository /absolute/path/to/worktree` and its commit.
The script uses `git clone --no-local`, so even git objects are not shared with the source checkout.
An existing workspace is rejected. When CPU preparation succeeds, the status becomes `ready_for_gpu`
and the script **stops before GPU execution**. The parent submits the generated `run_pipeline.sh` to the training queue
and runs `visualize.sh` after it finishes. Add GPU elapsed time, peak VRAM, scene checks and video review
results to the subsequent execution record.

The [verification record from 2026-10-21](../../knowledge/nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)
confirms **12 minutes 41 seconds** of CPU preparation, followed by GPU completion through the queue,
scene checks with the canonical reader and CPU visualization of the new scene.
This verification cloned the local worktree above rather than GitHub, using `git clone --no-local` and a pinned commit.
Third-party assets requiring registration or an application were verified copies of files already acquired by the user.
The timing excludes new registration, manual acquisition, CUDA toolkit installation and OS setup.
