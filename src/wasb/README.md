# WASB-based Tennis Dataset Augmentation

This directory contains utilities and models built around the WASB-SBDT codebase for
semi-automatic tennis ball annotation and dataset expansion, primarily to train
transformer-based models such as `hrcnet`.

## High-level Goal

Extend the original WASB tennis dataset (`data/tennis/game1..10`) with additional
games (`game11..`) generated from raw match videos:

1. Download many tennis match videos (e.g. from YouTube).
2. Run pre-trained WASB/HRNet models to obtain automatic ball trajectories.
3. Segment videos into rallies (clips) using a learned **clip segmenter** model.
4. Refine / complete noisy or missing trajectories using a learned **trajectory
   completer** model.
5. Export frames and annotations in the same format as the original
   `data/tennis` layout so that WASB-SBDT can train on the augmented dataset.

## Target Data Format (Tennis)

We follow the existing WASB tennis format:

```text
data/tennis/gameN/ClipK/
  frame_0000.jpg
  frame_0001.jpg
  ...
  Label.csv
```

The final `Label.csv` is extended as:

```text
file name,visibility,x-coordinate,y-coordinate,status,score
0000.jpg,1,599,423,0,0.95
0001.jpg,2,601,406,0,0.00
0002.jpg,0,-1,-1,0,0.00
...
```

- `visibility`:
  - `1`: direct HRNet/WASB detection (above score threshold)
  - `2`: value completed by the trajectory completer model
  - `0`: missing / unresolvable frame (no reliable detection or completion)
- `status`: currently always `0` (reserved for future use).
- `score`:
  - HRNet detection confidence for `visibility=1` frames.
  - `0.0` for completed (`visibility=2`) or missing (`visibility=0`) frames.

This keeps compatibility with existing loaders while adding extra signal.

## Components and Responsibilities

### 1. WASBPredictor (HRNet Inference)

`src/wasb/inference/hrnet_predictor.py`

- Wraps the original WASB-SBDT codebase and provides a simple API:

  - Input: `frames: np.ndarray (T, H, W, 3)` (RGB images, T >= frames_in)
  - Output dict containing:
    - `ball_uv`: normalized ball coordinates `(N, 2)`
    - `ball_xy_px`: pixel coordinates `(N, 2)`
    - `visibility`: visibility flags `(N,)`
    - `score`: detection scores `(N,)`
    - `frame_indices`: frame indices `(N,)`

- Internally:
  - Uses `models.build_model(cfg)` and pre-trained WASB checkpoints
    (e.g. `pretrained/wasb_tennis_best.pth.tar`).
  - Applies the WASB postprocessor + tracker to obtain a single trajectory per
    sequence.

This is the **teacher model** used to generate pseudo-annotations from raw videos.

### 2. Video Extraction Utilities

(Planned) `src/wasb/data/video_extractor.py`

- Convert raw match videos into frame sequences.
- Example responsibilities:
  - `extract_frames(video_path, output_dir, fps=25)`
  - Naming frames consistently (`frame_0000.jpg`, ...).
  - One video corresponds to one `gameN`, and later is internally split into
    multiple `ClipK` by the clip segmenter.

### 3. Clip Segmenter Model

(Planned) `src/wasb/models/clip_segmenter.py`

Goal: learn to segment a long video (HRNet trajectory) into tennis rallies
(clips).

- **Inputs** (per frame):
  - Ball coordinates `(x, y)` in pixels or normalized `[0, 1]`.
  - Detection score and/or visibility.
- **Outputs**:
  - Frame-wise labels: e.g. rally vs. non-rally, or begin/end boundaries.
- **Training data**:
  - Start from simple rule-based segmentation using HRNet outputs on top of
    `data/tennis`.
  - Manually refine some videos ("video -> HRNet -> manually split into
    clips") and use them as supervision.
- **API idea**:

  ```python
  class ClipSegmenter:
      def predict_segments(
          self,
          xy: np.ndarray,       # [T, 2]
          score: np.ndarray,    # [T]
          vis: np.ndarray,      # [T]
      ) -> list[tuple[int, int]]:  # [(start, end), ...]
          ...
  ```

Initially we will start with a rule-based heuristic, then optionally replace it
with a learned model.

### 4. Trajectory Completer Model

`src/wasb/models/trajectory_completer.py`

**Implemented.** Provides multiple strategies to complete noisy/missing trajectories:

#### Available Completers

1. **PhysicsInterpolator** (default for short gaps)
   - Uses quadratic interpolation (approximates parabolic motion)
   - Handles gap bridging and outlier detection
   - Configurable velocity/acceleration thresholds

2. **BiLSTMCompleter** (learned model)
   - Bidirectional LSTM for context-aware completion
   - Requires training on existing tennis data
   - Optional checkpoint loading

3. **HybridCompleter** (recommended)
   - Combines physics for short gaps + learned model for complex gaps
   - Falls back to extended physics if no trained model available

#### API

```python
from src.wasb.models import create_completer, PhysicsInterpolator

# Factory function
completer = create_completer(
    method="hybrid",  # "physics", "bilstm", or "hybrid"
    checkpoint_path=None,  # Optional path to trained BiLSTM weights
    physics_gap_threshold=5,
    max_gap=15,
)

# Direct usage
result = completer.complete(
    xy=xy_array,         # [T, 2] ball positions
    visibility=vis_mask, # [T] boolean mask
    score=scores,        # [T] detection scores
)

# Result contains:
# - result.xy: Completed trajectory [T, 2]
# - result.visibility: Updated visibility flags [T]
#     - 1: Original detection
#     - 2: Completed by model
#     - 0: Could not complete
# - result.confidence: Per-frame confidence [T]
# - result.gaps_filled: Number of frames filled
# - result.outliers_removed: Number of outliers detected
```

#### Pipeline Integration

Trajectory completion is automatically integrated into `AnnotationPipeline`:

```python
from src.wasb.pipeline import AnnotationPipeline, PipelineConfig

config = PipelineConfig(
    use_completion=True,           # Enable completion (default: True)
    completion_method="hybrid",    # Method to use
    physics_gap_threshold=5,       # Max gap for physics interpolation
    max_completion_gap=15,         # Max gap to attempt any completion
)
```

At inference time:
- Frames with strong HRNet detections → `visibility=1`
- Frames completed by model → `visibility=2`, `score=0.0`
- Unreliable/outlier frames → `visibility=0`

### 5. Tennis Format I/O Helpers

`src/wasb/tennis_format.py`

**Implemented.** Helpers for reading/writing tennis `Label.csv` files:

```python
from src.wasb.tennis_format import (
    load_label_csv,
    save_label_csv,
    TennisLabelRow,
    row_from_detection,      # Create visibility=1 row
    row_from_completion,     # Create visibility=2 row
    row_from_visibility,     # Create row with explicit visibility
    make_empty_row,          # Create visibility=0 row
)

# Load existing labels
rows = load_label_csv("data/tennis/game1/Clip1/Label.csv")

# Save labels
save_label_csv("output/Label.csv", rows)
```

## End-to-end Annotation Workflow (Planned)

For a single raw video (mapped to `gameN`):

1. **Extract frames**
   - Use the video extractor to save `frame_0000.jpg, frame_0001.jpg, ...`.

2. **Run HRNet / WASB inference**
   - Load a pre-trained WASB checkpoint.
   - Call `WASBPredictor.predict(frames)` and obtain per-frame ball positions,
     scores, and visibility.

3. **Segment into clips**
   - Use a rule-based method or `ClipSegmenter` to convert the long trajectory
     into a list of rally intervals `[(start, end), ...]`.
   - Each interval becomes `Clip1, Clip2, ...` under `gameN`.

4. **Create initial annotations**
   - For each clip, assign per-frame visibility and coordinates:
     - If `score >= threshold`: `visibility=1` and use HRNet `(x, y)`.
     - Otherwise: `visibility=0` with dummy coordinates.

5. **Run trajectory completion**
   - For each clip, pass the initial trajectory into `TrajectoryCompleter`.
   - For frames completed by the model:
     - Set `visibility=2` and `score=0.0`.
   - Frames that remain unreliable stay `visibility=0` and `score=0.0`.

6. **Export to tennis format**
   - Use the tennis I/O helpers to write `Label.csv` with the agreed columns
     and to ensure each frame has a corresponding label row.

7. **Integrate into WASB-SBDT training**
   - Add `gameN` to `third_party/WASB-SBDT/src/configs/dataset/tennis.yaml`
     (`train.matches` / `test.matches`).
   - Re-train HRNet/HRCNet on the extended dataset.

## Development Roadmap (Summary)

1. ✅ Implement tennis `Label.csv` I/O helpers.
2. ✅ Implement a small script to run `WASBPredictor` on a single video and save
   raw trajectories.
3. ✅ Build the trajectory completer model (physics-based + Bi-LSTM).
4. ✅ Implement rule-based clip segmentation and end-to-end generation of
   `game11` from one video.
5. 🔄 Train Bi-LSTM completer on existing tennis clips (optional, physics works well).
6. 🔲 Collect a small amount of manually segmented data and train a proper
   clip segmenter model.
7. 🔲 Replace the rule-based stage with the learned clip segmenter and scale to
   more videos.
