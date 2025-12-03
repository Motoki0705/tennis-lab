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

(Planned) `src/wasb/models/trajectory_completer.py`

Goal: given a noisy and partially missing 2D trajectory `[T, 2]`, predict a
clean and completed trajectory of the same length.

- **Training data source**:
  - Existing `data/tennis/game1..10` with ground-truth labels.
  - For each clip, we obtain a full sequence `[T, 2]` where the ball is
    visible (or nearly so) in all frames.
- **Input construction**:
  - Start from the clean trajectory and corrupt it with:
    - Random masking (set some timesteps as missing).
    - Additive noise on positions.
  - Inputs: `obs_xy [T, 2]` + mask or visibility flags.
- **Outputs**:
  - Predicted `clean_xy [T, 2]`.
- **Model examples**:
  - Bi-LSTM, 1D CNN, or Transformer encoder over the time dimension.
- **API idea**:

  ```python
  class TrajectoryCompleter:
      def complete(
          self,
          xy: np.ndarray,   # [T, 2], with NaNs or dummy values where missing
          vis: np.ndarray,  # [T], 0/1
          score: np.ndarray # [T]
      ) -> np.ndarray:      # [T, 2]
          ...
  ```

At inference time we combine this with HRNet predictions:

- Frames with strong, reliable HRNet detections become `visibility=1`.
- The completer fills in the remaining frames; we mark them `visibility=2` and
  `score=0`.
- Truly unreliable / outlier frames can be kept as `visibility=0` with dummy
  coordinates.

### 5. Tennis Format I/O Helpers

(Planned) `src/wasb/io/tennis_format.py`

Provide small helpers to read/write the tennis `Label.csv` files:

- `read_label_csv(path) -> pd.DataFrame or List[Record]`.
- `write_label_csv(records, path)`.

These helpers encapsulate the logic for:

- Mapping between `(x, y)`, `visibility`, `status`, `score` and the CSV
  columns.
- Ensuring that file names (`0000.jpg`, ...) are consistent with the actual
  frame files on disk.

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

1. Implement tennis `Label.csv` I/O helpers.
2. Implement a small script to run `WASBPredictor` on a single video and save
   raw trajectories.
3. Build and train the trajectory completer model using existing tennis clips.
4. Implement rule-based clip segmentation and end-to-end generation of
   `game11` from one video.
5. Collect a small amount of manually segmented data and train a proper
   clip segmenter model.
6. Replace the rule-based stage with the learned clip segmenter and scale to
   more videos.
