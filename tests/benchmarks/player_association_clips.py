"""Cross-camera player association on labelled clips of a structured dataset.

``observe`` (GPU, through the shared training queue) runs court
detection/calibration and person detection, tracking and pose of the default
``pipeline.yaml`` into a run-owned store per clip. Ball, body and
reconstruction nodes are disabled; the side comes from the reviewed ball when
the association is evaluated. Each camera runs on its own so that one camera
whose tracking stops (for example ``person_capacity_exceeded``) is reported
with its evidence while the other cameras still produce tracks.
``observe.json`` lists, per clip and camera, the status and every track with
its observed frame count. ``sheets`` (CPU) renders, per clip and camera, a
contact sheet of every observed track (evenly spaced crops with their frame
index) from the stored tracks; it is the reviewing aid for the evaluation
labels. ``labels`` (CPU) turns a reviewed track assignment (``--review``, see
``tests/benchmarks/labels/player_association``) into tracker-independent box
labels, one JSON per clip under ``--labels-dir``.

``calibrate`` (CPU) fits the data-driven association parameters on observed
clips that are *not* labelled (the labels stay a test set): cross-camera track
pairs sharing ``--min-shared-s`` are pseudo-labelled by their median footpoint
distance (same person below ``--positive-below-m``, different people above
``--negative-above-m``); the Rayleigh scale of the positives is the geometry
``sigma_m`` and a class-balanced logistic fit of their appearance cosines is
the appearance ``slope``/``center``. ``evaluate`` (CPU) associates every
labelled clip with ``--config`` (optionally geometry only) and scores it
against the labels; per clip it also draws the court-plane footpoints coloured
by predicted player and by label. Both read the side of every camera from the
reviewed-ball decision of ``court_side_clips.py`` (``--sides``). Track
appearances are cached per clip under ``--report/appearance``. Nothing else is
written outside ``--report``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.player_association.appearance.affinity import (
    fit_cosine_log_likelihood_ratio,
    segment_embedding,
)
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
    embed_tracks,
)
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    load_association_config,
)
from src.tasks.player_association.evaluation import (
    CameraPrediction,
    ClipLabels,
    evaluate,
    match_to_labels,
)
from src.tasks.player_association.evaluation.labels import (
    ReviewedTrack,
    materialize,
    review_sha256,
)
from src.tasks.player_association.geometry.affinity import rayleigh_scale
from src.tasks.player_association.geometry.footpoints import (
    ground_distance,
    ground_footpoints,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.video import OpenCVVideoFrameReader

CONFIG_DIR = Path(__file__).resolve().parents[2] / "src/tennis_scene/configs"


def compose_runtime(repo: Path, report: Path, device: str, overrides: list[str], name: str) -> tuple[PipelineRuntimeConfig, list[str]]:
    applied = [f"paths.project_root={CONFIG_DIR.parents[2]}", f"paths.data_root={repo / 'data'}",
        f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
        f"paths.artifact_root={report}", f"paths.output_root={report}", f"paths.cache_root={report / 'cache'}",
        f"device={device}", "output_directory=run", "ball_detection.enabled=false", "execution.court_side=load",
        "player_reconstruction.enabled=false", "ball_reconstruction.enabled=false", "gvhmr.enabled=false", *overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="pipeline", overrides=applied)
    (report / f"{name}.pipeline_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return PipelineRuntimeConfig.from_config(config, bind_inputs=False), applied


def clip_source(clip: Path) -> tuple[ClipManifest, ClipSource]:
    manifest = ClipManifest.load(clip)
    videos = tuple(manifest.media_path(camera) for camera in manifest.camera_ids)
    return manifest, build_clip_source(videos, tuple(manifest.camera_ids), clip_id=manifest.clip_id)


def observe(runtime: PipelineRuntimeConfig, code_identity: str, clip: Path, store_root: Path) -> dict[str, Any]:
    """Court calibration, then person detection, tracking and pose camera by camera."""
    _, source = clip_source(clip)
    nodes = standard_definition(runtime, source, code_identity=code_identity)
    store = ClipStore(store_root, json_value(source))
    ComponentRunner(nodes, store).run(targets=("court_calibration",))
    cameras: dict[str, Any] = {}
    for camera in source.camera_ids:
        runner = ComponentRunner(nodes, store)
        try:
            runner.run(targets=(f"pose_estimation/{camera}",))
        except ReconstructionUnavailable as stopped:
            cameras[camera] = {"status": "stopped", "reason": stopped.reason, "message": str(stopped),
                               "diagnostics": json_value(stopped.diagnostics), "seconds": runner.seconds}
            continue
        reference = store.active(f"person_tracking/{camera}")
        if reference is None:
            raise RuntimeError(f"person_tracking/{camera} completed without an adopted artifact")
        tracks = store.load(reference, ArtifactCodec(PersonTrackingOutput))
        cameras[camera] = {"status": "ok", "seconds": runner.seconds,
                           "tracks": [{"track_id": int(track), "observed_frames": int(tracks.observed[row].sum()),
                                       "source_track_ids": list(tracks.source_track_ids[row])}
                                      for row, track in enumerate(tracks.track_ids)],
                           "tracklet_links": len(tracks.tracklet_links)}
    return {"frames": source.num_frames, "cameras": cameras}


def track_sheet(video: Path, tracks: PersonTrackingOutput, *, samples: int, crop_height: int) -> np.ndarray | None:
    """One row per track: ``samples`` evenly spaced observed crops labelled ``t<id> f<frame>``."""
    wanted: dict[int, list[tuple[int, int]]] = {}
    for row in range(len(tracks.track_ids)):
        frames = np.flatnonzero(tracks.observed[row])
        if not len(frames):
            continue
        for column, frame in enumerate(frames[np.linspace(0, len(frames) - 1, min(samples, len(frames))).round().astype(int)]):
            wanted.setdefault(int(frame), []).append((row, column))
    if not wanted:
        return None
    width = crop_height // 2
    label = 120
    rows = sorted({row for cells in wanted.values() for row, _ in cells})
    canvas: np.ndarray = np.full((len(rows) * (crop_height + 18), label + samples * (width + 4), 3), 32, np.uint8)
    for index, row in enumerate(rows):
        y = index * (crop_height + 18)
        text = f"t{int(tracks.track_ids[row])} n={int(tracks.observed[row].sum())}"
        cv2.putText(canvas, text, (4, y + crop_height // 2), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    last = max(wanted)
    for packet in OpenCVVideoFrameReader(video, max_frames=last + 1):
        for row, column in wanted.get(packet.index, ()):
            x1, y1, x2, y2 = tracks.boxes_xyxy[row, packet.index]
            pad_x, pad_y = .1 * (x2 - x1), .05 * (y2 - y1)
            height, image_width = packet.frame.shape[:2]
            left, right = int(max(0, x1 - pad_x)), int(min(image_width, x2 + pad_x))
            top, bottom = int(max(0, y1 - pad_y)), int(min(height, y2 + pad_y))
            if right <= left or bottom <= top:
                continue
            crop = packet.frame[top:bottom, left:right]
            scale = min(crop_height / crop.shape[0], width / crop.shape[1])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            y = rows.index(row) * (crop_height + 18)
            x = label + column * (width + 4)
            canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(canvas, f"f{packet.index}", (x, y + crop_height + 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (200, 200, 0), 1)
    return canvas


def sheets(clip: Path, store_root: Path, output: Path, *, samples: int, crop_height: int) -> dict[str, str]:
    manifest, source = clip_source(clip)
    store = ClipStore(store_root, json_value(source))
    written: dict[str, str] = {}
    for video in source.videos:
        reference = store.active(f"person_tracking/{video.camera_id}")
        if reference is None:
            continue
        sheet = track_sheet(video.path, store.load(reference, ArtifactCodec(PersonTrackingOutput)), samples=samples, crop_height=crop_height)
        if sheet is None:
            continue
        path = output / f"{video.camera_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written[video.camera_id] = str(path)
    return written


def labels(dataset: Path, report: Path, review_path: Path, output: Path) -> list[str]:
    """Materialize the reviewed clips of ``review_path`` from the observation stores under ``report``."""
    review = yaml.safe_load(review_path.read_text())
    observed = json.loads((report / "observe.json").read_text())
    written = []
    for clip_id, clip_review in review["clips"].items():
        if observed["clips"].get(clip_id, {}).get("status") != "ok":
            raise ValueError(f"{clip_id} has no completed observation in {report}")
        video_id, clip_name = clip_id.split("/")
        _, source = clip_source(dataset / "videos" / video_id / "clips" / clip_name)
        store = ClipStore(report / "stores" / clip_id, json_value(source))
        tracks: dict[str, list[ReviewedTrack]] = {}
        for camera, record in observed["clips"][clip_id]["cameras"].items():
            if record["status"] != "ok":
                raise ValueError(f"{clip_id} {camera} tracking stopped ({record['reason']}); it cannot be labelled")
            reference = store.active(f"person_tracking/{camera}")
            if reference is None:
                raise RuntimeError(f"{clip_id} person_tracking/{camera} has no adopted artifact")
            output_tracks = store.load(reference, ArtifactCodec(PersonTrackingOutput))
            tracks[camera] = [ReviewedTrack(int(track), output_tracks.boxes_xyxy[row], output_tracks.observed[row])
                              for row, track in enumerate(output_tracks.track_ids)]
        provenance = {"review": {"path": str(review_path.name), "sha256": review_sha256(review_path),
                                 "selection": clip_review["selection"]},
                      "observation": {"run": review["observe_run"], "code_sha256": observed["code_sha256"],
                                      "config_overrides": observed["config_overrides"]}}
        clip_labels = materialize(clip_id, source.num_frames, clip_review, tracks, provenance)
        path = output / f"{clip_id}.json"
        clip_labels.save(path)
        written.append(str(path))
    return written


def load_observation(dataset: Path, observe_report: Path, clip_id: str
                     ) -> tuple[ClipSource, dict[str, PersonTrackingOutput], dict[str, PinholeCamera]]:
    """Source, per-camera tracks and camera-local (side-unresolved) cameras of one observed clip."""
    video_id, clip_name = clip_id.split("/")
    _, source = clip_source(dataset / "videos" / video_id / "clips" / clip_name)
    store = ClipStore(observe_report / "stores" / clip_id, json_value(source))
    calibration_ref = store.active("court_calibration")
    if calibration_ref is None:
        raise RuntimeError(f"{clip_id} has no court_calibration in {observe_report}")
    calibration = store.load(calibration_ref, ArtifactCodec(CourtCalibrationOutput)).calibration
    tracks = {}
    for camera in source.camera_ids:
        reference = store.active(f"person_tracking/{camera}")
        if reference is None:
            raise RuntimeError(f"{clip_id} has no person_tracking/{camera} in {observe_report}")
        tracks[camera] = store.load(reference, ArtifactCodec(PersonTrackingOutput))
    cameras = {view.camera.camera_id: view.camera for view in calibration.views}
    if set(cameras) != set(source.camera_ids):
        raise ValueError(f"{clip_id}: calibrated cameras {sorted(cameras)} differ from the source cameras (excluded: {calibration.excluded})")
    return source, tracks, cameras


def side_turns(sides: dict[str, Any], clip_id: str) -> dict[str, bool]:
    """Per-camera half turn of the reviewed-ball side decision; a clip without one stops."""
    records = [record for record in sides["clips"] if record["clip_id"] == clip_id]
    decision = records[0].get("annotation") if len(records) == 1 else None
    if not decision or not decision.get("decided"):
        raise ValueError(f"{clip_id} has no decided reviewed-ball side in the --sides report")
    return dict(zip(records[0]["camera_ids"], decision["view_half_turns"], strict=True))


def track_appearances(cache: Path, source: ClipSource, tracks: dict[str, PersonTrackingOutput], encoder_name: str, repo: Path,
                      device: str, sampling: CropSamplingConfig) -> dict[str, tuple[TrackAppearance, ...]]:
    """Sampled-crop embeddings of every track, cached per clip and encoder."""
    path = cache / f"{source.clip_id}.{encoder_name}.npz"
    key = json.dumps({"encoder": encoder_name, "sampling": json_value(sampling),
                      "tracks": {c: t.track_ids.tolist() for c, t in tracks.items()}}, sort_keys=True)
    if path.is_file():
        stored = np.load(path)
        if str(stored["key"]) != key:
            raise ValueError(f"{path} was cached for other tracks or sampling; use a new --report")
    else:
        encoder = build_encoder(encoder_name, checkpoint_root=repo / "ckpt", external_root=repo / "third_party", device=device)
        arrays: dict[str, Any] = {"key": np.asarray(key)}
        for video in source.videos:
            t = tracks[video.camera_id]
            appearances, _ = embed_tracks(video.path, t.boxes_xyxy, t.observed, source.size, encoder, sampling)
            for track, appearance in zip(t.track_ids.tolist(), appearances, strict=True):
                arrays[f"{video.camera_id}/{track}/frames"] = appearance.frames
                arrays[f"{video.camera_id}/{track}/embeddings"] = appearance.embeddings
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        stored = np.load(path)
    return {camera: tuple(TrackAppearance(stored[f"{camera}/{track}/frames"].astype(np.int64), stored[f"{camera}/{track}/embeddings"])
                          for track in t.track_ids.tolist()) for camera, t in tracks.items()}


def camera_tracks(source: ClipSource, tracks: dict[str, PersonTrackingOutput], cameras: dict[str, PinholeCamera], turns: dict[str, bool],
                  appearances: dict[str, tuple[TrackAppearance, ...]] | None) -> list[CameraTracks]:
    return [CameraTracks(cameras[camera].half_turned(turns[camera]), source.size, tracks[camera].track_ids, tracks[camera].boxes_xyxy,
                         tracks[camera].observed, None if appearances is None else appearances[camera]) for camera in source.camera_ids]


def calibrate(args: argparse.Namespace, dataset: Path, report: Path) -> dict[str, Any]:
    """Pseudo-labelled geometry and appearance calibration on unlabelled observed clips."""
    observe_report = args.observe.resolve()
    observed = json.loads((observe_report / "observe.json").read_text())
    sides = json.loads(args.sides.read_text())
    labelled = {ClipLabels.load(path).clip_id for path in sorted(args.labels_dir.resolve().glob("*/*.json"))}
    if set(args.clip) & labelled:
        raise ValueError(f"Labelled clips {sorted(set(args.clip) & labelled)} must not be used for calibration")
    wanted = args.clip or sorted(set(observed["clips"]) - labelled)
    min_shared = args.min_shared_s
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    config = load_association_config(args.config, players_per_side=args.players_per_side)
    for clip_id in wanted:
        record = observed["clips"][clip_id]
        stopped = [camera for camera, value in record.get("cameras", {}).items() if value["status"] != "ok"]
        if record["status"] != "ok" or stopped:
            skipped[clip_id] = f"observation incomplete (cameras stopped: {stopped})"
            continue
        try:
            turns = side_turns(sides, clip_id)
        except ValueError as error:
            skipped[clip_id] = str(error)
            continue
        source, tracks, cameras = load_observation(dataset, observe_report, clip_id)
        appearances = track_appearances(report / "appearance", source, tracks, args.encoder, args.repo.resolve(), args.device, CropSamplingConfig())
        items = []
        for tracked in camera_tracks(source, tracks, cameras, turns, appearances):
            points, valid = ground_footpoints(tracked.boxes_xyxy, tracked.observed, tracked.camera, source.size[1], config.footpoints)
            assert tracked.appearance is not None
            for row, track in enumerate(tracked.track_ids.tolist()):
                sampled = tracked.appearance[row]
                items.append((tracked.camera.camera_id, track, points[row], valid[row],
                              segment_embedding(sampled.frames, sampled.embeddings, 0, source.num_frames)))
        for a, b in combinations(items, 2):
            if a[0] == b[0]:
                continue
            distance = ground_distance(a[2], a[3], b[2], b[3])
            if distance.shared_frames < min_shared * source.fps:
                continue
            pair = {"clip": clip_id, "a": f"{a[0]}:{a[1]}", "b": f"{b[0]}:{b[1]}", "median_m": distance.median_m,
                    "shared_frames": distance.shared_frames,
                    "cosine": None if a[4] is None or b[4] is None else float(np.clip(a[4] @ b[4], -1, 1))}
            if distance.median_m < args.positive_below_m:
                positives.append(pair)
            elif distance.median_m > args.negative_above_m:
                negatives.append(pair)
        print(json.dumps({"clip": clip_id, "positives": len(positives), "negatives": len(negatives)}), flush=True)
    sigma = rayleigh_scale(np.asarray([pair["median_m"] for pair in positives]))
    slope, center = fit_cosine_log_likelihood_ratio(np.asarray([p["cosine"] for p in positives if p["cosine"] is not None]),
                                                    np.asarray([p["cosine"] for p in negatives if p["cosine"] is not None]))
    result = {"schema": "player_association_calibration_v1", "observe": str(observe_report), "sides": str(args.sides.resolve()),
              "encoder": args.encoder, "sampling": json_value(CropSamplingConfig()),
              "pseudo_labels": {"min_shared_s": min_shared, "positive_below_m": args.positive_below_m, "negative_above_m": args.negative_above_m},
              "excluded_labelled_clips": sorted(labelled), "skipped": skipped,
              "fit": {"geometry_sigma_m": sigma, "appearance_slope": slope, "appearance_center": center,
                      "positives": len(positives), "negatives": len(negatives),
                      "appearance_positives": sum(p["cosine"] is not None for p in positives),
                      "appearance_negatives": sum(p["cosine"] is not None for p in negatives)},
              "pairs": {"positive": positives, "negative": negatives}}
    write_json_atomic(report / "calibration.json", result)
    fit: dict[str, Any] = result["fit"]
    return fit


def plot_clip(path: Path, clip_id: str, points: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Court-plane footpoints by predicted player, by labelled player, and the boxes whose prediction is wrong.

    ``points`` rows: camera, footpoints (T, 2), valid (T,), predicted player (T,), labelled player (T, -1 = none) and
    error (T,): 1 = a wrong player ID, 2 = a labelled player's box excluded (also a tracker's duplicate box, which the
    metrics do not count as an error).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

    figure, axes = plt.subplots(1, 3, figsize=(17, 9), sharex=True, sharey=True)
    palette = plt.get_cmap("tab10")
    markers = {"cam0": "o", "cam1": "s", "cam2": "^"}
    titles = ("predicted player (grey = excluded)", "labelled player (grey = non-player / unmatched)", "wrong player ID (red), excluded player box (orange)")
    for column, (axis, title) in enumerate(zip(axes, titles, strict=True)):
        axis.add_patch(plt.Rectangle((-HALF_DOUBLES_WIDTH, -HALF_LENGTH), 2 * HALF_DOUBLES_WIDTH, 2 * HALF_LENGTH, fill=False, color="black"))
        axis.axhline(0, color="black", lw=.5)
        for camera, xy, valid, predicted, labelled, error in points:
            if column < 2:
                values = (predicted if column == 0 else labelled)[valid]
                colours = [palette(int(v) % 10) if v >= 0 else (.6, .6, .6, .4) for v in values]
            else:
                colours = [{1: (.85, .1, .1, 1.), 2: (1., .55, 0., 1.)}.get(int(e), (.75, .75, .75, .3)) for e in error[valid]]
            axis.scatter(xy[valid, 0], xy[valid, 1], s=2, c=colours, marker=markers.get(camera, "x"))
        axis.set(title=title, aspect="equal", xlim=(-15, 15), ylim=(-25, 25), xlabel="x (m)", ylabel="y (m)")
    axes[0].legend(handles=[Line2D([], [], marker=m, ls="", color="black", label=c) for c, m in markers.items()], loc="upper right")
    figure.suptitle(f"{clip_id}: box-bottom footpoints of every observed box")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def evaluate_clips(args: argparse.Namespace, dataset: Path, report: Path) -> dict[str, Any]:
    """Associate and score every labelled clip; stops are reported per clip with their reason."""
    observe_report = args.observe.resolve()
    sides = json.loads(args.sides.read_text())
    config = load_association_config(args.config, players_per_side=args.players_per_side,
                                     overrides={"appearance": None} if args.geometry_only else None)
    results: dict[str, Any] = {}
    for label_path in sorted(args.labels_dir.resolve().glob("*/*.json")):
        labels = ClipLabels.load(label_path)
        if args.clip and labels.clip_id not in args.clip:
            continue
        source, tracks, cameras = load_observation(dataset, observe_report, labels.clip_id)
        appearances = None if config.appearance is None else track_appearances(
            report / "appearance", source, tracks, config.appearance.encoder, args.repo.resolve(), args.device, CropSamplingConfig())
        tracked = camera_tracks(source, tracks, cameras, side_turns(sides, labels.clip_id), appearances)
        try:
            association = associate(tracked, source.fps, config)
        except AssociationUndecided as stopped:
            results[labels.clip_id] = {"status": "stopped", "reason": stopped.reason, "message": str(stopped), "diagnostics": stopped.diagnostics}
            print(json.dumps({"clip": labels.clip_id, "status": "stopped", "reason": stopped.reason, "message": str(stopped)}), flush=True)
            continue
        predictions = {camera: CameraPrediction(tracks[camera].track_ids, tracks[camera].boxes_xyxy.astype(np.float64), tracks[camera].observed,
                                                association.player_ids[view]) for view, camera in enumerate(association.camera_ids)}
        metrics = evaluate(labels, predictions)
        matched = match_to_labels(labels, predictions, .5)
        roles = labels.roles
        # A labelled player's predicted ID is the one most of its boxes carry; any other ID (or an ID on a non-player) is wrong.
        votes: dict[int, list[int]] = {}
        for view, camera in enumerate(association.camera_ids):
            person, ids = matched[camera], association.player_ids[view]
            for p_index, predicted in zip(person[(person >= 0) & tracks[camera].observed].tolist(),
                                          ids[(person >= 0) & tracks[camera].observed].tolist(), strict=True):
                if roles[p_index] == "player" and predicted >= 0:
                    votes.setdefault(p_index, []).append(predicted)
        expected: np.ndarray = np.full(len(labels.people), -9, np.int64)
        for p_index, values in votes.items():
            expected[p_index] = max(set(values), key=values.count)
        drawn = []
        for view, item in enumerate(tracked):
            camera = item.camera.camera_id
            xy, valid = ground_footpoints(item.boxes_xyxy, item.observed, item.camera, source.size[1], config.footpoints)
            person, ids = matched[camera], association.player_ids[view]
            is_player = (person >= 0) & (roles[np.maximum(person, 0)] == "player")
            player_label = np.where(is_player, person, -1)
            wanted = expected[np.maximum(person, 0)]
            error = np.where(is_player, np.where(ids < 0, 2, (ids != wanted).astype(np.int64)), ((person >= 0) & (ids >= 0)).astype(np.int64))
            for row in range(len(item.track_ids)):
                drawn.append((camera, xy[row], valid[row], ids[row], player_label[row], error[row]))
        plot_clip(report / "figures" / f"{labels.clip_id}.png", labels.clip_id, drawn)
        results[labels.clip_id] = {"status": "ok", "metrics": metrics, "diagnostics": association.diagnostics}
        print(json.dumps({"clip": labels.clip_id, "pair_f1": metrics["pairs"]["f1"], "group_accuracy": metrics["group_accuracy"]["accuracy"],
                          "exclusion": [metrics["exclusion"]["precision"], metrics["exclusion"]["recall"]],
                          "id_switch": [metrics["id_switch"]["precision"], metrics["id_switch"]["recall"]]}), flush=True)
    summary = _summary(results)
    write_json_atomic(report / "evaluate.json", {"schema": "player_association_evaluate_v1", "observe": str(observe_report),
                                                 "sides": str(args.sides.resolve()), "config": json_value(config),
                                                 "config_path": str(args.config.resolve()), "geometry_only": args.geometry_only,
                                                 "summary": summary, "clips": results})
    return summary


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    """Pooled counts over the clips that were not stopped."""
    done = [value["metrics"] for value in results.values() if value["status"] == "ok"]
    def pooled(key: str) -> dict[str, Any]:
        tp, fp, fn = (sum(m[key][name] for m in done) for name in ("tp", "fp", "fn"))
        precision, recall = (tp / (tp + fp) if tp + fp else None), (tp / (tp + fn) if tp + fn else None)
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    frames = sum(m["group_accuracy"]["frames_scored"] for m in done)
    correct = sum(m["group_accuracy"]["frames_correct"] for m in done)
    switches = {name: sum(m["id_switch"][name] for m in done) for name in ("true", "predicted", "matched")}
    return {"clips": len(results), "stopped": {clip: value["reason"] for clip, value in results.items() if value["status"] != "ok"},
            "pairs": pooled("pairs"), "exclusion": pooled("exclusion"),
            "group_accuracy": correct / frames if frames else None, "group_frames": frames,
            "id_switch": {**switches, "precision": switches["matched"] / switches["predicted"] if switches["predicted"] else None,
                          "recall": switches["matched"] / switches["true"] if switches["true"] else None}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Root holding data/, ckpt/ and third_party/")
    parser.add_argument("--dataset", type=Path, required=True, help="Structured dataset directory (dataset.json)")
    parser.add_argument("--report", type=Path, required=True, help="Run-owned directory: player_association/evaluate/<experiment>/<run-id>")
    parser.add_argument("--phase", choices=("observe", "sheets", "labels", "calibrate", "evaluate"), default="observe")
    parser.add_argument("--review", type=Path, help="labels: reviewed track assignment (YAML)")
    parser.add_argument("--labels-dir", type=Path, help="labels: output directory (<video>/<clip>.json)")
    parser.add_argument("--samples", type=int, default=12, help="sheets: crops per track")
    parser.add_argument("--crop-height", type=int, default=160, help="sheets: crop height in pixels")
    parser.add_argument("--clip", action="append", default=[], help="Restrict to clip IDs (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="Extra pipeline.yaml override")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observe", type=Path, help="calibrate/evaluate: report directory of the observe phase")
    parser.add_argument("--sides", type=Path, help="calibrate/evaluate: decisions JSON of court_side_clips.py (reviewed-ball sides)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="calibrate/evaluate: association config YAML")
    parser.add_argument("--players-per-side", type=int, default=1, help="calibrate/evaluate: 1 = singles (every Meiji 3cam clip), 2 = doubles")
    parser.add_argument("--geometry-only", action="store_true", help="evaluate: ignore the appearance section of --config")
    parser.add_argument("--encoder", default="clipreid_vitb16_market1501", help="calibrate: appearance encoder")
    parser.add_argument("--min-shared-s", type=float, default=2., help="calibrate: shared observation time of a pseudo-labelled pair")
    parser.add_argument("--positive-below-m", type=float, default=4., help="calibrate: median distance of a same-person pair")
    parser.add_argument("--negative-above-m", type=float, default=10., help="calibrate: median distance of a different-person pair")
    args = parser.parse_args()
    repo, dataset, report = args.repo.resolve(), args.dataset.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    if args.phase in ("calibrate", "evaluate"):
        if args.observe is None or args.sides is None or args.labels_dir is None:
            parser.error(f"--phase {args.phase} requires --observe, --sides and --labels-dir")
        result = calibrate(args, dataset, report) if args.phase == "calibrate" else evaluate_clips(args, dataset, report)
        print(json.dumps(result), flush=True)
        return
    if args.phase == "labels":
        if args.review is None or args.labels_dir is None:
            parser.error("--phase labels requires --review and --labels-dir")
        print(json.dumps({"labels": labels(dataset, report, args.review.resolve(), args.labels_dir.resolve())}), flush=True)
        return
    manifest = load_dataset_manifest(dataset)
    records = [manifest.clips[key] for key in sorted(manifest.clips) if not args.clip or key in args.clip]
    if args.clip and len(records) != len(args.clip):
        raise ValueError(f"Unknown clip IDs: {sorted(set(args.clip) - {r.clip_id for r in records})}")
    if args.phase == "sheets":
        for record in records:
            written = sheets(dataset / record.path, report / "stores" / record.clip_id, report / "sheets" / record.clip_id,
                             samples=args.samples, crop_height=args.crop_height)
            print(json.dumps({"clip": record.clip_id, "sheets": written}), flush=True)
        return
    runtime, overrides = compose_runtime(repo, report, args.device, args.override, "observe")
    code_identity = TennisSceneOrchestrator(runtime).code_identity
    observed: dict[str, Any] = {"schema": "player_association_observe_v1", "config_overrides": overrides,
                                "code_sha256": code_identity, "clips": {}}
    for record in records:
        try:
            observed["clips"][record.clip_id] = {"status": "ok", **observe(runtime, code_identity, dataset / record.path,
                                                                           report / "stores" / record.clip_id)}
        except (ReconstructionUnavailable, ValueError) as error:
            # A clip whose court cannot be calibrated is reported, never skipped silently.
            observed["clips"][record.clip_id] = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                                 "reason": getattr(error, "reason", None)}
        write_json_atomic(report / "observe.json", observed)
        print(json.dumps({"clip": record.clip_id, "status": observed["clips"][record.clip_id]["status"]}), flush=True)


if __name__ == "__main__":
    main()
