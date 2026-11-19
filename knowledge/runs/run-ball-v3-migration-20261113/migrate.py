"""One-time execution of the frozen #1714 selection through production APIs."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.tasks.ball_detection.annotation.video import (
    BallTarget, FrameIdentity, VideoAnnotationInput, VideoIdentity,
    read_annotation, summarize_annotation, write_annotation,
)
from src.tasks.ball_detection.annotation.video.migration import MigrationEvidence, convert_v2_annotation
from src.tasks.ball_detection.data.windows.active_ball import ActiveBallWindowDataset
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy, MigratedVideoAnnotation, MigrationPublishRequest, SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import BallPublisher
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import load_snapshot
from src.tasks.base.annotation.artifacts import read_document, verify_reference, write_asset, write_document
from src.tasks.base.annotation.collection.local import retain_local_source
from src.tasks.base.annotation.collection.preparation import prepare_frame_range
from src.tasks.base.annotation.contracts import FrameRange, MediaFingerprint, PublicationReceipt, SourceIdentity, content_digest
from src.tasks.base.annotation.media import probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import ProductPlan
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_annotations import BallAnnotationOutput, ImportedBallAnnotationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.imports import BallAnnotationImporter
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec

from inventory import digest, write_once


class NoExecution:
    version = 1

    def assemble(self, context, artifacts):
        raise RuntimeError("migration scene import must remain load-only")


def check_legacy(root, selection, image_inventory, image_baseline):
    base = root / selection["legacy_root"]
    assert digest(base / "dataset.json") == selection["dataset_sha256"]
    for snapshot in selection["snapshots"]:
        directory = root / snapshot["path"]
        assert digest(directory / "metadata.json") == snapshot["metadata_sha256"]
        assert digest(directory / "index.npz") == snapshot["index_sha256"]
    for entry in selection["entries"]:
        directory = base / entry["clip"]
        assert digest(directory / "clip.json") == entry["clip_manifest_sha256"]
        assert digest(directory / "outsource" / f"{entry['camera']}_annotations.json") == entry["original_sha256"]
    hashes = {image["path"]: image for image in image_baseline["files"]}
    assert len(hashes) == len(image_inventory)
    checked = {}
    for image in image_inventory:
        path = root / image["path"]
        stat = path.stat()
        assert stat.st_size == image["size"] == hashes[image["path"]]["size_bytes"]
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in checked:
            checked[identity] = digest(path)
        assert checked[identity] == hashes[image["path"]]["sha256"]


def pinned_input(root, selection, entry):
    directory = root / selection["legacy_root"] / entry["clip"]
    path = (directory / entry["video"]).resolve()
    assert path.stat().st_size == entry["media_size_bytes"]
    assert digest(path) == entry["media_sha256"]
    timeline = probe_video(path)
    expected = VideoAnnotationInput(
        source=VideoIdentity(recording_id=entry["recording_id"], camera_id=entry["camera"],
                             file_name=path.name, media_sha256=entry["media_sha256"],
                             size_bytes=path.stat().st_size, width=timeline.width, height=timeline.height,
                             frame_count=len(timeline.pts), time_base=str(timeline.time_base), nominal_fps=str(timeline.rate)),
        target=BallTarget(target_semantics="active_ball", track_id="1"),
        frame_range=FrameRange(start=0, stop=len(timeline.pts)),
        frames=tuple(FrameIdentity(frame_index=i, pts=p) for i, p in enumerate(timeline.pts)),
    )
    return path, expected, timeline


def check_conversion(original, annotation):
    mapping = {"observed": ("in_frame", "not_occluded"), "interpolated": ("in_frame", "unknown"),
               "occlusion_estimated": ("in_frame", "occluded"), "unresolved": ("unknown", "unknown")}
    examples = {}
    for old, new in zip(original["frames"], annotation.frames, strict=True):
        assert (new.state, new.occlusion) == mapping[old["status"]]
        assert (new.frame_index, new.pts, new.break_before, new.notes) == (
            old["frame_index"], old["pts"], old["break_before"], old.get("notes", ""))
        point = new.position
        assert (None if point is None else {"x": point.x, "y": point.y}) == old["center_px"]
        if point is not None:
            assert point.provenance.kind == old["status"]
            assert list(point.provenance.support_frames) == old["support_frames"]
        examples.setdefault(old["status"], dict(frame=new.frame_index, pts=new.pts, v2=old["status"],
                                               state=new.state, occlusion=new.occlusion,
                                               position=None if point is None else point.model_dump(mode="json")))
    return examples


def scene_import(layout, prepared, output):
    groups = defaultdict(list)
    for entry, expected, saved in prepared:
        groups[entry["clip"]].append((entry, expected, saved))
    results = []
    for clip, inputs in sorted(groups.items()):
        assert len(inputs) == 3
        videos = tuple(SourceVideo(e["camera"], verify_reference(layout, saved.source_media),
                                   expected.source.media_sha256, len(expected.frames), float(Fraction(expected.source.nominal_fps)),
                                   expected.source.width, expected.source.height) for e, expected, saved in inputs)
        source = ClipSource(clip_id=clip, videos=videos)
        nodes = tuple(ComponentNode(f"ball_detection/{v.camera_id}", ImportedBallAnnotationModule(), ImportedBallAnnotationModule.io,
                                    NoExecution(), {}, AssemblyContext(source, v.camera_id), {}, "saved_v3_migration", "load") for v in videos)
        importer = BallAnnotationImporter({e["camera"]: verify_reference(layout, saved.annotation) for e, expected, saved in inputs},
                                          {e["camera"]: expected for e, expected, saved in inputs})
        store = ClipStore(output / clip, json_value(source))
        refs = importer.publish(nodes, store, source)
        assert importer.publish(nodes, store, source) == refs
        runner = ComponentRunner(nodes, store)
        runner.run()
        assert set(runner.statuses.values()) == {"loaded"}
        reloaded = ClipStore(output / clip, json_value(source))
        for e, expected, saved in inputs:
            value = reloaded.load(refs[f"ball_detection/{e['camera']}"], ArtifactCodec(BallAnnotationOutput))
            np.testing.assert_array_equal(value.pts, [f.pts for f in expected.frames])
            counts = e["counts"]
            assert int(value.observed.sum()) == counts["observed"]
            assert int(value.estimated.sum()) == counts["interpolated"] + counts["occlusion_estimated"]
            assert int(value.position_usable.sum()) == len(expected.frames) - counts["unresolved"]
        results.append(dict(clip=clip, references={k: asdict(v) for k, v in refs.items()}))
    return results


def verify_snapshot(layout, receipt_ref, prepared):
    receipt = read_document(layout, receipt_ref, PublicationReceipt)
    snapshot, store = load_snapshot(layout, receipt.snapshot)
    by_input = {(expected.source.media_sha256, expected.source.camera_id): (e, expected, saved) for e, expected, saved in prepared}
    assert len(store.clips) == len(prepared)
    assert len(store) == sum(e["frames"] for e, _, _ in prepared)
    assert len(receipt.added_clips) == len(prepared) and not receipt.reused_clips and not receipt.adoptions
    assert set(receipt.annotations) == {saved.annotation for _, _, saved in prepared}
    splits = Counter()
    kinds = Counter()
    for clip in store.clips:
        annotation = store.video_annotations[clip.clip_id]
        entry, expected, saved = by_input[(annotation.source.media_sha256, annotation.source.camera_id)]
        assert annotation == read_annotation(verify_reference(layout, saved.annotation), expected_input=expected, require_complete=True)
        assert clip.group_id == content_digest(entry["recording_id"])
        assert clip.split == entry["split"]
        rows = store.clip_rows(clip)
        positive = np.array([f.position is not None for f in annotation.frames])
        np.testing.assert_array_equal(store.frames["pts"][rows], [f.pts for f in expected.frames])
        np.testing.assert_array_equal(store.teacher_masks.coordinate[rows], positive.astype(np.uint8))
        np.testing.assert_array_equal(store.teacher_masks.heatmap[rows], 2 * positive.astype(np.uint8))
        splits[clip.split] += clip.frame_count
        kinds.update(entry["counts"])
    samples = []
    for split in sorted(splits):
        dataset = ActiveBallWindowDataset(store, split=split, window_length=1, stride=1)
        assert len(dataset) == splits[split]
        seen = set()
        for index, (clip, frame_index) in enumerate(dataset.windows):
            row = store.video_annotations[clip.clip_id].frames[frame_index]
            kind = row.position.provenance.kind if row.position else "unresolved"
            if kind in seen:
                continue
            seen.add(kind)
            sample = dataset[index]
            assert sample["frame_indices"].item() == row.frame_index
            assert sample["pts"].item() == row.pts
            assert not sample["active_ball_negative"].item()
            assert sample["position_weight"].item() == int(row.position is not None)
            if row.position:
                np.testing.assert_array_equal(sample["xy"].numpy()[0], np.array([row.position.x, row.position.y], np.float32))
            samples.append(dict(split=split, kind=kind, frame=row.frame_index, pts=row.pts, shape=list(sample["rgb"].shape)))
    return dict(snapshot=receipt.snapshot.model_dump(mode="json"), clips=len(snapshot.clips), frames=len(store),
                split_frames=dict(splits), counts=dict(kinds), learning_samples=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--legacy-image-baseline", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-clip")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_bytes())
    images = json.loads(args.selection.with_name("legacy-images.json").read_bytes())
    image_baseline = json.loads(args.legacy_image_baseline.read_bytes())
    assert image_baseline["schema_version"] == "ball.relocated-legacy-images.v1"
    assert image_baseline["previous_image_inventory_sha256"] == digest(args.selection.with_name("legacy-images.json"))
    assert selection["schema_version"] == "ball.v3-migration-selection.v1"
    if args.apply:
        if not args.version or not args.code_commit:
            parser.error("apply requires a new dataset version and fixed code commit")
        subprocess.run(["git", "diff", "--exit-code", args.code_commit, "--", "src", "tests", str(Path(__file__).resolve().relative_to(Path.cwd()))], check=True)
    check_legacy(args.source_root, selection, images, image_baseline)
    selected = [e for e in selection["entries"] if e["decision"] == "convert" and (args.only_clip is None or e["key"].rsplit("/", 1)[0] == args.only_clip)]
    assert selected and len(selected) % 3 == 0
    storage = args.storage_root.resolve()
    layout = AnnotationLayout(storage, storage / "outputs/chat_annotation", "production")
    if (storage / "data").stat().st_dev != layout.root.stat().st_dev:
        raise ValueError("annotation images and publication destination must share a filesystem")
    prepared, results = [], []
    for number, entry in enumerate(selected, 1):
        directory = args.source_root / selection["legacy_root"] / entry["clip"]
        original_path = directory / "outsource" / f"{entry['camera']}_annotations.json"
        original = json.loads(original_path.read_bytes())
        path, expected, timeline = pinned_input(args.source_root, selection, entry)
        evidence = MigrationEvidence(original_sha256=entry["original_sha256"], target_semantics="active_ball",
                                     target_evidence="retained original #/target: main-court rally ball; adjacent balls excluded",
                                     processing_complete=True, processing_evidence=entry["processing_evidence"])
        annotation, record = convert_v2_annotation(original_path, expected_input=expected, evidence=evidence)
        examples = check_conversion(original, annotation)
        result = dict(key=entry["key"], summary=asdict(summarize_annotation(annotation)), examples=examples)
        if args.apply:
            root = layout.path("migration_archive", "ball-v2-v3", entry["original_sha256"])
            source_media = retain_local_source(layout, path, MediaFingerprint(digest=expected.source.media_sha256, size_bytes=expected.source.size_bytes))
            manifest = prepare_frame_range(layout, SourceIdentity(provider="local", channel_id=content_digest(entry["recording_id"]), video_id=expected.source.sha256,
                                                                  media=source_media, origin_url=path.as_uri()), timeline,
                                           source_range=expected.frame_range, target_range=FrameRange(start=0, stop=len(expected.frames)),
                                           preparation_digest=expected.sha256, retained_media="jpeg_q90")
            saved = MigratedVideoAnnotation(
                expected_input=write_document(layout, root / "input.json", expected),
                original=write_asset(layout, root / "original-v2.json", original_path.read_bytes()),
                migration=write_document(layout, root / "migration.json", record),
                annotation=write_annotation(layout, root / "annotation-v3.json", annotation, expected_input=expected, require_complete=True),
                manifest=manifest, source_media=source_media,
            )
            prepared.append((entry, expected, saved))
            result["references"] = saved.model_dump(mode="json")
        results.append(result)
        print(json.dumps(dict(prepared=number, total=len(selected), key=entry["key"], frames=len(expected.frames))), flush=True)
    publication = None
    scenes = []
    if args.apply:
        groups = {e["recording_id"]: e["split"] for e, _, _ in prepared}
        policy = write_document(layout, layout.path("migration_plans", args.version, "publication-policy.json"), BallPublicationPolicy(
            jpeg_quality=90, target_semantics="active_ball", source_splits=tuple(SourceSplit(source_id=content_digest(group), split=split) for group, split in sorted(groups.items())),
        ))
        request = MigrationPublishRequest(output=ProductPlan(product="frames", dataset_task="ball_detection", dataset_version=args.version, component="frames",
                                                            base_snapshot=None, generation_policy=policy),
                                          annotations=tuple(sorted((s for _, _, s in prepared), key=lambda s: s.expected_input.digest)))
        request_ref = write_document(layout, layout.path("migration_plans", args.version, "publication-request.json"), request)
        print("publishing fixed selection", flush=True)
        reference = BallPublisher(layout).publish(request)
        print("verifying same-request replay", flush=True)
        assert BallPublisher(layout).publish(request) == reference
        publication = verify_snapshot(layout, reference, prepared)
        publication.update(receipt=reference.model_dump(mode="json"), request=request_ref.model_dump(mode="json"), replay_equal=True)
        print("importing saved annotations into scene stores", flush=True)
        scenes = scene_import(layout, prepared, storage / "outputs/ball_detection/migrate/video-v3/20261113" / args.version / "scene-imports")
    check_legacy(args.source_root, selection, images, image_baseline)
    report = dict(schema_version="ball.v3-migration-result.v1", code_commit=args.code_commit,
                  selection_sha256=digest(args.selection), selected=len(selected), entries=results,
                  publication=publication, scene_imports=scenes,
                  legacy_unchanged=dict(annotation_files=len(selection["entries"]), metadata_and_indexes=2 * len(selection["snapshots"]),
                                        image_stats_checked=len(images), image_payload_hashes_recomputed=True,
                                        image_baseline_sha256=digest(args.legacy_image_baseline)))
    write_once(args.output, report)
    print(json.dumps(dict(report=str(args.output), cameras=len(selected), frames=sum(e["frames"] for e in selected),
                          publication=publication, scene_clips=len(scenes))), flush=True)


if __name__ == "__main__":
    main()
