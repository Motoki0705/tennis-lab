from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.contracts import LocalVideoManifest
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.annotation.reviews import read_local_context
from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ClipManifest
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.jpeg_media import JpegArchive, decode_jpeg
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.storage import StorageCatalog
from src.tasks.base.annotation.workers.contracts import WorkRequest
from tests.support.tasks.local_annotation import local_configuration


@pytest.mark.parametrize("publish_dataset", [False, True])
def test_short_vfr_cameras_retain_original_pixels_timeline_group_and_work(
    tmp_path: Path,
    publish_dataset: bool,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout).model_copy(
        update={"publish_dataset": publish_dataset}
    )
    originals = {
        v.annotation.source.camera_id: Path(v.path).read_bytes() for v in data.videos
    }
    result = configure_local_videos(layout, data)
    receipt = read_document(layout, result, LocalVideoConfigurationReceipt)
    plan = read_document(layout, receipt.plan, LocalAnnotationPlan)
    inputs = read_document(layout, plan.input_manifest, LocalVideoManifest)
    assert plan.publish_dataset is publish_dataset
    assert {v.annotation.source.recording_id for v in inputs.videos} == {"recording-a"}
    assert len(receipt.registrations) == 2
    subjects, work_ids, image_paths = set(), set(), set()
    contexts = []
    for entry in receipt.registrations:
        context = read_document(layout, entry.context, BlankAnnotationContext)
        contexts.append(context)
        expected = read_local_context(layout, entry.context, receipt.plan)
        camera = expected.source.camera_id
        clip = read_document(layout, context.manifest, ClipManifest)
        request = read_document(layout, entry.request, WorkRequest)
        video = next(
            v for v in inputs.videos if v.annotation.source.camera_id == camera
        )
        assert request.subject_id == clip.clip_id
        assert set(request.inputs) == {
            context.task_input,
            context.manifest,
            context.source_media,
        }
        subjects.add(request.subject_id)
        work_ids.add(request.work_id)
        image_paths.add(clip.media.path)
        assert Path(video.path).read_bytes() == originals[camera]
        assert (
            verify_reference(layout, context.source_media).read_bytes()
            == originals[camera]
        )
        assert len(expected.frames) == (3 if camera == "cam0" else 2)
        assert clip.jpeg is not None and len(clip.jpeg.frames) == len(expected.frames)
        # Independent oracle: known VFR times and source presentation indices.
        with av.open(video.path) as source:
            decoded = list(source.decode(video=0))
        assert len(decoded) == 5
        actual_times = [frame.pts * Fraction(frame.time_base) for frame in decoded]
        assert actual_times == [Fraction(t, 16000) for t in (0, 640, 1920, 2560, 4480)]
        archive = JpegArchive(
            verify_reference(layout, clip.media), clip.jpeg, clip.width, clip.height
        )
        for source_index, (mapping, payload) in enumerate(
            zip(clip.frames, archive.iter_frames(), strict=True),
            expected.frame_range.start,
        ):
            original_frame = decoded[source_index]
            assert mapping.source_frame_index == source_index
            assert (
                mapping.source_pts * Fraction(clip.time_base)
                == actual_times[source_index]
            )
            assert (
                mapping.clip_pts * Fraction(clip.time_base)
                == actual_times[source_index] - actual_times[expected.frame_range.start]
            )
            # Metadata equality alone would miss repeated/reordered/cross-camera pixels.
            ok, encoded = cv2.imencode(
                ".jpg",
                original_frame.to_ndarray(format="bgr24"),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            assert ok
            np.testing.assert_array_equal(
                decode_jpeg(payload, clip.width, clip.height),
                cv2.imdecode(encoded, cv2.IMREAD_COLOR),
            )
    assert len(subjects) == len(work_ids) == len(image_paths) == 2
    for context, other in ((contexts[0], contexts[1]), (contexts[1], contexts[0])):
        assert context.source_media != other.source_media
        assert context.manifest != other.manifest
        assert context.task_input != other.task_input
    # Both retained originals and their image archives are protected from storage GC.
    with StorageCatalog(layout).registry.transaction() as db:
        documents = [
            row[0] for row in db.execute("SELECT document FROM storage_protections")
        ]
    assert all(
        any(context.source_media.path in document for document in documents)
        for context in contexts
    )
    assert all(any(path in document for document in documents) for path in image_paths)
    assert configure_local_videos(layout, data) == result
