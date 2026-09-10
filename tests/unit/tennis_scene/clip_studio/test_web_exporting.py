"""Publication is transactional even when replacing an existing annotated clip."""

from dataclasses import replace

import pytest

from src.tennis_scene.clip_studio.export import ExportSettings, plan_clip_export
from src.tennis_scene.clip_studio.web.exporting import publish_export


def test_failed_publication_restores_existing_clip(
    two_camera_project, two_camera_infos, tmp_path, monkeypatch
):
    settings = ExportSettings(tmp_path / "dataset", 30, 64, 36, 17, True)
    plan = plan_clip_export(
        two_camera_project, two_camera_infos, two_camera_project.clips[0], settings
    )
    relative = f"clips/{plan.recording_id}/{plan.clip_name}"
    destination = settings.output_dir / relative
    destination.mkdir(parents=True)
    (destination / "annotation.txt").write_text("existing annotation")
    staging = tmp_path / "staging"
    staged_clip = staging / relative
    staged_clip.mkdir(parents=True)
    (staged_clip / "new.mp4").write_bytes(b"new video")

    def fail(*args, **kwargs):
        raise OSError("index write failed")

    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.exporting.register_exported_clip", fail
    )
    with pytest.raises(OSError, match="index write failed"):
        publish_export(staging, plan, settings)
    assert (destination / "annotation.txt").read_text() == "existing annotation"
    assert not (destination / "new.mp4").exists()
    assert (staged_clip / "new.mp4").read_bytes() == b"new video"
    assert not (staging / "previous-output").exists()
    with pytest.raises(ValueError, match="destination appeared"):
        publish_export(staging, plan, replace(settings, overwrite=False))
    assert (destination / "annotation.txt").is_file()


def test_final_worker_message_arriving_during_exit_check_is_drained(
    two_camera_project, two_camera_infos, tmp_path, monkeypatch
):
    import threading

    from src.tennis_scene.clip_studio.web.exporting import export_interruptibly

    settings = ExportSettings(tmp_path / "dataset", 30, 64, 36, 17, False)
    plan = plan_clip_export(
        two_camera_project, two_camera_infos, two_camera_project.clips[0], settings
    )

    class Receiver:
        checks = 0
        received = False

        def poll(self, timeout=0):
            self.checks += 1
            return self.checks > 1  # final send races the first timeout

        def recv(self):
            if self.received:
                raise EOFError
            self.received = True
            return ("done",)

        def close(self):
            pass

    class Sender:
        def close(self):
            pass

    class Process:
        exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def close(self):
            pass

    receiver = Receiver()

    class Context:
        def Pipe(self, duplex=False):
            return receiver, Sender()

        def Process(self, **kwargs):
            return Process()

    published = []
    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.exporting.multiprocessing.get_context",
        lambda _: Context(),
    )
    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.exporting.publish_export",
        lambda *args: published.append(True),
    )
    export_interruptibly(plan, settings, threading.Event(), lambda *args: None)
    assert receiver.received
    assert published == [True]


def test_successful_worker_waits_for_delayed_shutdown(
    two_camera_project, two_camera_infos, tmp_path, monkeypatch
):
    import threading

    from src.tennis_scene.clip_studio.web.exporting import export_interruptibly

    settings = ExportSettings(tmp_path / "dataset", 30, 64, 36, 17, False)
    plan = plan_clip_export(
        two_camera_project, two_camera_infos, two_camera_project.clips[0], settings
    )

    class Receiver:
        messages = iter((('done',),))

        def poll(self, timeout=0):
            return True

        def recv(self):
            try:
                return next(self.messages)
            except StopIteration as error:
                raise EOFError from error

        def close(self):
            pass

    class Sender:
        def close(self):
            pass

    class Process:
        exitcode = None
        joins = 0

        def start(self):
            pass

        def is_alive(self):
            return self.exitcode is None

        def join(self, timeout=None):
            assert timeout == 0.1
            self.joins += 1
            if self.joins == 3:
                self.exitcode = 0

        def close(self):
            pass

    receiver = Receiver()
    process = Process()

    class Context:
        def Pipe(self, duplex=False):
            return receiver, Sender()

        def Process(self, **kwargs):
            return process

    published = []
    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.exporting.multiprocessing.get_context",
        lambda _: Context(),
    )
    monkeypatch.setattr(
        "src.tennis_scene.clip_studio.web.exporting.publish_export",
        lambda *args: published.append(True),
    )
    export_interruptibly(plan, settings, threading.Event(), lambda *args: None)
    assert process.joins == 3
    assert published == [True]
