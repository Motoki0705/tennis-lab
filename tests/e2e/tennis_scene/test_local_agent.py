"""Exercise the packaged workflow with real video I/O and a local fake Codex executable."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray


def test_local_agent_cli_annotation_and_adoption(tmp_path: Path) -> None:
    repository = Path(__file__).parents[3]
    source = tmp_path / "fixture.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        stream.options = {"bf": "0"}
        for index in range(24):
            image: NDArray[np.uint8] = np.zeros((180, 320, 3), dtype=np.uint8)
            image[89:92, 159:162] = (230, 230, 0)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tennis_scene.chat_annotation.scripts.prepare",
            f"paths.data_root={tmp_path}",
            f"paths.output_root={tmp_path / 'output'}",
            "source.local_video=fixture.mp4",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    root = tmp_path / "output" / "chat_annotation"
    campaign = tmp_path / "run with spaces"
    fake = tmp_path / "fake codex"
    fake.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json
import pathlib
import subprocess
import sys

args = sys.argv[1:]
attempt = pathlib.Path(args[args.index('-C') + 1])
last = pathlib.Path(args[args.index('-o') + 1])
task = json.loads((attempt / 'task.json').read_text())
(attempt / 'fake_invocation.json').write_text(json.dumps({'args': args, 'prompt': sys.stdin.read()}))
print(json.dumps({'type': 'thread.started', 'thread_id': 'fake-no-model-session'}), flush=True)
base = [sys.executable, '-m', 'src.tennis_scene.chat_annotation.local_agent', '--campaign', task['campaign_dir'], 'worker']
def tool(*arguments):
    return subprocess.run(base + list(arguments), cwd=attempt, check=True, capture_output=True, text=True).stdout
tool('init', str(attempt))
annotation = json.loads(pathlib.Path(task['annotation']).read_text())
tool('frames', str(attempt), '--start', '0', '--stop', str(annotation['frame_count']), '--scale', '1')
edit = {'status': 'completed', 'issues': [], 'ranges': [{'start': 0, 'stop': annotation['frame_count'], 'set': {'reviewed': True, 'notes': '', 'balls': [{'track_id': 'b1', 'status': 'visible', 'center_px': [160, 90], 'interpolation_frames': None}]}}]}
(attempt / 'work' / 'edits.json').write_text(json.dumps(edit))
tool('apply', str(attempt), '--edits', str(attempt / 'work' / 'edits.json'))
(attempt / 'NOTES.md').write_text('合成動画の全フレームにある中心160,90の球を確認し、注釈を保存した。画像生成・検証を実行。')
message = tool('finish', str(attempt), '--outcome', 'completed', '--summary', '合成動画の全フレームを確認')
last.write_text(message)
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0}}), flush=True)
"""
    )
    fake.chmod(0o755)

    def command(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "src.tennis_scene.chat_annotation.local_agent",
                "--campaign",
                str(campaign),
                *arguments,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    command(
        "init",
        "--root",
        str(root),
        "--codex-binary",
        str(fake),
        "--codex-home",
        str(tmp_path / "fake home"),
    )
    plan = json.loads(command("run", "--dry-run").stdout)
    assert len(plan["candidates"]) == 1
    task_id = plan["candidates"][0]
    command("run", "--once")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        command("run", "--once")
        state = json.loads((campaign / "state.json").read_text())
        if state["tasks"][task_id]["status"] != "running":
            break
        time.sleep(0.1)
    assert state["tasks"][task_id]["status"] == "review", (
        state,
        list(campaign.rglob("stderr.log")),
    )
    attempt = Path(state["tasks"][task_id]["attempts"][-1]["dir"])
    invocation = json.loads((attempt / "fake_invocation.json").read_text())
    assert invocation["args"][invocation["args"].index("-s") + 1] == "workspace-write"
    assert 'approval_policy="never"' in invocation["args"]
    assert "--ignore-user-config" in invocation["args"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in invocation["args"]
    assert "球が手を離れたフレーム" in invocation["prompt"]
    assert list((attempt / "work" / "sheets").glob("*.jpg"))
    accepted = json.loads(
        command(
            "intake", "adopt", "--note", "合成球の全フレームを確認", "--", task_id
        ).stdout
    )
    assert accepted[0]["decision"] == "accepted"
    clip_id = state["tasks"][task_id]["clip_id"]
    assert (
        root / "annotated" / "processed" / "ball" / f"{clip_id}.json"
    ).read_bytes() == (attempt / f"annotation_{clip_id}.json").read_bytes()
    assert len(list((root / "annotated" / "raw").glob("*.zip"))) == 1
    command("run", "--exit-when-idle")
