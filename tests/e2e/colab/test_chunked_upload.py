"""Exercise the emitted remote assembly code against a simulated Contents API."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.colab.workflow import transfer
from scripts.colab.workflow.common import WorkflowError


@pytest.mark.parametrize("corrupt", [False, True])
def test_chunked_upload_publishes_only_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    monkeypatch.setattr(transfer, "UPLOAD_CHUNK_BYTES", 16)
    source = tmp_path / "source"
    target = tmp_path / "target with 'quotes'"
    payload = bytes(range(113))
    source.write_bytes(payload)
    target.write_bytes(b"previous valid file")
    uploaded_sizes = []

    def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
        stdout = io.StringIO()
        if args[0] == "upload":
            local, remote = Path(args[-2]), Path(args[-1])
            uploaded_sizes.append(local.stat().st_size)
            shutil.copyfile(local, remote)
            if corrupt and len(uploaded_sizes) == 1:
                remote.write_bytes(b"corrupted")
        else:
            assert args[0] == "exec"
            code = Path(args[args.index("-f") + 1]).read_text()
            with contextlib.redirect_stdout(stdout):
                try:
                    exec(compile(code, "assemble.py", "exec"), {})
                except ValueError as error:
                    # Model CLI 0.6.0: a remote exception can still exit 0.
                    print(error)
        return subprocess.CompletedProcess(args, 0, stdout=stdout.getvalue(), stderr="")

    if corrupt:
        with pytest.raises(WorkflowError, match="did not verify"):
            transfer.upload_chunked(source, str(target), session="test", invoke=invoke)
        assert target.read_bytes() == b"previous valid file"
    else:
        transfer.upload_chunked(source, str(target), session="test", invoke=invoke)
        assert target.read_bytes() == payload
        assert not list(tmp_path.glob("*.upload-*"))
    assert uploaded_sizes == [16] * 7 + [1]
