"""Trusted namespace bootstrap, also executed with isolated system Python.

This file imports only the standard library. No repository/test code runs until
the filesystem, capabilities, cgroup and network have been checked.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def command(*argv: str) -> None:
    subprocess.run(argv, check=True, stdin=subprocess.DEVNULL, close_fds=True)


def bind(source: str, target: Path, *, readonly: bool = True) -> None:
    directory = Path(source).is_dir()
    if directory:
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    # In a user namespace, inherited nested mounts (notably WSL's /usr mounts)
    # are locked together and cannot be stripped by a non-recursive bind.
    command("/usr/bin/mount", "--rbind" if directory else "--bind", source, str(target))
    if readonly:
        readonly_mount(target, recursive=directory)


def readonly_mount(target: Path, *, recursive: bool) -> None:
    class Attributes(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in ("attr_set", "attr_clr", "propagation", "userns_fd")
        ]

    # mount_setattr(2), common syscall number on Linux x86_64/aarch64. Preserve
    # locked flags while adding read-only and nosuid to every nested mount.
    libc = ctypes.CDLL(None, use_errno=True)
    attrs = Attributes(0x1 | 0x2, 0, 0, 0)
    result = libc.syscall(
        442,
        -100,
        ctypes.c_char_p(os.fsencode(target)),
        0x8000 if recursive else 0,
        ctypes.byref(attrs),
        ctypes.sizeof(attrs),
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def membership() -> Path:
    rows = Path("/proc/self/cgroup").read_text().splitlines()
    unified = [row[3:] for row in rows if row.startswith("0::/")]
    if len(unified) != 1:
        raise RuntimeError("a unified cgroup v2 membership is required")
    return Path("/sys/fs/cgroup") / unified[0].lstrip("/")


def bootstrap(config: dict[str, Any]) -> None:
    if os.getpid() != 1:
        raise RuntimeError("bootstrap must be PID 1 in its own PID namespace")
    cgroup = membership()
    if cgroup.name != config["unit"]:
        raise RuntimeError("process did not enter the dedicated systemd scope")
    if int((cgroup / "memory.max").read_text()) != config["memory_bytes"]:
        raise RuntimeError("aggregate memory limit differs from the request")
    if (cgroup / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("swap is not disabled for the scope")
    config["cgroup"] = str(cgroup)
    # Prevent any mount operation from propagating back to the host.
    command("/usr/bin/mount", "--make-rprivate", "/")
    root = Path(config["root"])
    command(
        "/usr/bin/mount",
        "-t",
        "tmpfs",
        "-o",
        "size=64m,nosuid,nodev",
        "tmpfs",
        str(root),
    )
    bind("/usr", root / "usr")
    for name in ("bin", "sbin", "lib", "lib64"):
        source = Path("/") / name
        if source.is_symlink():
            (root / name).symlink_to(os.readlink(source))
        elif source.is_dir():
            bind(str(source), root / name)
    # WSL's GPU libraries can open /dev/dxg through non-CUDA APIs.
    wsl = root / "usr/lib/wsl"
    if wsl.is_dir():
        command(
            "/usr/bin/mount",
            "-t",
            "tmpfs",
            "-o",
            "ro,nosuid,nodev,noexec",
            "tmpfs",
            str(wsl),
        )
    for directory in (
        "etc",
        "proc",
        "dev",
        "dev/shm",
        "tmp",
        "home/sandbox",
        "control",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "tmp").chmod(0o1777)
    (root / "dev/shm").chmod(0o1777)
    (root / "etc/passwd").write_text("root:x:0:0:sandbox:/home/sandbox:/bin/false\n")
    (root / "etc/group").write_text("root:x:0:\n")
    for name in ("ld.so.cache", "localtime"):
        if (Path("/etc") / name).is_file():
            bind(str(Path("/etc") / name), root / "etc" / name)
    for name in ("null", "zero", "random", "urandom"):
        bind("/dev/" + name, root / "dev" / name)
    (root / "dev/fd").symlink_to("/proc/self/fd")
    for name, fd in (("stdin", 0), ("stdout", 1), ("stderr", 2)):
        (root / "dev" / name).symlink_to(f"/proc/self/fd/{fd}")
    for target in config["runtime_targets"]:
        bind(config["runtime"], root / target.lstrip("/"))
    bind(config["checkout"], root / "checkout", readonly=False)
    # The host launcher is hidden by the real, read-only virtualenv mount.
    bind(config["venv"], root / "checkout/.venv")
    if (root / "checkout/.git").is_dir():
        bind(config["checkout"] + "/.git", root / "checkout/.git")
    bind(str(Path(__file__).resolve()), root / "control/entry.py")
    # Blender execution is outside the CPU-test contract.
    blocked = root / "control/blocked"
    blocked.touch(mode=0o000)
    for binary in ("/usr/bin/blender", "/usr/local/bin/blender"):
        if Path(binary).exists():
            bind(str(blocked), root / binary.lstrip("/"))
    command(
        "/usr/bin/mount",
        "-t",
        "proc",
        "-o",
        "ro,nosuid,nodev,noexec",
        "proc",
        str(root / "proc"),
    )
    (root / "control/config.json").write_text(json.dumps(config))
    # Open the attestation before chroot; close it before executing test code.
    fd = os.open(config["attestation"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.set_inheritable(fd, True)
    # No manager/namespace descriptor may provide access back to the host.
    for name in os.listdir("/proc/self/fd"):
        inherited = int(name)
        if inherited > 2 and inherited != fd:
            try:
                os.close(inherited)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    os.chroot(root)
    os.chdir("/checkout")
    os.execve(
        "/usr/bin/setpriv",
        [
            "setpriv",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--no-new-privs",
            "--securebits=+noroot,+noroot_locked",
            "/usr/bin/python3",
            "-I",
            "-S",
            "/control/entry.py",
            "check",
            str(fd),
        ],
        config["environment"],
    )


def inaccessible(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return
    os.close(fd)
    raise RuntimeError(f"host resource is accessible: {path}")


def check(config: dict[str, Any], fd: int) -> None:
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    for key in ("CapEff", "CapPrm", "CapInh", "CapBnd", "CapAmb"):
        if int(status[key].strip(), 16) != 0:
            raise RuntimeError(f"capabilities were not dropped: {key}")
    if status["NoNewPrivs"].strip() != "1" or os.getpid() != 1:
        raise RuntimeError("no-new-privileges / PID containment is missing")
    if os.readlink("/proc/self/ns/net") == config["host_network"]:
        raise RuntimeError("network namespace was not isolated")
    if socket.if_nameindex() != [(1, "lo")]:
        raise RuntimeError("network namespace contains an unexpected interface")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        flags = fcntl.ioctl(sock, 0x8913, struct.pack("256s", b"lo"))
        if struct.unpack_from("H", flags, 16)[0] & 1:
            raise RuntimeError("sandbox loopback is up")
    denied: dict[str, int] = {}
    for host, port in (("192.0.2.1", 9), ("127.0.0.1", config["host_port"])):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            code = sock.connect_ex((host, port))
            if not code:
                raise RuntimeError("network connection escaped the namespace")
            denied[host] = code
    for path in config["hidden_paths"]:
        inaccessible(path)
    allowed_devices = {
        "null",
        "zero",
        "random",
        "urandom",
        "fd",
        "stdin",
        "stdout",
        "stderr",
        "shm",
    }
    if set(os.listdir("/dev")) != allowed_devices:
        raise RuntimeError("unexpected device in sandbox")
    for path in ("/dev/dxg", "/dev/nvidia0", "/dev/nvidiactl", "/dev/dri/renderD128"):
        inaccessible(path)
    if Path("/usr/lib/wsl").exists() and list(Path("/usr/lib/wsl").iterdir()):
        raise RuntimeError("WSL GPU libraries are visible")
    if dict(os.environ) != config["environment"]:
        raise RuntimeError(
            "sandbox environment differs from the credential-free allowlist"
        )
    for path in ("/usr", "/checkout/.venv", *config["runtime_targets"]):
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise RuntimeError(f"host runtime mount is writable: {path}")
    # A dropped root UID must not be able to remount the host runtime writable.
    remount = subprocess.run(
        ["/usr/bin/mount", "-o", "remount,rw", "/usr"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if remount.returncode == 0:
        raise RuntimeError("sandbox can remount the host runtime")
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {
                "verified": True,
                "pid": os.getpid(),
                "cgroup": config["cgroup"],
                "memory_bytes": config["memory_bytes"],
                "swap_bytes": 0,
                "network_denied": denied,
                "capabilities": "none",
                "devices": sorted(allowed_devices),
                "hidden_paths": config["hidden_paths"],
                "environment_names": sorted(os.environ),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    mode = config["mode"]
    if mode == "pytest":
        os.execve(
            "/checkout/.venv/bin/python", config["pytest_argv"], config["environment"]
        )
    if mode == "memory":
        memory_challenge()
    elif mode in {"descendants", "timeout", "closed-output"}:
        child = subprocess.Popen(
            [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-c",
                "import os,time; os.setsid(); print('detached',flush=True); time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert child.stdout is not None
        if child.stdout.readline() != b"detached\n":
            raise RuntimeError("detached descendant did not start")
        print("detached descendant started", flush=True)
        if mode == "closed-output":
            os.close(1)
            os.close(2)
        if mode != "descendants":
            time.sleep(60)
    elif mode != "isolation":
        raise RuntimeError("unknown fixed backend probe")
    print("challenge passed", flush=True)


def memory_challenge() -> None:
    # Each child fits the scope alone; two together exceed the aggregate limit.
    script = (
        "import sys,time; "
        "x=bytearray(128*1024**2); print('allocated',flush=True); time.sleep(float(sys.argv[1]))"
    )
    argv = ["/usr/bin/python3", "-I", "-S", "-c", script]
    single = subprocess.run([*argv, "0"], capture_output=True, check=False)
    if single.returncode or single.stdout != b"allocated\n":
        raise RuntimeError(
            f"single allocation did not fit the limit: exit={single.returncode}; "
            + single.stderr.decode(errors="replace")
        )
    print("single allocation fits", flush=True)
    children = [
        subprocess.Popen([*argv, "1"], stdout=subprocess.PIPE) for _ in range(2)
    ]
    results = [child.communicate(timeout=10) for child in children]
    if not any(child.returncode == -signal.SIGKILL for child in children):
        raise RuntimeError("aggregate memory challenge did not cause an OOM kill")
    print(
        json.dumps(
            {
                "single_fits": True,
                "aggregate_oom": True,
                "exit_codes": [p.returncode for p in children],
                "outputs": [out.decode() for out, _ in results],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if sys.argv[1] == "bootstrap":
        bootstrap(json.loads(Path(sys.argv[2]).read_text()))
    elif sys.argv[1] == "check":
        check(json.loads(Path("/control/config.json").read_text()), int(sys.argv[2]))
    else:
        raise SystemExit("unknown trusted sandbox entrypoint")
