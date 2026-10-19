"""Refuse user customization of the wrapper-owned persistent dsh profile."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.dsh.run_state import RunError, json_bytes, private_write

MANIFEST = ".runner-profile.json"


def fingerprint(home: Path) -> dict[str, str]:
    values = {}
    # Credentials/settings are prohibited even when a copied manifest describes
    # them. The wrapper never opens their contents.
    for name in (
        ".env",
        ".credentials.yaml",
        "settings.yaml",
        "settings.yml",
        "config.json",
        "config.yml",
        "config.yaml",
        "cordis.patch.yml",
        "cordis.patch.yaml",
        "patches",
        "plugins",
    ):
        if (home / name).exists() or (home / name).is_symlink():
            raise RunError(
                "UNSAFE_DSH_HOME",
                "The dedicated DSH_HOME must not contain credentials, .env, settings or custom plugins/patches.",
            )
    profiles = home / "profiles"
    if profiles.is_symlink():
        raise RunError(
            "UNSAFE_DSH_HOME", "The dedicated DSH_HOME profile must not be symlinked."
        )
    if profiles.exists():
        for path in sorted(profiles.rglob("*")):
            if path.is_symlink() or path.relative_to(profiles).parts[0] != "headless":
                raise RunError(
                    "UNSAFE_DSH_HOME",
                    "Only the wrapper-generated headless profile is permitted.",
                )
            info = path.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != (
                0o700 if path.is_dir() else 0o600
            ):
                raise RunError(
                    "UNSAFE_DSH_HOME",
                    "The managed profile must be private (0700 directories, 0600 files) and owned by you.",
                )
            if path.is_file():
                values[str(path.relative_to(home))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return values


def verify_home(home: Path) -> None:
    current = fingerprint(home)
    path = home / MANIFEST
    if not path.exists():
        if current:
            raise RunError(
                "UNSAFE_DSH_HOME",
                "The DSH_HOME contains an unmanaged profile. Use a fresh dedicated external directory.",
            )
        return
    try:
        if (
            path.is_symlink()
            or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077
        ):
            raise ValueError("unsafe manifest")
        saved = json.loads(path.read_bytes())
        if saved != current:
            raise ValueError("profile changed")
    except (OSError, ValueError) as error:
        raise RunError(
            "UNSAFE_DSH_HOME",
            "The wrapper-owned headless profile changed outside the wrapper. Preserve evidence and use a fresh DSH_HOME.",
        ) from error


def remember_home(home: Path) -> dict[str, Any]:
    current = fingerprint(home)
    target = home / MANIFEST
    temporary = target.with_name(target.name + ".tmp-" + uuid4().hex)
    private_write(temporary, json_bytes(current))
    temporary.replace(target)
    return {"profile_sha256": current}
