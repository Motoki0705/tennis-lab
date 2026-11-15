"""Resolve repository identity from real local git configuration, never gh."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "global.gitconfig"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    original = subprocess.run

    def local(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args[0] == "git", "repository resolution must never invoke gh or HTTP"
        return cast(subprocess.CompletedProcess[str], original(args, **kwargs))

    monkeypatch.setattr(subprocess, "run", local)
    return importlib.import_module("issues.cli")


def remote(name: str, url: str) -> None:
    subprocess.run(["git", "remote", "add", name, url], check=True)


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:Owner/Repo.git",
        "ssh://git@github.com/Owner/Repo.git",
        "https://github.com/Owner/Repo.git",
        "https://github.com/Owner/Repo",
        "https://GITHUB.COM:443/Owner/Repo.git",
        "ssh://git@github.com:22/Owner/Repo.git",
    ],
)
def test_unique_github_remote_is_local(repo: ModuleType, url: str) -> None:
    remote("origin", url)
    assert repo.GitHub(None).repo == "Owner/Repo"


def test_equivalent_remotes_and_push_urls_share_identity(repo: ModuleType) -> None:
    remote("origin", "git@github.com:Owner/Repo.git")
    remote("upstream", "https://github.com/owner/repo.git")
    subprocess.run(
        [
            "git",
            "config",
            "remote.origin.pushurl",
            "ssh://git@github.com/OWNER/REPO.git",
        ],
        check=True,
    )
    assert repo.GitHub(None).repo.casefold() == "owner/repo"


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:other/repo.git",
        "https://gitlab.com/Owner/Repo.git",
        "/tmp/repository.git",
        "https://github.com:secret-token/Owner/Repo.git",
        "https://user:secret-token@github.com/Owner/Repo.git",
        "https://github.com/Owner/Repo.git?secret-token",
        "https://github.com/Owner/Repo.git#secret-token",
    ],
)
def test_ambiguous_or_non_github_remotes_fail_closed(
    repo: ModuleType, url: str
) -> None:
    remote("origin", "git@github.com:Owner/Repo.git")
    remote("other", url)
    with pytest.raises(ValueError) as error:
        repo.GitHub(None)
    assert "--repo" in str(error.value)
    assert "secret-token" not in str(error.value)
    assert repo.GitHub("explicit/repository").repo == "explicit/repository"


def test_missing_remote_requires_explicit_repo(repo: ModuleType) -> None:
    with pytest.raises(ValueError, match="specify --repo"):
        repo.GitHub(None)


@pytest.mark.parametrize("rule", ["insteadOf", "pushInsteadOf"])
@pytest.mark.parametrize(
    "replacement", ["https://github.com/Other/", "https://gitlab.com/Owner/"]
)
def test_effective_url_rewrites_cannot_hide_another_repository(
    repo: ModuleType, rule: str, replacement: str
) -> None:
    remote("origin", "https://github.com/Owner/Repo.git")
    remote("upstream", "git@github.com:Owner/Repo.git")
    subprocess.run(
        ["git", "config", f"url.{replacement}.{rule}", "https://github.com/Owner/"],
        check=True,
    )
    with pytest.raises(ValueError, match="specify --repo"):
        repo.GitHub(None)
    assert repo.GitHub("explicit/repository").repo == "explicit/repository"


@pytest.mark.parametrize("rule", ["insteadOf", "pushInsteadOf"])
def test_unambiguous_effective_url_rewrites_are_supported(
    repo: ModuleType, rule: str
) -> None:
    if rule == "insteadOf":
        remote("origin", "github:Owner/Repo.git")
        source, target = "github:", "https://github.com/"
    else:
        remote("origin", "https://github.com/Owner/Repo.git")
        source, target = "https://github.com/", "git@github.com:"
    subprocess.run(["git", "config", f"url.{target}.{rule}", source], check=True)
    assert repo.GitHub(None).repo == "Owner/Repo"


@pytest.mark.parametrize("kind", ["url", "pushurl"])
def test_every_effective_fetch_and_push_url_is_checked(
    repo: ModuleType, kind: str
) -> None:
    remote("origin", "https://github.com/Owner/Repo.git")
    if kind == "pushurl":
        subprocess.run(
            ["git", "config", "remote.origin.pushurl", "git@github.com:Owner/Repo.git"],
            check=True,
        )
    subprocess.run(
        ["git", "config", "--add", f"remote.origin.{kind}", "mirror:Other/Repo.git"],
        check=True,
    )
    subprocess.run(
        ["git", "config", "url.https://github.com/.insteadOf", "mirror:"],
        check=True,
    )
    with pytest.raises(ValueError, match="specify --repo"):
        repo.GitHub(None)


def test_effective_url_lookup_failure_is_explicit_and_hides_git_output(
    repo: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote("origin", "https://github.com/Owner/Repo.git")
    original = subprocess.run

    def failed(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["git", "remote", "get-url"]:
            return subprocess.CompletedProcess(args, 2, "secret URL", "secret error")
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(ValueError, match="specify --repo") as error:
        repo.GitHub(None)
    assert "secret" not in str(error.value)


def test_worktree_resolves_shared_git_remote(
    repo: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote("origin", "git@github.com:Owner/Repo.git")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    worktree = tmp_path / "linked"
    subprocess.run(
        ["git", "worktree", "add", "-qb", "linked", str(worktree)],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(worktree)
    assert repo.GitHub(None).repo == "Owner/Repo"
