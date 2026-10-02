"""Repository-owned entry point for local ball annotation campaigns."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)
from src.utils.paths import PROJECT_ROOT

from ..artifacts.configuration import artifact_path_resolver
from .configuration import (
    CampaignConfig,
    ControlConfig,
    campaign_context,
    file_sha256,
    load_config,
)
from .path_contracts import validate_command_paths, validate_initial_paths

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.chat_annotation.local_agent",
    fields=(BoundaryPathField("campaign", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY,
                              must_exist=True, allow_role_root=True),),
)



def initialize(args: argparse.Namespace, campaign_dir: Path | None) -> int:
    from .campaign_state import initial_state
    from .common import atomic_write_json

    for name, value in (('root', args.root), ('campaign', campaign_dir), ('project-root', args.project_root),
                        ('python', args.python), ('codex-home', args.codex_home), ('ball-checkpoint', args.ball_checkpoint)):
        if value is not None and not value.is_absolute():
            raise ValueError(f'--{name} must be an explicit absolute path')
    annotation_root = args.root.resolve()
    if not (annotation_root / "_preparation").is_dir():
        raise ValueError(
            "--root must contain prepared annotation manifests under _preparation/"
        )
    directory = (
        campaign_dir.resolve()
        if campaign_dir is not None
        else annotation_root / "local_agent"
    )
    if directory.exists():
        raise FileExistsError(
            "campaign already exists; use another --campaign or resume it with run"
        )
    checkpoint = (
        args.ball_checkpoint.resolve() if args.ball_checkpoint is not None else None
    )
    codex_home = args.codex_home
    if codex_home is None and os.environ.get("CODEX_HOME"):
        codex_home = Path(os.environ["CODEX_HOME"])
    if codex_home is None:
        codex_home = Path.home() / '.codex'
    if not codex_home.is_absolute():
        raise ValueError('CODEX_HOME must be an absolute path')
    config = CampaignConfig(
        annotation_root=annotation_root,
        campaign_dir=directory,
        project_root=args.project_root.resolve(),
        python_executable=args.python,
        codex_binary=args.codex_binary,
        codex_home=codex_home.resolve(),
        ball_checkpoint=checkpoint,
        ball_checkpoint_sha256=file_sha256(checkpoint)
        if checkpoint is not None
        else None,
    )
    control = ControlConfig(
        model=args.model, effort=args.effort, max_parallel=args.parallel
    )
    validate_initial_paths(config)
    with campaign_context(config):
        state = initial_state(list(control.targets))
        directory.mkdir(parents=True)
        for name in (
            "tasks",
            "logs",
            "qa",
            "cache/timeline",
            "cache/cands_ball",
            "cache/locks",
        ):
            (directory / name).mkdir(parents=True, exist_ok=True)
        for prefix, count in (("timeline_slot", 6), ("model_slot", 4)):
            for index in range(count):
                (config.locks / f"{prefix}_{index}").touch()
        atomic_write_json(directory / "campaign.json", config.model_dump(mode="json"))
        atomic_write_json(config.control, control.model_dump(mode="json"))
        atomic_write_json(config.state, state)
        atomic_write_json(directory / "versions.json", {"labels": {}})
    print(
        json.dumps(
            {
                "campaign": str(directory),
                "tasks": len(state["tasks"]),
                "model": control.model,
                "mode": control.mode,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        help="Explicit campaign directory; required except for init",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="Create an independent campaign without launching any agent"
    )
    init.add_argument(
        "--root", type=Path, required=True, help="Prepared annotation output root"
    )
    init.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT
    )
    init.add_argument("--python", type=Path, default=Path(sys.executable))
    init.add_argument("--codex-binary", default="codex")
    init.add_argument("--codex-home", type=Path)
    init.add_argument("--ball-checkpoint", type=Path)
    init.add_argument("--model", default="gpt-6.1-sol")
    init.add_argument(
        "--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"]
    )
    init.add_argument("--parallel", type=int, default=1)
    for name in (
        "run",
        "status",
        "worker",
        "qa",
        "intake",
        "phase2",
        "prefetch",
        "efficiency",
        "audit",
    ):
        command = sub.add_parser(
            name,
            add_help=False,
            help={
                "run": "Dispatch sandboxed Codex workers",
                "worker": "Per-attempt annotation and image tools",
                "intake": "Archive candidates and record review decisions",
                "phase2": "Rank and compare re-annotation candidates",
            }.get(name, name),
        )
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    launch = sub.add_parser("launch-worker", help=argparse.SUPPRESS)
    launch.add_argument("attempt_dir", type=Path)
    raw = sys.argv[1:] if argv is None else argv
    commands = {
        "run",
        "status",
        "worker",
        "qa",
        "intake",
        "phase2",
        "prefetch",
        "efficiency",
        "audit",
    }
    position = 0
    while position < len(raw):
        if raw[position] == "--campaign":
            position += 2
        elif raw[position].startswith("--campaign="):
            position += 1
        else:
            break
    start = position if position < len(raw) and raw[position] in commands else None
    if start is not None:
        args = parser.parse_args(raw[: start + 1])
        args.arguments = raw[start + 1 :]
    else:
        args = parser.parse_args(raw)
    try:
        if args.command == "init":
            return initialize(args, args.campaign)
        if args.campaign is None:
            if args.command != "launch-worker" and args.arguments in (
                ["--help"],
                ["-h"],
            ):
                module_name = {"worker": "ct", "run": "dispatcher"}.get(
                    args.command, args.command
                )
                module = importlib.import_module(f"{__package__}.{module_name}")
                help_result: int = module.main(args.arguments)
                return help_result
            parser.error("--campaign is required for this command")
        checked = PATH_BOUNDARY.validate({"campaign": args.campaign}, resolver=artifact_path_resolver(args.campaign.resolve()))
        with campaign_context(load_config(checked.declared("campaign").path)):
            validate_command_paths()
            if args.command == "launch-worker":
                from .launcher import supervise

                return supervise(args.attempt_dir)
            module_name = {"worker": "ct", "run": "dispatcher"}.get(
                args.command, args.command
            )
            module = importlib.import_module(f"{__package__}.{module_name}")
            result: int = module.main(args.arguments)
            return result
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
