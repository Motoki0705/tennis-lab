"""Strict TOML job registry for Colab training workflows."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import NAME_RE, SCHEMA_VERSION, WorkflowError, strict_relative_path

KNOWN_HOOKS = frozenset({"base", "submodules", "cuda_ops", "nht"})
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "description",
        "accelerator",
        "timeout_seconds",
        "setup",
        "protected_override_keys",
        "command",
        "inputs",
        "outputs",
        "output_storage",
    }
)
ACCELERATORS = frozenset({"cpu", "gpu"})
_OVERRIDE_SEGMENT = r"[A-Za-z_][A-Za-z0-9_-]*"
OVERRIDE_KEY_RE = re.compile(
    rf"^(?:{_OVERRIDE_SEGMENT}(?:\.{_OVERRIDE_SEGMENT})*|"
    rf"{_OVERRIDE_SEGMENT}(?:/{_OVERRIDE_SEGMENT})+)$"
)


@dataclass(frozen=True)
class InputMapping:
    """A strict Drive-relative source to repository-relative destination."""

    source: str
    destination: str
    writable: bool


@dataclass(frozen=True)
class Job:
    """A validated workflow definition and its immutable digest."""

    name: str
    description: str
    accelerator: str
    timeout_seconds: int
    setup: tuple[str, ...]
    protected_override_keys: tuple[str, ...]
    command_kind: str
    module: str | None
    command_argv: tuple[str, ...]
    default_args: tuple[str, ...]
    inputs: tuple[InputMapping, ...]
    outputs: tuple[str, ...]
    output_storage: str
    definition_path: Path
    definition_digest: str

    def resolved_argv(self, extra_args: list[str]) -> list[str]:
        """Build the remote argv without shell interpretation."""

        if self.command_kind == "python_module":
            assert self.module is not None
            prefix = [".venv/bin/python", "-m", self.module]
        else:
            prefix = list(self.command_argv)
        return [*prefix, *self.default_args, *extra_args]


def _expect_fields(
    value: dict[str, Any], allowed: frozenset[str], required: frozenset[str], label: str
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise WorkflowError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise WorkflowError(f"{label} is missing fields: {sorted(missing)}")


def _string_list(
    value: Any, label: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WorkflowError(f"{label} must be an array of strings")
    if not allow_empty and not value:
        raise WorkflowError(f"{label} must not be empty")
    if any("\x00" in item for item in value):
        raise WorkflowError(f"{label} must not contain NUL bytes")
    if any("/content/drive" in item for item in value):
        raise WorkflowError(f"{label} must not place job data on the Drive FUSE mount")
    return tuple(value)


def load_job(path: Path) -> Job:
    """Load one job definition, rejecting all schema ambiguity."""

    try:
        raw_bytes = path.read_bytes()
        value = tomllib.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise WorkflowError(f"cannot load job definition {path}: {error}") from error
    if not isinstance(value, dict):
        raise WorkflowError(f"job definition must be a TOML table: {path}")
    _expect_fields(
        value, TOP_LEVEL_FIELDS, TOP_LEVEL_FIELDS - {"output_storage"}, str(path)
    )
    output_storage = value.get("output_storage", "local")
    if output_storage not in {"local", "drive"}:
        raise WorkflowError("output_storage must be local or drive")

    if value["schema_version"] != SCHEMA_VERSION:
        raise WorkflowError(
            f"{path}: unsupported schema_version {value['schema_version']!r}"
        )
    name = value["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise WorkflowError(f"{path}: invalid job name {name!r}")
    if name != path.stem:
        raise WorkflowError(f"{path}: job name must match filename stem {path.stem!r}")
    description = value["description"]
    if not isinstance(description, str) or not description.strip():
        raise WorkflowError(f"{path}: description must be a non-empty string")
    accelerator = value["accelerator"]
    if accelerator not in ACCELERATORS:
        raise WorkflowError(f"{path}: accelerator must be 'cpu' or 'gpu'")
    timeout_seconds = value["timeout_seconds"]
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise WorkflowError(f"{path}: timeout_seconds must be an integer")
    if not 60 <= timeout_seconds <= 7 * 24 * 60 * 60:
        raise WorkflowError(f"{path}: timeout_seconds must be between 60 and 604800")

    setup = _string_list(value["setup"], f"{path}: setup", allow_empty=False)
    if setup[0] != "base":
        raise WorkflowError(f"{path}: setup must begin with the base hook")
    if len(set(setup)) != len(setup):
        raise WorkflowError(f"{path}: setup hooks must not repeat")
    unknown_hooks = set(setup) - KNOWN_HOOKS
    if unknown_hooks:
        raise WorkflowError(f"{path}: unknown setup hooks: {sorted(unknown_hooks)}")
    if "cuda_ops" in setup and "submodules" not in setup:
        raise WorkflowError(f"{path}: cuda_ops requires the submodules hook")
    if "nht" in setup and "submodules" not in setup:
        raise WorkflowError(f"{path}: nht requires the submodules hook")
    if "cuda_ops" in setup and setup.index("submodules") > setup.index("cuda_ops"):
        raise WorkflowError(f"{path}: submodules must run before cuda_ops")
    if "nht" in setup and setup.index("submodules") > setup.index("nht"):
        raise WorkflowError(f"{path}: submodules must run before nht")

    protected_override_keys = _string_list(
        value["protected_override_keys"], f"{path}: protected_override_keys"
    )
    if len(set(protected_override_keys)) != len(protected_override_keys):
        raise WorkflowError(f"{path}: protected_override_keys must not repeat")
    for key in protected_override_keys:
        if not OVERRIDE_KEY_RE.fullmatch(key):
            raise WorkflowError(f"{path}: invalid protected override key {key!r}")

    command = value["command"]
    if not isinstance(command, dict):
        raise WorkflowError(f"{path}: command must be a table")
    if "module" in command:
        _expect_fields(
            command,
            frozenset({"module", "args"}),
            frozenset({"module", "args"}),
            f"{path}: command",
        )
        module = command["module"]
        if (
            not isinstance(module, str)
            or not module
            or any(not part.isidentifier() for part in module.split("."))
        ):
            raise WorkflowError(f"{path}: command.module must be a Python module")
        command_kind = "python_module"
        command_argv: tuple[str, ...] = ()
    else:
        _expect_fields(
            command,
            frozenset({"argv", "args"}),
            frozenset({"argv", "args"}),
            f"{path}: command",
        )
        module = None
        command_kind = "argv"
        command_argv = _string_list(
            command["argv"], f"{path}: command.argv", allow_empty=False
        )
        if any(
            Path(item).name in {"bash", "dash", "sh", "zsh"}
            and index + 1 < len(command_argv)
            and command_argv[index + 1] == "-c"
            for index, item in enumerate(command_argv)
        ):
            raise WorkflowError(f"{path}: shell -c commands are forbidden")
    default_args = _string_list(command["args"], f"{path}: command.args")

    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, list):
        raise WorkflowError(f"{path}: inputs must be an array of tables")
    inputs: list[InputMapping] = []
    destinations: set[str] = set()
    for index, mapping in enumerate(raw_inputs):
        if not isinstance(mapping, dict):
            raise WorkflowError(f"{path}: inputs[{index}] must be a table")
        _expect_fields(
            mapping,
            frozenset({"source", "destination", "writable"}),
            frozenset({"source", "destination", "writable"}),
            f"{path}: inputs[{index}]",
        )
        source = strict_relative_path(mapping["source"], f"inputs[{index}].source")
        destination = strict_relative_path(
            mapping["destination"], f"inputs[{index}].destination"
        )
        if destination in destinations:
            raise WorkflowError(f"{path}: duplicate input destination {destination}")
        writable = mapping["writable"]
        if not isinstance(writable, bool):
            raise WorkflowError(f"{path}: inputs[{index}].writable must be a boolean")
        destinations.add(destination)
        inputs.append(
            InputMapping(source=source, destination=destination, writable=writable)
        )

    outputs = tuple(
        strict_relative_path(output, f"outputs[{index}]")
        for index, output in enumerate(
            _string_list(value["outputs"], f"{path}: outputs", allow_empty=False)
        )
    )
    if len(set(outputs)) != len(outputs):
        raise WorkflowError(f"{path}: outputs must not repeat")
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            if first.startswith(f"{second}/") or second.startswith(f"{first}/"):
                raise WorkflowError(
                    f"{path}: output paths must not overlap: {first}, {second}"
                )
    protected_paths = [item.destination for item in inputs]
    for index, first in enumerate(protected_paths):
        for second in protected_paths[index + 1 :]:
            if first.startswith(f"{second}/") or second.startswith(f"{first}/"):
                raise WorkflowError(
                    f"{path}: input destinations must not overlap: {first}, {second}"
                )
    for input_mapping in inputs:
        destination = input_mapping.destination
        overlaps = [
            output
            for output in outputs
            if (
                destination == output
                or destination.startswith(f"{output}/")
                or output.startswith(f"{destination}/")
            )
        ]
        if input_mapping.writable and destination not in outputs:
            raise WorkflowError(
                f"{path}: writable input destination must be declared as an exact "
                f"output: {destination}"
            )
        if not input_mapping.writable and overlaps:
            raise WorkflowError(
                f"{path}: read-only input destination and output must not overlap: "
                f"{destination}, {overlaps[0]}"
            )

    if output_storage == "drive" and (
        default_args.count("paths.output_root=outputs/colab") != 1
        or any(not item.startswith("outputs/colab/") for item in outputs)
    ):
        raise WorkflowError(
            "Drive outputs require paths.output_root=outputs/colab and outputs below it"
        )

    return Job(
        name=name,
        description=description.strip(),
        accelerator=accelerator,
        timeout_seconds=timeout_seconds,
        setup=setup,
        protected_override_keys=protected_override_keys,
        command_kind=command_kind,
        module=module,
        command_argv=command_argv,
        default_args=default_args,
        inputs=tuple(inputs),
        outputs=outputs,
        output_storage=output_storage,
        definition_path=path.resolve(),
        definition_digest=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_registry(directory: Path) -> dict[str, Job]:
    """Load all job definitions and reject duplicate names."""

    if not directory.is_dir():
        raise WorkflowError(f"job registry directory does not exist: {directory}")
    jobs: dict[str, Job] = {}
    for path in sorted(directory.glob("*.toml")):
        job = load_job(path)
        if job.name in jobs:
            raise WorkflowError(f"duplicate job name: {job.name}")
        jobs[job.name] = job
    if not jobs:
        raise WorkflowError(f"job registry is empty: {directory}")
    return jobs
