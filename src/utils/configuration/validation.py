"""Executable negative-validation facilities for strict boundary contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.utils.configuration.errors import (
    ConfigurationError,
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathContractError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.configuration.schema import (
    ConfigField,
    StrictConfigSchema,
    mutually_exclusive,
)

__all__ = ["NegativeValidationCase", "main", "run_negative_validation"]


@dataclass(frozen=True, slots=True)
class NegativeValidationCase:
    """One input which a runtime boundary must reject before doing work."""

    name: str
    invoke: Callable[[], object]
    expected_error: type[ConfigurationError]
    message_fragment: str


def run_negative_validation(cases: Sequence[NegativeValidationCase]) -> tuple[str, ...]:
    """Execute cases and raise AssertionError on acceptance or the wrong failure."""
    completed: list[str] = []
    seen_names: set[str] = set()
    for case in cases:
        if not case.name or case.name in seen_names:
            raise ValueError(
                f"Negative validation case names must be unique: {case.name!r}."
            )
        seen_names.add(case.name)
        try:
            case.invoke()
        except case.expected_error as error:
            if case.message_fragment not in str(error):
                raise AssertionError(
                    f"{case.name}: expected message containing {case.message_fragment!r}, "
                    f"got {str(error)!r}."
                ) from error
        except Exception as error:
            raise AssertionError(
                f"{case.name}: expected {case.expected_error.__name__}, "
                f"got {type(error).__name__}: {error}"
            ) from error
        else:
            raise AssertionError(f"{case.name}: invalid input was accepted.")
        completed.append(case.name)
    return tuple(completed)


def _foundation_cases() -> tuple[NegativeValidationCase, ...]:
    leaf = StrictConfigSchema(name="run", fields={"enabled": ConfigField.of(bool)})
    schema = StrictConfigSchema(
        name="application",
        fields={
            "run": ConfigField.mapping(leaf),
            "resume": ConfigField.of(bool),
            "init_weights": ConfigField.of(bool),
        },
        semantic_checks=(mutually_exclusive("resume", "init_weights"),),
    )
    repository_root = Path.cwd().resolve()
    valid_roots = {f"{role.value}_root": role.value for role in PathRole}
    roots = RuntimePathRoots.from_mapping(valid_roots, repository_root=repository_root)
    resolver = PathResolver(roots)
    return (
        NegativeValidationCase(
            "missing-key",
            lambda: schema.validate({"resume": False, "init_weights": False}),
            MissingConfigurationKeyError,
            "application.run",
        ),
        NegativeValidationCase(
            "unknown-key",
            lambda: schema.validate(
                {
                    "run": {"enabled": True, "typo": True},
                    "resume": False,
                    "init_weights": False,
                }
            ),
            UnknownConfigurationKeyError,
            "application.run.typo",
        ),
        NegativeValidationCase(
            "wrong-exact-type",
            lambda: schema.validate(
                {"run": {"enabled": 1}, "resume": False, "init_weights": False}
            ),
            ConfigurationTypeError,
            "application.run.enabled",
        ),
        NegativeValidationCase(
            "mutually-exclusive",
            lambda: schema.validate(
                {"run": {"enabled": True}, "resume": True, "init_weights": True}
            ),
            SemanticConfigurationError,
            "resume, init_weights",
        ),
        NegativeValidationCase(
            "path-escape",
            lambda: resolver.resolve(PathRole.DATA, "../outside"),
            PathContractError,
            "escapes its declared parent",
        ),
    )


def main() -> int:
    """Run the foundation matrix; domain matrices may call the same runner."""
    completed = run_negative_validation(_foundation_cases())
    print(f"Strict negative validation passed: {', '.join(completed)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - executable validation boundary
    raise SystemExit(main())
