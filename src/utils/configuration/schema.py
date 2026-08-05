"""Strict recursive schemas for configuration runtime boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

from src.utils.configuration.errors import (
    ConfigurationError,
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

SemanticCheck: TypeAlias = Callable[[Mapping[str, object]], None]

__all__ = [
    "ConfigField",
    "ConfigFieldContract",
    "ConfigurationAbsencePolicy",
    "ConfigurationDefaultPolicy",
    "ConfigurationPrecedence",
    "SemanticCheck",
    "StrictConfigSchema",
    "inspect_schema",
    "mutually_exclusive",
]


class ConfigurationDefaultPolicy(StrEnum):
    """Where a runtime field's selected/default value is allowed to originate."""

    COMPOSITION_OWNED = "composition-owned-no-python-default"


class ConfigurationPrecedence(StrEnum):
    """The sole accepted precedence route after configuration composition."""

    COMPOSED_VALUE_ONLY = "composed-value-only-no-fallback-or-alias"


class ConfigurationAbsencePolicy(StrEnum):
    """Observable behavior when a declared execution input is absent."""

    REQUIRED = "required-input-must-be-present"
    OPTIONAL_OMITTED = "optional-input-remains-omitted"
    OPTIONAL_AS_NONE = "optional-input-omitted-as-none"


@dataclass(frozen=True, slots=True)
class ConfigFieldContract:
    """Flattened, executable inspection record for one schema field."""

    path: str
    expected_types: tuple[str, ...]
    required: bool
    absence_policy: ConfigurationAbsencePolicy
    value_constraints: tuple[str, ...]
    default_policy: ConfigurationDefaultPolicy
    precedence_authority: ConfigurationPrecedence


def _type_names(expected: tuple[type[object], ...]) -> str:
    return " | ".join(candidate.__name__ for candidate in expected)


def _display_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


@dataclass(frozen=True, slots=True)
class ConfigField:
    """Declare one exact-typed field in a :class:`StrictConfigSchema`.

    Optional fields may be absent, but no value is synthesized for them. A
    nested mapping or sequence item contract is validated recursively.
    """

    expected_types: tuple[type[object], ...]
    required: bool = True
    mapping_schema: StrictConfigSchema | None = None
    sequence_item: ConfigField | None = None
    default_policy: ConfigurationDefaultPolicy = (
        ConfigurationDefaultPolicy.COMPOSITION_OWNED
    )
    precedence_authority: ConfigurationPrecedence = (
        ConfigurationPrecedence.COMPOSED_VALUE_ONLY
    )

    def __post_init__(self) -> None:
        if not self.expected_types:
            raise ValueError("ConfigField.expected_types must not be empty.")
        if self.default_policy is not ConfigurationDefaultPolicy.COMPOSITION_OWNED:
            raise ValueError(
                "Runtime schema fields may not declare a Python-owned default."
            )
        if self.precedence_authority is not ConfigurationPrecedence.COMPOSED_VALUE_ONLY:
            raise ValueError(
                "Runtime schema fields may not declare fallback or alias precedence."
            )
        if self.mapping_schema is not None and not any(
            issubclass(candidate, Mapping) for candidate in self.expected_types
        ):
            raise ValueError("mapping_schema requires a Mapping expected type.")
        if self.sequence_item is not None and not any(
            issubclass(candidate, Sequence) and candidate not in (str, bytes)
            for candidate in self.expected_types
        ):
            raise ValueError(
                "sequence_item requires a non-string Sequence expected type."
            )

    @classmethod
    def of(
        cls,
        *expected_types: type[object],
        required: bool = True,
    ) -> ConfigField:
        """Build a scalar/container field with one or more accepted exact types."""
        return cls(expected_types=expected_types, required=required)

    @classmethod
    def mapping(
        cls,
        schema: StrictConfigSchema,
        *,
        required: bool = True,
    ) -> ConfigField:
        """Build a recursively validated mapping field."""
        return cls(
            expected_types=(dict,),
            required=required,
            mapping_schema=schema,
        )

    @classmethod
    def sequence(
        cls,
        item: ConfigField,
        *,
        required: bool = True,
    ) -> ConfigField:
        """Build a recursively validated list/tuple field."""
        return cls(
            expected_types=(list, tuple),
            required=required,
            sequence_item=item,
        )

    def validate(self, value: object, *, path: str) -> object:
        """Validate and recursively copy one value."""
        if self.mapping_schema is not None:
            if not isinstance(value, Mapping):
                raise ConfigurationTypeError(
                    f"{path}: expected mapping, got {type(value).__name__}."
                )
            return self.mapping_schema.validate(value, path=path)
        if self.sequence_item is not None:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ConfigurationTypeError(
                    f"{path}: expected non-string sequence, got {type(value).__name__}."
                )
            return tuple(
                self.sequence_item.validate(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if type(value) not in self.expected_types:
            raise ConfigurationTypeError(
                f"{path}: expected {_type_names(self.expected_types)}, "
                f"got {type(value).__name__}."
            )
        return value


@dataclass(frozen=True, slots=True)
class StrictConfigSchema:
    """An exact-key schema with recursive type and semantic validation."""

    fields: Mapping[str, ConfigField]
    semantic_checks: tuple[SemanticCheck, ...] = field(default_factory=tuple)
    name: str = "configuration"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StrictConfigSchema.name must not be empty.")
        invalid_keys = tuple(
            key for key in self.fields if not isinstance(key, str) or not key
        )
        if invalid_keys:
            raise ValueError(
                f"Schema keys must be non-empty strings: {invalid_keys!r}."
            )
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def validate(
        self,
        value: Mapping[str, object],
        *,
        path: str | None = None,
    ) -> Mapping[str, object]:
        """Return an immutable validated copy or raise a precise error."""
        location = self.name if path is None else path
        if not isinstance(value, Mapping):
            raise ConfigurationTypeError(
                f"{location}: expected mapping, got {type(value).__name__}."
            )
        non_string_keys = tuple(key for key in value if not isinstance(key, str))
        if non_string_keys:
            raise ConfigurationTypeError(
                f"{location}: all keys must be strings; got {non_string_keys!r}."
            )
        unknown = sorted(set(value) - set(self.fields))
        if unknown:
            rendered = ", ".join(_display_path(location, key) for key in unknown)
            raise UnknownConfigurationKeyError(
                f"Unknown configuration key(s): {rendered}."
            )
        missing = sorted(
            key
            for key, specification in self.fields.items()
            if specification.required and key not in value
        )
        if missing:
            rendered = ", ".join(_display_path(location, key) for key in missing)
            raise MissingConfigurationKeyError(
                f"Missing required configuration key(s): {rendered}."
            )

        validated = {
            key: self.fields[key].validate(
                raw_value,
                path=_display_path(location, key),
            )
            for key, raw_value in value.items()
        }
        immutable = MappingProxyType(validated)
        for check in self.semantic_checks:
            try:
                check(immutable)
            except ConfigurationError:
                raise
            except (TypeError, ValueError) as error:
                check_name = type(check).__name__
                raise SemanticConfigurationError(
                    f"{location}: semantic check {check_name!r} failed: {error}"
                ) from error
        return immutable

    def inspect(self) -> tuple[ConfigFieldContract, ...]:
        """Return required/optional/default/precedence policy for every field."""
        return inspect_schema(self)


def inspect_schema(schema: StrictConfigSchema) -> tuple[ConfigFieldContract, ...]:
    """Flatten ``schema`` into stable, explicitly auditable field contracts."""
    contracts: list[ConfigFieldContract] = []

    def visit(current: StrictConfigSchema, *, prefix: str) -> None:
        for key, specification in current.fields.items():
            path = _display_path(prefix, key)
            contracts.append(
                ConfigFieldContract(
                    path=path,
                    expected_types=tuple(
                        candidate.__name__ for candidate in specification.expected_types
                    ),
                    required=specification.required,
                    absence_policy=(
                        ConfigurationAbsencePolicy.REQUIRED
                        if specification.required
                        else ConfigurationAbsencePolicy.OPTIONAL_OMITTED
                    ),
                    value_constraints=(
                        "exact-runtime-type",
                        "required-key" if specification.required else "optional-key",
                        *(
                            ("strict-nested-mapping",)
                            if specification.mapping_schema is not None
                            else ()
                        ),
                        *(
                            ("exact-sequence-item-type",)
                            if specification.sequence_item is not None
                            else ()
                        ),
                    ),
                    default_policy=specification.default_policy,
                    precedence_authority=specification.precedence_authority,
                )
            )
            if specification.mapping_schema is not None:
                visit(specification.mapping_schema, prefix=path)

    visit(schema, prefix=schema.name)
    return tuple(contracts)


def mutually_exclusive(*keys: str) -> SemanticCheck:
    """Return a semantic check rejecting more than one truthy direct child key."""
    if len(keys) < 2 or any(not key for key in keys):
        raise ValueError("mutually_exclusive requires at least two non-empty keys.")

    def check(config: Mapping[str, object]) -> None:
        selected = tuple(key for key in keys if key in config and bool(config[key]))
        if len(selected) > 1:
            raise SemanticConfigurationError(
                f"Mutually exclusive configuration values are enabled: {', '.join(selected)}."
            )

    return check
