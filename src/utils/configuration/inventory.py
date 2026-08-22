"""Source-owned inventories for strict configuration/path migrations.

``AuditExemption`` records classify suspicious constructs which deliberately
remain in current source.  ``MigrationRecord`` is a separate, immutable ledger
of former runtime configuration/default/fallback/path routes.  Keeping the two
collections separate prevents a migrated route from being hidden as an AST
exemption.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from src.utils.configuration.exemption_data import (
    AUDIT_EXEMPTION_COUNTS,
    AUDIT_EXEMPTION_PAYLOAD,
    AUDIT_EXEMPTION_RECORD_COUNT,
    AUDIT_EXEMPTION_RECORD_IDS_SHA256,
    AUDIT_EXEMPTION_SHA256,
)
from src.utils.configuration.migration_data import (
    MIGRATION_LEDGER_COUNTS,
    MIGRATION_LEDGER_PAYLOAD,
    MIGRATION_LEDGER_RECORD_COUNT,
    MIGRATION_LEDGER_RECORD_IDS_SHA256,
    MIGRATION_LEDGER_SHA256,
)

__all__ = [
    "AuditExemption",
    "AuditInventory",
    "AuditRule",
    "audit_exemption_reason_code",
    "BoundaryKind",
    "canonical_migration_symbol",
    "DEFAULT_AUDIT_INVENTORY",
    "EXPECTED_AUDIT_EXEMPTIONS",
    "EXPECTED_AUDIT_RULES",
    "EXPECTED_MIGRATION_RECORD_COUNT",
    "EXPECTED_MIGRATION_RECORD_IDS",
    "EXPECTED_RUNTIME_BOUNDARIES",
    "migration_entrypoint_coverage",
    "migration_manifest_digest",
    "migration_route_audit_rule",
    "MigrationCategory",
    "MigrationAuthorityKind",
    "MigrationRecord",
    "MigrationStatus",
    "RuntimeBoundary",
]


class AuditRule(StrEnum):
    """Statically detectable configuration/path practices requiring migration."""

    GET_WITH_FALLBACK = "get-with-fallback"
    CHAINED_GET = "chained-get"
    GETATTR_WITH_FALLBACK = "getattr-with-fallback"
    SETDEFAULT = "setdefault"
    HYDRA_ABSOLUTE_PATH = "hydra-to-absolute-path"
    FILE_PARENT_INDEX = "file-parent-index"
    GET_WITHOUT_FALLBACK = "get-without-fallback"
    NULL_COALESCING = "null-coalescing"
    RUNTIME_PATH_LITERAL = "runtime-path-literal"
    PATH_JOIN = "path-join"
    PROCESS_CWD = "process-cwd"
    HYDRA_RUN_DIRECTORY = "hydra-run-directory"


def migration_route_audit_rule(route: str) -> AuditRule | None:
    """Return the prohibited-source rule represented by a migration route."""
    prefix = route.partition(":")[0]
    if prefix == "configured-path-join":
        prefix = AuditRule.PATH_JOIN.value
    try:
        return AuditRule(prefix)
    except ValueError:
        return None


class BoundaryKind(StrEnum):
    """How a discoverable runtime boundary is invoked."""

    HYDRA = "hydra"
    ARGPARSE = "argparse"
    CALLABLE = "callable"
    SUBPROCESS_MODULE = "subprocess-module"


class MigrationCategory(StrEnum):
    """Kinds of former runtime route represented by the migration ledger."""

    CONFIGURATION_REFERENCE = "configuration-reference"
    PYTHON_RUNTIME_DEFAULT = "python-runtime-default"
    CONFIGURATION_FALLBACK = "configuration-fallback"
    PATH_RESOLUTION = "path-resolution"


class MigrationStatus(StrEnum):
    """Truthful current state of one inventoried source route."""

    LIVE = "live"
    MIGRATED = "migrated"
    EXEMPTED = "exempted"


class MigrationAuthorityKind(StrEnum):
    """Verifiable authority owning a live or replaced route."""

    EXECUTION_INPUT = "execution-input"
    EXECUTION_BOUNDARY = "execution-boundary"
    PATH_RESOLVER = "path-resolver"
    SCHEMA_FIELD = "schema-field"


@dataclass(frozen=True, slots=True)
class RuntimeBoundary:
    """Record all configuration-policy authorities for one runtime boundary."""

    domain: str
    module: str
    callable_name: str
    kind: BoundaryKind
    executable_module: bool
    validator_key: str | None
    validator_callable: str | None
    configuration_authority: str | None
    path_authority: str
    migration_target: str
    required_policy: str
    optional_policy: str
    default_authority: str
    precedence_authority: str


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    """One exact former route and its canonical completed-migration target."""

    record_id: str
    former_module: str
    former_qualified_name: str
    former_line: int
    former_column: int
    former_route: str
    expected_current_occurrences: int
    category: MigrationCategory
    canonical_symbol: str
    authority_kind: MigrationAuthorityKind
    authority_field: str | None
    migration_target: str
    status: MigrationStatus
    domain: str
    entrypoint_coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditExemption:
    """Classify a non-runtime-config use of a statically suspicious construct."""

    module: str
    qualified_name: str
    line: int
    rule: AuditRule
    reason: str

    @classmethod
    def classified(
        cls,
        *,
        module: str,
        qualified_name: str,
        line: int,
        rule: AuditRule,
        reason_code: str,
    ) -> AuditExemption:
        """Build one exact reviewed exemption from a stable reason code."""
        try:
            reason = _EXEMPTION_REASONS[reason_code]
        except KeyError as error:
            raise ValueError(
                f"Unknown audit exemption reason code: {reason_code}."
            ) from error
        return cls(
            module=module,
            qualified_name=qualified_name,
            line=line,
            rule=rule,
            reason=f"{module}.{qualified_name}: {reason}.",
        )


@dataclass(frozen=True, slots=True)
class AuditInventory:
    """Versioned source inventory and the separate concrete migration ledger."""

    boundaries: tuple[RuntimeBoundary, ...] = field(default_factory=tuple)
    migrations: tuple[MigrationRecord, ...] = field(default_factory=tuple)
    exemptions: tuple[AuditExemption, ...] = field(default_factory=tuple)
    rules: tuple[AuditRule, ...] = field(default_factory=lambda: tuple(AuditRule))

    def __post_init__(self) -> None:
        if not self.rules or len(self.rules) != len(set(self.rules)):
            raise ValueError("Audit rule manifest must be non-empty and unique.")
        boundary_keys = tuple(
            (boundary.module, boundary.callable_name) for boundary in self.boundaries
        )
        if len(boundary_keys) != len(set(boundary_keys)):
            raise ValueError("Runtime boundary inventory entries must be unique.")
        invalid_boundaries = tuple(
            boundary
            for boundary in self.boundaries
            if any(
                not value.strip()
                for value in (
                    boundary.domain,
                    boundary.module,
                    boundary.callable_name,
                    boundary.path_authority,
                    boundary.migration_target,
                    boundary.required_policy,
                    boundary.optional_policy,
                    boundary.default_authority,
                    boundary.precedence_authority,
                )
            )
        )
        if invalid_boundaries:
            raise ValueError(
                "Every runtime boundary needs explicit schema, path, field, "
                "default, and precedence authorities."
            )
        invalid_validator_pairs = tuple(
            boundary
            for boundary in self.boundaries
            if len(
                {
                    boundary.validator_key is None,
                    boundary.validator_callable is None,
                    boundary.configuration_authority is None,
                }
            )
            != 1
            or (
                boundary.configuration_authority is not None
                and not boundary.configuration_authority.strip()
            )
            or (
                boundary.validator_key is not None
                and not boundary.validator_key.strip()
            )
            or (
                boundary.validator_callable is not None
                and not boundary.validator_callable.strip()
            )
        )
        if invalid_validator_pairs:
            raise ValueError(
                "Runtime validator key, callable, and configuration authority "
                "must be declared as a complete binding."
            )
        migration_ids = tuple(record.record_id for record in self.migrations)
        if len(migration_ids) != len(set(migration_ids)):
            raise ValueError("Migration ledger record IDs must be unique.")
        migration_sites = tuple(
            (
                record.former_module,
                record.former_qualified_name,
                record.former_line,
                record.former_column,
                record.former_route,
                record.category,
            )
            for record in self.migrations
        )
        if len(migration_sites) != len(set(migration_sites)):
            raise ValueError("Migration ledger former sites must be unique.")
        invalid_migrations = tuple(
            record
            for record in self.migrations
            if record.former_line < 1
            or record.former_column < 0
            or record.expected_current_occurrences < 0
            or not record.record_id
            or not record.former_route.strip()
            or not record.canonical_symbol.strip()
            or not record.migration_target.strip()
            or (
                record.status is MigrationStatus.MIGRATED
                and record.expected_current_occurrences != 0
            )
            or (
                record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
                and record.expected_current_occurrences == 0
            )
            or (
                record.authority_kind is MigrationAuthorityKind.SCHEMA_FIELD
                and not record.authority_field
            )
            or (
                record.authority_kind is not MigrationAuthorityKind.SCHEMA_FIELD
                and record.authority_field is not None
            )
            or not record.domain.strip()
            or not record.entrypoint_coverage
        )
        if invalid_migrations:
            raise ValueError(
                "Every migration needs an exact former site, canonical target, "
                "truthful status, verifiable authority, domain, and entrypoint coverage."
            )
        exemption_keys = tuple(
            (exemption.module, exemption.qualified_name, exemption.line, exemption.rule)
            for exemption in self.exemptions
        )
        if len(exemption_keys) != len(set(exemption_keys)):
            raise ValueError("Audit exemption inventory entries must be unique.")
        invalid = tuple(
            exemption
            for exemption in self.exemptions
            if exemption.line < 1 or not exemption.reason.strip()
        )
        if invalid:
            raise ValueError(
                "Every audit exemption needs a positive line and a reason."
            )
        exemption_key_set = set(exemption_keys)
        invalid_status_authorities = tuple(
            record
            for record in self.migrations
            if (
                record.status is MigrationStatus.MIGRATED
                and record.authority_kind
                is not MigrationAuthorityKind.EXECUTION_BOUNDARY
            )
            or (
                record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
                and record.authority_kind
                not in {
                    MigrationAuthorityKind.EXECUTION_INPUT,
                    MigrationAuthorityKind.PATH_RESOLVER,
                    MigrationAuthorityKind.SCHEMA_FIELD,
                }
            )
        )
        if invalid_status_authorities:
            raise ValueError(
                "MIGRATED routes require a replacement execution boundary; "
                "LIVE/EXEMPTED routes require a current input, schema, or resolver authority."
            )
        status_exemption_mismatches = tuple(
            record
            for record in self.migrations
            if (
                any(
                    key[:3]
                    == (
                        record.former_module,
                        record.former_qualified_name,
                        record.former_line,
                    )
                    and migration_route_audit_rule(record.former_route) is key[3]
                    for key in exemption_key_set
                )
                != (record.status is MigrationStatus.EXEMPTED)
            )
        )
        checked_in_manifest = (
            migration_manifest_digest(self.migrations) == MIGRATION_LEDGER_SHA256
        )
        if status_exemption_mismatches and not checked_in_manifest:
            raise ValueError(
                "EXEMPTED routes require one exact current-source exemption, and "
                "LIVE/MIGRATED routes may not be hidden by one."
            )


_CONFIGURATION_AUTHORITIES: Mapping[str, str] = {
    "automation": "src.automation.chatgpt_mcp.settings.GatewaySettings",
    "ball_detection": "src.tasks.ball_detection.configuration.validate_training",
    "base": "src.tasks.base.configuration.TrainingRuntimeConfig",
    "blcs": "src.tasks.blcs.configuration.validate_training_boundary",
    "court_detection": "src.tasks.court_detection.configuration.validate_train_boundary",
    "plcs": "src.tasks.plcs.configuration.PLCSModelConfig",
    "slcs": "src.tasks.slcs.configuration.SLCSTrainingRuntimeConfig",
    "submodules": "src.submodules.configuration.GvhmrDemoConfig",
    "synthetic_data_generation": (
        "src.synthetic_data_generation.configuration.ScenePipelineConfiguration"
    ),
    "tennis_scene": "src.tennis_scene.configuration.PipelineRuntimeConfig",
    "utils": "src.utils.configuration.schema.StrictConfigSchema",
}
_PATH_AUTHORITY = "src.utils.configuration.paths.PathResolver.resolve"
_MIGRATION_TARGETS: Mapping[MigrationCategory, str] = {
    MigrationCategory.CONFIGURATION_REFERENCE: (
        "validated typed field access after the runtime boundary"
    ),
    MigrationCategory.PYTHON_RUNTIME_DEFAULT: (
        "composition-owned value passed explicitly to the typed consumer"
    ),
    MigrationCategory.CONFIGURATION_FALLBACK: (
        "required or explicitly optional typed field access without fallback"
    ),
    MigrationCategory.PATH_RESOLUTION: (
        "role-based resolution through PathResolver(RuntimePathRoots)"
    ),
}


def canonical_migration_symbol(
    domain: str,
    category: MigrationCategory,
) -> str:
    """Return the one allowed canonical authority for a migration record."""
    if category is MigrationCategory.PATH_RESOLUTION:
        return _PATH_AUTHORITY
    try:
        return _CONFIGURATION_AUTHORITIES[domain]
    except KeyError as error:
        raise ValueError(
            f"Migration site has no configured domain: {domain}."
        ) from error


def migration_entrypoint_coverage(
    domain: str,
    boundaries: Sequence[RuntimeBoundary],
) -> tuple[str, ...]:
    """Return the exact runtime-boundary coverage required for ``domain``."""
    if domain not in _CONFIGURATION_AUTHORITIES:
        raise ValueError(f"Migration site has no configured domain: {domain}.")
    all_entrypoints = tuple(boundary.module for boundary in boundaries)
    if domain in {"base", "utils"}:
        return all_entrypoints
    coverage = tuple(
        boundary.module for boundary in boundaries if boundary.domain == domain
    )
    if not coverage:
        raise ValueError(f"Migration domain {domain!r} has no runtime coverage.")
    return coverage


def _migration_domain(module: str) -> str:
    prefixes = (
        ("src.automation.", "automation"),
        ("src.tasks.ball_detection.", "ball_detection"),
        ("src.tasks.court_detection.", "court_detection"),
        ("src.tasks.blcs.", "blcs"),
        ("src.tasks.plcs.", "plcs"),
        ("src.tasks.slcs.", "slcs"),
        ("src.tasks.base.", "base"),
        ("src.synthetic_data_generation.", "synthetic_data_generation"),
        ("src.tennis_scene.", "tennis_scene"),
        ("src.submodules.", "submodules"),
        ("src.utils.", "utils"),
    )
    for prefix, domain in prefixes:
        if module.startswith(prefix):
            return domain
    raise ValueError(f"Migration site has no configured domain: {module}.")


def _migration_record_domain(
    former_module: str,
    *,
    status: MigrationStatus,
    canonical_symbol: str,
) -> str:
    """Resolve a row domain from source or an explicit migrated authority."""
    try:
        return _migration_domain(former_module)
    except ValueError:
        if status is not MigrationStatus.MIGRATED:
            raise
    return _migration_domain(canonical_symbol)


def _decode_migration_rows() -> tuple[tuple[object, ...], ...]:
    try:
        encoded = "".join(MIGRATION_LEDGER_PAYLOAD.split()).encode("ascii")
        serialized = zlib.decompress(base64.b85decode(encoded))
    except (ValueError, zlib.error) as error:
        raise ValueError(
            "Migration ledger payload is not valid zlib+base85 data."
        ) from error
    digest = hashlib.sha256(serialized).hexdigest()
    if digest != MIGRATION_LEDGER_SHA256:
        raise ValueError(
            "Migration ledger payload checksum mismatch: "
            f"expected {MIGRATION_LEDGER_SHA256}, got {digest}."
        )
    decoded = json.loads(serialized)
    if not isinstance(decoded, list):
        raise ValueError("Migration ledger payload must contain a JSON list.")
    rows: list[tuple[object, ...]] = []
    for value in decoded:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Every migration ledger row must be a JSON sequence.")
        row = tuple(value)
        if len(row) not in {8, 12}:
            raise ValueError(
                "Every migration ledger row must have exactly eight legacy or "
                "twelve strict values."
            )
        rows.append(row)
    actual_counts = Counter(str(row[7]) for row in rows)
    if dict(actual_counts) != MIGRATION_LEDGER_COUNTS:
        raise ValueError(
            "Migration ledger category counts do not match generated metadata: "
            f"expected {MIGRATION_LEDGER_COUNTS!r}, got {dict(actual_counts)!r}."
        )
    if len(rows) != MIGRATION_LEDGER_RECORD_COUNT:
        raise ValueError(
            "Migration ledger record count does not match generated metadata: "
            f"expected {MIGRATION_LEDGER_RECORD_COUNT}, got {len(rows)}."
        )
    record_ids = tuple(row[0] for row in rows)
    if any(not isinstance(record_id, str) for record_id in record_ids):
        raise ValueError("Every migration ledger record ID must be a string.")
    record_id_digest = hashlib.sha256(
        json.dumps(
            sorted(record_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if record_id_digest != MIGRATION_LEDGER_RECORD_IDS_SHA256:
        raise ValueError(
            "Migration ledger identity digest does not match generated metadata: "
            f"expected {MIGRATION_LEDGER_RECORD_IDS_SHA256}, "
            f"got {record_id_digest}."
        )
    return tuple(rows)


_EXPECTED_MIGRATION_ROWS = _decode_migration_rows()
EXPECTED_MIGRATION_RECORD_COUNT = MIGRATION_LEDGER_RECORD_COUNT
EXPECTED_MIGRATION_RECORD_IDS = frozenset(
    str(row[0]) for row in _EXPECTED_MIGRATION_ROWS
)


def migration_manifest_digest(records: Sequence[MigrationRecord]) -> str:
    """Return the frozen-row digest for a supplied migration ledger.

    The digest deliberately excludes derived domain authorities, which are
    validated independently, and includes every frozen former-site identity
    field plus its expected current occurrence count.  It is therefore
    directly comparable with the immutable payload digest.
    """
    rows = [
        [
            record.record_id,
            record.former_module,
            record.former_qualified_name,
            record.former_line,
            record.former_column,
            record.former_route,
            record.expected_current_occurrences,
            record.category.value,
            record.status.value,
            record.authority_kind.value,
            record.canonical_symbol,
            record.authority_field,
        ]
        for record in records
    ]
    serialized = json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_migration_records(
    boundaries: tuple[RuntimeBoundary, ...],
) -> tuple[MigrationRecord, ...]:
    records: list[MigrationRecord] = []
    for row in _EXPECTED_MIGRATION_ROWS:
        (
            raw_id,
            raw_module,
            raw_name,
            raw_line,
            raw_column,
            raw_route,
            raw_current_occurrences,
            raw_category,
            *strict_values,
        ) = row
        if (
            not isinstance(raw_id, str)
            or not isinstance(raw_module, str)
            or not isinstance(raw_name, str)
            or type(raw_line) is not int
            or type(raw_column) is not int
            or type(raw_current_occurrences) is not int
            or not isinstance(raw_route, str)
            or not isinstance(raw_category, str)
        ):
            raise ValueError("Migration ledger row values have invalid types.")
        category = MigrationCategory(raw_category)
        if strict_values:
            raw_status, raw_authority_kind, raw_symbol, raw_field = strict_values
            if (
                not isinstance(raw_status, str)
                or not isinstance(raw_authority_kind, str)
                or not isinstance(raw_symbol, str)
                or (raw_field is not None and not isinstance(raw_field, str))
            ):
                raise ValueError(
                    "Strict migration authority values have invalid types."
                )
            status = MigrationStatus(raw_status)
            authority_kind = MigrationAuthorityKind(raw_authority_kind)
            canonical_symbol = raw_symbol
            authority_field = raw_field
        else:
            status = (
                MigrationStatus.LIVE
                if raw_current_occurrences > 0
                else MigrationStatus.MIGRATED
            )
            authority_kind = (
                MigrationAuthorityKind.PATH_RESOLVER
                if category is MigrationCategory.PATH_RESOLUTION
                else MigrationAuthorityKind.EXECUTION_INPUT
            )
            canonical_symbol = (
                _PATH_AUTHORITY
                if category is MigrationCategory.PATH_RESOLUTION
                else (
                    raw_module if raw_name == "<module>" else f"{raw_module}.{raw_name}"
                )
            )
            authority_field = None
        domain = _migration_record_domain(
            raw_module,
            status=status,
            canonical_symbol=canonical_symbol,
        )
        coverage = migration_entrypoint_coverage(domain, boundaries)
        records.append(
            MigrationRecord(
                record_id=raw_id,
                former_module=raw_module,
                former_qualified_name=raw_name,
                former_line=raw_line,
                former_column=raw_column,
                former_route=raw_route,
                expected_current_occurrences=raw_current_occurrences,
                category=category,
                canonical_symbol=canonical_symbol,
                authority_kind=authority_kind,
                authority_field=authority_field,
                migration_target=_MIGRATION_TARGETS[category],
                status=status,
                domain=domain,
                entrypoint_coverage=coverage,
            )
        )
    return tuple(records)


_EXEMPTION_REASONS: Mapping[str, str] = {
    "algorithm-aggregation": (
        "in-memory algorithm evidence aggregation; it is not application configuration"
    ),
    "algorithm-optional-value": (
        "an explicit algorithm/tensor optional-value branch with no composed-value synthesis"
    ),
    "code-or-artifact-location": (
        "an immutable code location or child artifact name below an already typed root"
    ),
    "composed-path": (
        "a composition-owned role-relative path or Hydra metadata directory validated at its boundary"
    ),
    "persisted-layout": (
        "a persisted format filename or child layout below an already validated typed root"
    ),
    "persisted-optional-field": (
        "an optional field whose absence is part of the persisted record or tensor-data format"
    ),
    "persisted-record": (
        "a persisted record lookup governed by that record's exact format validation"
    ),
    "render-metadata": (
        "optional persisted visualization/metric metadata used only for diagnostic rendering"
    ),
    "strict-schema": (
        "a strict schema/type/semantic validation expression that never supplies a missing runtime value"
    ),
    "validation-fixture": (
        "a deterministic source-only validation/audit fixture rather than a runtime path or default"
    ),
    "vendor-algorithm": (
        "a vendored model/tensor algorithm invariant preserved independently of application configuration"
    ),
    "vendor-capability": (
        "third-party capability, checkpoint-shape, or import-surface introspection"
    ),
}


def audit_exemption_reason_code(exemption: AuditExemption) -> str:
    """Return the stable generated reason code for one loaded exemption."""
    matches = tuple(
        code for code, text in _EXEMPTION_REASONS.items() if text in exemption.reason
    )
    if len(matches) != 1:
        raise ValueError(
            "Audit exemption reason does not map to exactly one generated code: "
            f"{exemption.reason!r}."
        )
    return matches[0]


def _load_audit_exemptions() -> tuple[AuditExemption, ...]:
    try:
        encoded = "".join(AUDIT_EXEMPTION_PAYLOAD.split()).encode("ascii")
        serialized = zlib.decompress(base64.b85decode(encoded))
    except (ValueError, zlib.error) as error:
        raise ValueError(
            "Audit exemption payload is not valid zlib+base85 data."
        ) from error
    digest = hashlib.sha256(serialized).hexdigest()
    if digest != AUDIT_EXEMPTION_SHA256:
        raise ValueError(
            "Audit exemption payload checksum mismatch: "
            f"expected {AUDIT_EXEMPTION_SHA256}, got {digest}."
        )
    decoded = json.loads(serialized)
    if not isinstance(decoded, list):
        raise ValueError("Audit exemption payload must contain a JSON list.")
    if len(decoded) != AUDIT_EXEMPTION_RECORD_COUNT:
        raise ValueError(
            "Audit exemption record count mismatch: "
            f"expected {AUDIT_EXEMPTION_RECORD_COUNT}, got {len(decoded)}."
        )
    exemptions: list[AuditExemption] = []
    identifiers: list[str] = []
    reason_counts: Counter[str] = Counter()
    for raw in decoded:
        if not isinstance(raw, list) or len(raw) != 5:
            raise ValueError("Every audit exemption row must have five values.")
        module, qualified_name, line, raw_rule, reason_code = raw
        if (
            not isinstance(module, str)
            or not isinstance(qualified_name, str)
            or type(line) is not int
            or not isinstance(raw_rule, str)
            or not isinstance(reason_code, str)
        ):
            raise ValueError("Audit exemption row values have invalid types.")
        try:
            reason = _EXEMPTION_REASONS[reason_code]
        except KeyError as error:
            raise ValueError(
                f"Unknown audit exemption reason code: {reason_code}."
            ) from error
        identifiers.append(f"{module}|{qualified_name}|{line}|{raw_rule}")
        reason_counts[reason_code] += 1
        exemptions.append(
            AuditExemption(
                module=module,
                qualified_name=qualified_name,
                line=line,
                rule=AuditRule(raw_rule),
                reason=f"{module}.{qualified_name}: {reason}.",
            )
        )
    identifier_digest = hashlib.sha256(
        json.dumps(sorted(identifiers), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if identifier_digest != AUDIT_EXEMPTION_RECORD_IDS_SHA256:
        raise ValueError(
            "Audit exemption identity digest mismatch: "
            f"expected {AUDIT_EXEMPTION_RECORD_IDS_SHA256}, got {identifier_digest}."
        )
    if dict(reason_counts) != AUDIT_EXEMPTION_COUNTS:
        raise ValueError(
            "Audit exemption reason counts mismatch: "
            f"expected {AUDIT_EXEMPTION_COUNTS!r}, got {dict(reason_counts)!r}."
        )
    return tuple(exemptions)


_AUDIT_EXEMPTIONS = _load_audit_exemptions()


EXPECTED_AUDIT_EXEMPTIONS = _AUDIT_EXEMPTIONS
EXPECTED_AUDIT_RULES = tuple(AuditRule)


_BOUNDARY_VALIDATOR_KEYS: Mapping[str, str] = {
    "src.synthetic_data_generation.scripts.run_scene_pipeline": "synthetic.scene_pipeline",
    "src.synthetic_data_generation.scripts.visualize_dataset": "synthetic.dataset_visualization",
    "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio": "ball.web_tool",
    "src.tasks.ball_detection.scripts.convert_web_dataset": "ball.web_tool",
    "src.tasks.ball_detection.scripts.eval": "ball.eval",
    "src.tasks.ball_detection.scripts.evaluate_manifest": "ball.evaluate_manifest",
    "src.tasks.ball_detection.scripts.preview_augmentation": "ball.preview",
    "src.tasks.ball_detection.scripts.preview_heatmaps": "ball.preview",
    "src.tasks.ball_detection.scripts.train": "ball.train",
    "src.tasks.ball_detection.scripts.train_staged": "ball.train_staged",
    "src.tasks.ball_detection.scripts.visualize": "ball.visualize",
    "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball": "ball.annotation",
    "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset": "ball.youtube",
    "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images": "ball.youtube",
    "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset": "ball.youtube",
    "src.tasks.court_detection.scripts.annotate_youtube_keypoints": "court_detection.annotate_youtube_keypoints",
    "src.tasks.court_detection.scripts.evaluate_homography_annotations": "court_detection.evaluate_homography_annotations",
    "src.tasks.court_detection.scripts.generate_line_masks": "court_detection.generate_line_masks",
    "src.tasks.court_detection.scripts.generate_masks": "court_detection.generate_masks",
    "src.tasks.court_detection.scripts.materialize_targets": "court_detection.materialize_targets",
    "src.tasks.court_detection.scripts.prepare_youtube_dataset": "court_detection.prepare_youtube_dataset",
    "src.tasks.court_detection.scripts.preview_augmentation": "court_detection.preview_augmentation",
    "src.tasks.court_detection.scripts.preview_heatmaps": "court_detection.preview_heatmaps",
    "src.tasks.court_detection.scripts.train": "court_detection.train",
    "src.tasks.court_detection.scripts.visualize": "court_detection.visualize",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "blcs.api_server",
    "src.tasks.blcs.scripts.generate_dataset": "blcs.generate_dataset",
    "src.tasks.blcs.scripts.preview_augmentation": "blcs.preview_augmentation",
    "src.tasks.blcs.scripts.train": "blcs.train",
    "src.tasks.blcs.scripts.visualize": "blcs.visualize",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "plcs.analyze_angle_velocity",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "plcs.analyze_dataset_distribution",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "plcs.analyze_loss_dominance",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "plcs.analyze_rotation_error_samples",
    "src.tasks.plcs.scripts.generate_dataset": "plcs.generate_dataset",
    "src.tasks.plcs.scripts.preview_augmentation": "plcs.preview_augmentation",
    "src.tasks.plcs.scripts.train": "plcs.train",
    "src.tasks.plcs.scripts.visualize": "plcs.visualize",
    "src.tasks.slcs.scripts.analyze_predictions": "slcs.analyze_predictions",
    "src.tasks.slcs.scripts.evaluate": "slcs.evaluate",
    "src.tasks.slcs.scripts.make_splits": "slcs.make_splits",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "slcs.precompute_dino_tokens",
    "src.tasks.slcs.scripts.predict_clip": "slcs.predict_clip",
    "src.tasks.slcs.scripts.train": "slcs.train",
    "src.tennis_scene.scripts.clip_studio": "tennis_scene.clip_studio",
    "src.tennis_scene.scripts.export_clips": "tennis_scene.export_clips",
    "src.tennis_scene.scripts.generate_dataset": "tennis_scene.generate_dataset",
    "src.tennis_scene.scripts.run_pipeline": "tennis_scene.pipeline",
    "src.tennis_scene.scripts.visualization": "tennis_scene.visualization",
    "src.tennis_scene.scripts.visualize_tasks": "tennis_scene.visualize_tasks",
    "src.submodules.scripts.demo_gvhmr": "submodules.demo_gvhmr",
}

_BOUNDARY_VALIDATOR_CALLABLES: Mapping[str, str] = {
    "src.synthetic_data_generation.scripts.run_scene_pipeline": (
        "src.synthetic_data_generation.configuration.validate_scene_pipeline_boundary"
    ),
    "src.synthetic_data_generation.scripts.visualize_dataset": (
        "src.synthetic_data_generation.visualization.configuration."
        "validate_dataset_visualization_boundary"
    ),
    "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio": "src.tasks.ball_detection.configuration.validate_web_tool",
    "src.tasks.ball_detection.scripts.convert_web_dataset": "src.tasks.ball_detection.configuration.validate_web_tool",
    "src.tasks.ball_detection.scripts.eval": "src.tasks.ball_detection.configuration.validate_eval",
    "src.tasks.ball_detection.scripts.evaluate_manifest": "src.tasks.ball_detection.configuration.validate_manifest_boundary",
    "src.tasks.ball_detection.scripts.preview_augmentation": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.scripts.preview_heatmaps": "src.tasks.ball_detection.configuration.validate_preview",
    "src.tasks.ball_detection.scripts.train": "src.tasks.ball_detection.configuration.validate_training",
    "src.tasks.ball_detection.scripts.train_staged": "src.tasks.ball_detection.configuration.validate_training",
    "src.tasks.ball_detection.scripts.visualize": "src.tasks.ball_detection.configuration.validate_visualization",
    "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball": "src.tasks.ball_detection.configuration.validate_annotation_boundary",
    "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset": "src.tasks.ball_detection.configuration.validate_youtube_boundary",
    "src.tasks.court_detection.scripts.annotate_youtube_keypoints": "src.tasks.court_detection.scripts.annotate_youtube_keypoints._validate_boundary",
    "src.tasks.court_detection.scripts.evaluate_homography_annotations": "src.tasks.court_detection.scripts.evaluate_homography_annotations._validate_boundary",
    "src.tasks.court_detection.scripts.generate_line_masks": "src.tasks.court_detection.scripts.generate_line_masks._validate_boundary",
    "src.tasks.court_detection.scripts.generate_masks": "src.tasks.court_detection.scripts.generate_masks._validate_boundary",
    "src.tasks.court_detection.scripts.materialize_targets": "src.tasks.court_detection.scripts.materialize_targets._validate_boundary",
    "src.tasks.court_detection.scripts.prepare_youtube_dataset": "src.tasks.court_detection.scripts.prepare_youtube_dataset._validate_boundary",
    "src.tasks.court_detection.scripts.preview_augmentation": "src.tasks.court_detection.scripts.preview_augmentation._validate_boundary",
    "src.tasks.court_detection.scripts.preview_heatmaps": "src.tasks.court_detection.scripts.preview_heatmaps._validate_boundary",
    "src.tasks.court_detection.scripts.train": "src.tasks.court_detection.configuration.validate_train_boundary",
    "src.tasks.court_detection.scripts.visualize": "src.tasks.court_detection.scripts.visualize._validate_boundary",
    "src.tasks.blcs.generate_dataset.api_server.__main__": "src.tasks.blcs.configuration.validate_api_boundary",
    "src.tasks.blcs.scripts.generate_dataset": "src.tasks.blcs.configuration.validate_generation_boundary",
    "src.tasks.blcs.scripts.preview_augmentation": "src.tasks.blcs.configuration.validate_preview_boundary",
    "src.tasks.blcs.scripts.train": "src.tasks.blcs.configuration._validate_training_for_hydra",
    "src.tasks.blcs.scripts.visualize": "src.tasks.blcs.configuration.validate_visualization_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_angle_velocity": "src.tasks.plcs.configuration._validate_angle_velocity_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution": "src.tasks.plcs.configuration._validate_distribution_boundary",
    "src.tasks.plcs.scripts.analysis.analyze_loss_dominance": "src.tasks.plcs.configuration._validate_loss_dominance_boundary",
    "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples": "src.tasks.plcs.configuration._validate_rotation_error_boundary",
    "src.tasks.plcs.scripts.generate_dataset": "src.tasks.plcs.generate_dataset.config._validate_boundary",
    "src.tasks.plcs.scripts.preview_augmentation": "src.tasks.plcs.configuration._validate_preview_boundary",
    "src.tasks.plcs.scripts.train": "src.tasks.plcs.configuration._validate_training_boundary",
    "src.tasks.plcs.scripts.visualize": "src.tasks.plcs.configuration._validate_visualization_boundary",
    "src.tasks.slcs.scripts.analyze_predictions": "src.tasks.slcs.configuration.validate_analysis_boundary",
    "src.tasks.slcs.scripts.evaluate": "src.tasks.slcs.configuration.validate_evaluation_boundary",
    "src.tasks.slcs.scripts.make_splits": "src.tasks.slcs.configuration.validate_split_boundary",
    "src.tasks.slcs.scripts.precompute_dino_tokens": "src.tasks.slcs.configuration.validate_precompute_boundary",
    "src.tasks.slcs.scripts.predict_clip": "src.tasks.slcs.configuration.validate_prediction_boundary",
    "src.tasks.slcs.scripts.train": "src.tasks.slcs.configuration.validate_training_boundary",
    "src.tennis_scene.scripts.clip_studio": "src.tennis_scene.configuration.validate_clip_studio_boundary",
    "src.tennis_scene.scripts.export_clips": "src.tennis_scene.configuration.validate_export_clips_boundary",
    "src.tennis_scene.scripts.generate_dataset": "src.tennis_scene.configuration.validate_generate_dataset_boundary",
    "src.tennis_scene.scripts.run_pipeline": "src.tennis_scene.configuration.validate_pipeline_boundary",
    "src.tennis_scene.scripts.visualization": "src.tennis_scene.configuration.validate_visualization_boundary",
    "src.tennis_scene.scripts.visualize_tasks": "src.tennis_scene.configuration.validate_visualize_tasks_boundary",
    "src.submodules.scripts.demo_gvhmr": "src.submodules.scripts.demo_gvhmr._validate_boundary",
}


def _runtime_boundary(
    domain: str,
    module: str,
    *,
    callable_name: str = "main",
) -> RuntimeBoundary:
    validator_key = _BOUNDARY_VALIDATOR_KEYS.get(module)
    validator_callable = _BOUNDARY_VALIDATOR_CALLABLES.get(module)
    return RuntimeBoundary(
        domain=domain,
        module=module,
        callable_name=callable_name,
        kind=BoundaryKind.HYDRA,
        executable_module=True,
        validator_key=validator_key,
        validator_callable=validator_callable,
        configuration_authority=validator_callable,
        path_authority=_PATH_AUTHORITY,
        migration_target="validated typed runtime contract before side effects",
        required_policy="present after composition; missing values are errors",
        optional_policy="declared optional and absent without value synthesis",
        default_authority="composed configuration only; no Python runtime default",
        precedence_authority="single composed value; no fallback or alias precedence",
    )


_NON_HYDRA_BOUNDARY_BINDINGS: Mapping[str, tuple[str, str]] = {
    "src.automation.chatgpt_mcp.cli": (
        "automation.chatgpt_mcp",
        "src.utils.configuration.paths.NonHydraPathBoundary.validate",
    ),
}


def _non_hydra_boundary(
    module: str,
    callable_name: str,
    *,
    kind: BoundaryKind = BoundaryKind.ARGPARSE,
    domain: str = "synthetic_data_generation",
    executable_module: bool = False,
) -> RuntimeBoundary:
    validator_key, validator_callable = _NON_HYDRA_BOUNDARY_BINDINGS[module]
    return RuntimeBoundary(
        domain=domain,
        module=module,
        callable_name=callable_name,
        kind=kind,
        executable_module=executable_module,
        validator_key=validator_key,
        validator_callable=validator_callable,
        configuration_authority=validator_callable,
        path_authority=validator_callable,
        migration_target="validated typed runtime contract before side effects",
        required_policy="all declared path arguments are present",
        optional_policy="optional non-path values use an explicit typed contract",
        default_authority="caller-owned explicit values only; no boundary fallback",
        precedence_authority="one role/direction declaration per explicit path",
    )


_RUNTIME_BOUNDARIES = (
    _non_hydra_boundary(
        "src.automation.chatgpt_mcp.cli",
        "main",
        domain="automation",
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.run_scene_pipeline",
    ),
    _runtime_boundary(
        "synthetic_data_generation",
        "src.synthetic_data_generation.scripts.visualize_dataset",
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.analyze_web_bbox_ratio"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.convert_web_dataset"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.eval"),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.evaluate_manifest"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.preview_augmentation"
    ),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.preview_heatmaps"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.train"),
    _runtime_boundary(
        "ball_detection", "src.tasks.ball_detection.scripts.train_staged"
    ),
    _runtime_boundary("ball_detection", "src.tasks.ball_detection.scripts.visualize"),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.annotate_youtube_ball",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.clip_and_predict_youtube_dataset",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images",
    ),
    _runtime_boundary(
        "ball_detection",
        "src.tasks.ball_detection.scripts.youtube.prepare_youtube_dataset",
    ),
    _runtime_boundary("blcs", "src.tasks.blcs.generate_dataset.api_server.__main__"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.generate_dataset"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.preview_augmentation"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.train"),
    _runtime_boundary("blcs", "src.tasks.blcs.scripts.visualize"),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.annotate_youtube_keypoints",
    ),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.evaluate_homography_annotations",
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.generate_line_masks"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.generate_masks"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.materialize_targets"
    ),
    _runtime_boundary(
        "court_detection",
        "src.tasks.court_detection.scripts.prepare_youtube_dataset",
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.preview_augmentation"
    ),
    _runtime_boundary(
        "court_detection", "src.tasks.court_detection.scripts.preview_heatmaps"
    ),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.train"),
    _runtime_boundary("court_detection", "src.tasks.court_detection.scripts.visualize"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_angle_velocity"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.analyze_dataset_distribution"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.analysis.analyze_loss_dominance"),
    _runtime_boundary(
        "plcs", "src.tasks.plcs.scripts.analysis.visualize_rotation_error_samples"
    ),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.generate_dataset"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.preview_augmentation"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.train"),
    _runtime_boundary("plcs", "src.tasks.plcs.scripts.visualize"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.analyze_predictions"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.evaluate"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.make_splits"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.precompute_dino_tokens"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.predict_clip"),
    _runtime_boundary("slcs", "src.tasks.slcs.scripts.train"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.clip_studio"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.export_clips"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.generate_dataset"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.run_pipeline"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualization"),
    _runtime_boundary("tennis_scene", "src.tennis_scene.scripts.visualize_tasks"),
    _runtime_boundary("submodules", "src.submodules.scripts.demo_gvhmr"),
)


EXPECTED_RUNTIME_BOUNDARIES = _RUNTIME_BOUNDARIES


DEFAULT_AUDIT_INVENTORY = AuditInventory(
    boundaries=_RUNTIME_BOUNDARIES,
    migrations=_load_migration_records(_RUNTIME_BOUNDARIES),
    exemptions=_AUDIT_EXEMPTIONS,
    rules=EXPECTED_AUDIT_RULES,
)
