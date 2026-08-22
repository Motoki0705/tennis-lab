"""Current-source configuration and path audit tests."""

from __future__ import annotations

from pathlib import Path

from src.utils.configuration import AuditInventory, AuditRule
from src.utils.configuration.audit import audit_source, inspect_source
from src.utils.paths import PROJECT_ROOT


def _write_source(tmp_path: Path, source: str) -> Path:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "sample.py").write_text(source, encoding="utf-8")
    return source_root


def test_repository_source_satisfies_current_audit() -> None:
    report = inspect_source((PROJECT_ROOT / "src").resolve())

    assert report.passed
    assert not report.findings
    assert not report.boundary_issues


def test_raw_configuration_fallbacks_fail_closed(tmp_path: Path) -> None:
    source_root = _write_source(
        tmp_path,
        "from collections.abc import Mapping\n"
        "def read_config(config: Mapping[str, object]) -> object:\n"
        "    first = config.get('first', 1)\n"
        "    second = config.get('section').get('second')\n"
        "    third = getattr(config, 'third', 3)\n"
        "    config.setdefault('fourth', 4)\n"
        "    return config['required'] or first or second or third\n",
    )

    rules = {finding.rule for finding in audit_source(source_root)}

    assert {
        AuditRule.GET_WITH_FALLBACK,
        AuditRule.CHAINED_GET,
        AuditRule.GETATTR_WITH_FALLBACK,
        AuditRule.SETDEFAULT,
        AuditRule.NULL_COALESCING,
    } <= rules


def test_unvalidated_runtime_paths_fail_closed(tmp_path: Path) -> None:
    source_root = _write_source(
        tmp_path,
        "import os\n"
        "from collections.abc import Mapping\n"
        "from pathlib import Path\n"
        "def to_absolute_path(value: str) -> str:\n"
        "    return value\n"
        "def read_config(config: Mapping[str, object]) -> object:\n"
        "    raw = Path(config['root'])\n"
        "    return (\n"
        "        raw, Path('data/cache'), Path.cwd(), os.getcwd(),\n"
        "        to_absolute_path('outputs/run'), Path(__file__).parents[1],\n"
        "    )\n",
    )

    rules = {finding.rule for finding in audit_source(source_root)}

    assert {
        AuditRule.RAW_PATH_CONSTRUCTION,
        AuditRule.RUNTIME_PATH_LITERAL,
        AuditRule.PROCESS_CWD,
        AuditRule.HYDRA_ABSOLUTE_PATH,
        AuditRule.FILE_PARENT_INDEX,
    } <= rules


def test_raw_configuration_aliases_remain_auditable(tmp_path: Path) -> None:
    source_root = _write_source(
        tmp_path,
        "from collections.abc import Mapping\n"
        "from pathlib import Path\n"
        "def read_config(config: Mapping[str, object]) -> object:\n"
        "    optional = config['optional']\n"
        "    alias = optional\n"
        "    root = config.get('root')\n"
        "    return alias or 'fallback', Path(root)\n",
    )

    rules = {finding.rule for finding in audit_source(source_root)}

    assert AuditRule.NULL_COALESCING in rules
    assert AuditRule.RAW_PATH_CONSTRUCTION in rules


def test_typed_children_and_persisted_records_are_not_configuration_findings(
    tmp_path: Path,
) -> None:
    source_root = _write_source(
        tmp_path,
        "from pathlib import Path\n"
        "class AppConfig:\n"
        "    output_ext: str\n"
        "    optional: str | None\n"
        "def build(cfg: AppConfig, root: Path, payload: dict[str, object]) -> object:\n"
        "    target = root / f'frame.{cfg.output_ext}'\n"
        "    optional = 'derived' if cfg.optional is None else cfg.optional\n"
        "    return target, optional, payload.get('persisted', 'missing')\n"
        "def save_metadata(config: dict[str, object] | None = None) -> object:\n"
        "    return config or {}\n",
    )

    report = inspect_source(source_root, inventory=AuditInventory())

    assert report.passed
    assert not report.findings


def test_ordinary_source_edits_need_no_snapshot_refresh(tmp_path: Path) -> None:
    source_root = _write_source(
        tmp_path,
        "def value() -> int:\n"
        "    return 1\n",
    )

    before = inspect_source(source_root, inventory=AuditInventory())
    (source_root / "sample.py").write_text(
        "\n\ndef value() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    after = inspect_source(source_root, inventory=AuditInventory())

    assert before.passed
    assert after.passed
