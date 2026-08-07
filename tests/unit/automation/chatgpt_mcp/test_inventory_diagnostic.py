from pathlib import Path

from src.utils.configuration.audit import regenerate_exemption_rows

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_print_new_inventory_findings() -> None:
    _, _, unresolved = regenerate_exemption_rows((PROJECT_ROOT / "src").resolve())
    if not unresolved:
        return
    details = "\n".join(
        f"{finding.module}|{finding.qualified_name}|{finding.line}|"
        f"{finding.column}|{finding.rule.value}"
        for finding in unresolved
    )
    raise AssertionError("NEW_INVENTORY_FINDINGS\n" + details)
