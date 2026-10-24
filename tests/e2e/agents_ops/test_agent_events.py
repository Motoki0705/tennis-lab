"""Regressions for Codex's flattened web-search IDs at the CLI boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
WEB_ITEM = (
    '{"id":"item_13","type":"web_search","id":"exec-search",'
    '"query":"tennis","action":{"type":"search","queries":["tennis"]}}'
)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops/lib"))
    return importlib.import_module("agent_events")


@pytest.mark.parametrize("kind", ["item.started", "item.updated", "item.completed"])
def test_web_search_keeps_both_ids_and_search_payload(events: Any, kind: str) -> None:
    raw = '{"type":"' + kind + '","item":' + WEB_ITEM + "}"
    item = events.parse_cli_event(raw.encode(), "codex")["item"]
    assert item == {
        "id": "item_13",
        "type": "web_search",
        "web_search_id": "exec-search",
        "query": "tennis",
        "action": {"type": "search", "queries": ["tennis"]},
    }
    # The exception must not leak into saved records or final JSON parsing.
    with pytest.raises(ValueError, match="duplicate key"):
        events.parse_json(raw)
    with pytest.raises(ValueError, match="duplicate key"):
        events.parse_cli_event(raw, "claude")


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"item.started","type":"item.started","item":' + WEB_ITEM + "}",
        '{"type":"turn.started","item":' + WEB_ITEM + "}",
        '{"type":"item.started","nested":{"item":' + WEB_ITEM + "}}",
        '{"type":"item.started","items":[' + WEB_ITEM + "]}",
        '{"type":"item.started","item":' + WEB_ITEM + ',"item":{}}',
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"web_search"', '"agent_message"')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"id":"item_13"', '"id":"exec-reversed"')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"exec-search"', '""')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"exec-search"', "null")
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"id":"item_13"', '"id":13')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"query":', '"id":"third","query":')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"query":', '"web_search_id":"collision","query":')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace(
            '"type":"web_search"', '"type":"web_search","type":"web_search"'
        )
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"query":', '"query":"other","query":')
        + "}",
        '{"type":"item.started","item":'
        + WEB_ITEM.replace('"type":"search"', '"type":"search","type":"open"')
        + "}",
    ],
)
def test_other_duplicates_and_ambiguous_web_search_ids_are_rejected(
    events: Any, raw: str
) -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        events.parse_cli_event(raw, "codex")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_web_search_does_not_relax_non_finite_json(events: Any, constant: str) -> None:
    raw = (
        '{"type":"item.started","item":' + WEB_ITEM.replace('"tennis"', constant) + "}"
    )
    with pytest.raises(ValueError, match="non-finite"):
        events.parse_cli_event(raw, "codex")


def test_web_search_without_duplicate_ids_remains_valid(events: Any) -> None:
    raw = '{"type":"item.started","item":{"type":"web_search","id":"search"}}'
    assert events.parse_cli_event(raw, "codex") == events.parse_json(raw)
