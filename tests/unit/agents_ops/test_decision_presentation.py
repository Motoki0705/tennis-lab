"""Decoded and formatted display text must pass the public guard before escape."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import (
    PublicationError,
    projection,
    result,
    safety,
    schema,
)

navigation: Any = importlib.import_module("decision_logs.navigation")
presentation: Any = importlib.import_module("decision_logs.presentation")


@pytest.mark.parametrize("syntax", ["json", "python", "field"])
def test_decoded_credential_blocks_readme_generation(syntax: str) -> None:
    # Dummy only: keep the prefix escaped even in the source of this regression.
    encoded = r"ghp\u005f" + "x" * 30
    payload = (
        ('{"value":"' + encoded + '"}')
        if syntax != "python"
        else ("{'value':'" + encoded + "'}")
    )
    summary = "value=" + payload if syntax == "field" else payload
    row = result()
    row["data"]["summary"] = summary
    envelope = projection.render(row)
    with pytest.raises(PublicationError, match="^credential$"):
        navigation.generate({navigation.event_path(row): schema.json_bytes(envelope)})


@pytest.mark.parametrize(
    "encoded,code",
    [
        (r"\u002fhome\u002ffixture\u002fprivate.txt", "absolute_path"),
        (
            r"12345678\u002d1234\u002d1234\u002d1234\u002d123456789abc",
            "private_identifier",
        ),
        (r"https:\u002f\u002flocalhost\u002fprivate", "unsafe_url"),
        (r"hidden\u0000control", "control_character"),
    ],
)
def test_decoded_display_values_use_all_public_guard_checks(
    encoded: str, code: str
) -> None:
    row = result()
    row["data"]["summary"] = '{"nested":[{"value":"' + encoded + '"}]}'
    envelope = projection.render(row)
    with pytest.raises(PublicationError, match="^" + code + "$"):
        navigation.generate({navigation.event_path(row): schema.json_bytes(envelope)})


@pytest.mark.parametrize(
    "key,code",
    [
        (r"api\u005fkey", "credential"),
        (r"session\u005fid", "private_identifier"),
    ],
)
def test_field_label_is_checked_with_its_formatted_delimiter(
    key: str, code: str
) -> None:
    row = result()
    row["data"]["summary"] = '{"' + key + '":"dummy-value"}'
    envelope = projection.render(row)
    with pytest.raises(PublicationError, match="^" + code + "$"):
        navigation.generate({navigation.event_path(row): schema.json_bytes(envelope)})


def test_prose_checks_full_value_before_truncation() -> None:
    value = "ordinary prefix " * 30 + "ghp" + chr(95) + "x" * 30
    with pytest.raises(PublicationError, match="^credential$"):
        presentation.prose(value, limit=10)


def test_prose_rechecks_a_boundary_created_by_truncation() -> None:
    value = "ghp" + chr(95) + "x" * 30
    safety.Guard().check(value + "é")
    with pytest.raises(PublicationError, match="^credential$"):
        presentation.prose(value + "é", limit=len(value))


def test_activity_title_rechecks_its_shortened_display_value() -> None:
    row = result()
    row["scope"].update(issue=None, delegation=None)
    row["data"]["summary"] = "ghp" + chr(95) + "x" * 130 + "é"
    envelope = projection.render(row)
    with pytest.raises(PublicationError, match="^credential$"):
        navigation.generate({navigation.event_path(row): schema.json_bytes(envelope)})


def test_entity_conversion_is_checked_before_reescaping() -> None:
    encoded = "ghp&#95;" + "x" * 30
    safety.Guard().check(encoded)
    with pytest.raises(PublicationError, match="^credential$"):
        presentation.checked_unescape(encoded)


def test_decoded_container_is_checked_before_display_omissions() -> None:
    encoded = r"ghp\u005f" + "x" * 30
    row = result()
    row["data"]["summary"] = "[" + ",".join(['"ordinary"'] * 12) + ',"' + encoded + '"]'
    envelope = projection.render(row)
    with pytest.raises(PublicationError, match="^credential$"):
        navigation.generate({navigation.event_path(row): schema.json_bytes(envelope)})
