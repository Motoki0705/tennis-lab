"""An empty target set must never become a successful review request."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path[:0] = [str(OPS), str(OPS / "review")]
policy: Any = importlib.import_module("parent_review_policy")


def test_empty_review_request_is_rejected() -> None:
    assert policy.snapshots([]) == []
