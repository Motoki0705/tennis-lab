#!/usr/bin/env python3
"""Ops report / daily triage entry point. See README.md in this directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opsreport.cli import main

if __name__ == "__main__":
    sys.exit(main())
