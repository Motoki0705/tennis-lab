#!/usr/bin/env python3
"""Weekly ops report / daily triage entry point. See README.md in this directory."""

from __future__ import annotations

import sys

from opsreport.cli import main

if __name__ == "__main__":
    sys.exit(main())
