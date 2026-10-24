"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import mcp_server as shared

if __name__ == "__main__":
    raise SystemExit(shared.main())
sys.modules[__name__] = shared
