"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import repro_isolation as shared

sys.modules[__name__] = shared
