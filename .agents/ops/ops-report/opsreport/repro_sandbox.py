"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import repro_sandbox as shared

sys.modules[__name__] = shared
