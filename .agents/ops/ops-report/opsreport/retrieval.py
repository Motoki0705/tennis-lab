"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import retrieval as shared

sys.modules[__name__] = shared
