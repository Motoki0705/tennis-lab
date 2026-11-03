"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import retrieval_github as shared

sys.modules[__name__] = shared
