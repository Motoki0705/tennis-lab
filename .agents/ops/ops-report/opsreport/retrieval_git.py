"""Compatibility import for the shared implementation."""

import sys

from lib.review_tools import retrieval_git as shared

sys.modules[__name__] = shared
