"""Launch a checkout's test module in a fresh Python interpreter."""

from __future__ import annotations

import sys

_RUN_MODULE = """
import runpy
import sys
from pathlib import Path
from types import ModuleType

package = ModuleType("tests")
package.__path__ = [str(Path.cwd() / "tests")]
sys.modules["tests"] = package
runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)
"""


def isolated_module_command(module: str, *arguments: str) -> list[str]:
    """Run from the repo root so installed packages named tests cannot shadow it."""
    return [sys.executable, "-c", _RUN_MODULE, module, *arguments]
