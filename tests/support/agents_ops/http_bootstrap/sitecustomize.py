"""Patch only an imported transport in pytest's explicitly marked child env.

Delay until import so checkout tests retain their own module search paths and
unrelated Python tools/fake executables do not import the GitHub client.
"""

import os
import sys

# A fake gh backend never imports the client. Avoid charging importlib's hook
# machinery to every backend invocation in the HTTP fixtures.
if (
    os.environ.get("AGENTS_TEST_GITHUB_HTTP") == "1"
    and sys.argv[0].rpartition("/")[2] != "gh"
):
    import importlib.abc
    import importlib.machinery
    import importlib.util
    from pathlib import Path
    from typing import Any

    class TransportLoader(importlib.abc.Loader):
        def __init__(self, loader: Any) -> None:
            self.loader = loader

        def create_module(self, spec: Any) -> Any:
            return self.loader.create_module(spec)

        def exec_module(self, module: Any) -> None:
            self.loader.exec_module(module)
            path = Path(__file__).resolve().parents[1] / "github_http.py"
            spec = importlib.util.spec_from_file_location("github_http", path)
            assert spec is not None and spec.loader is not None
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            module._test_bridge = helper.install(module)

    class TransportFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname != "issues.transport":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = TransportLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, TransportFinder())
