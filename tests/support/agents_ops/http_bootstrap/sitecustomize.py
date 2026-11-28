"""Patch the transport and opted-in launch boundary in pytest child processes.

Delay until import so checkout tests retain their own module search paths and
unrelated Python tools/fake executables do not import the GitHub client.
"""

import os
import sys

# The same subprocess bootstrap also isolates WSL drive probes from the host.
if "TENNIS_TEST_D_CAPACITY_BYTES_FILE" in os.environ:
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "fake_d_capacity/sitecustomize.py"),
        run_name="sitecustomize",
    )

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

    hooks = {"issues.transport": "github_http.py"}
    if os.environ.get("AGENTS_TEST_DIRECT_MERGE_GATE") == "1":
        hooks["lib.ops_review.merge_gate_impl"] = "merge_gate_launcher.py"

    class FixtureLoader(importlib.abc.Loader):
        def __init__(self, loader: Any, fixture: str) -> None:
            self.loader = loader
            self.fixture = fixture

        def create_module(self, spec: Any) -> Any:
            return self.loader.create_module(spec)

        def exec_module(self, module: Any) -> None:
            self.loader.exec_module(module)
            path = Path(__file__).resolve().parents[1] / self.fixture
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            module._test_bridge = helper.install(module)

    class FixtureFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname not in hooks:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = FixtureLoader(spec.loader, hooks[fullname])
            return spec

    sys.meta_path.insert(0, FixtureFinder())

    # Preserve an explicit startup hook used by another fixture. Python loads
    # only one sitecustomize, so install our lazy transport hook and then run
    # the customization that would otherwise have been selected.
    directory = Path(__file__).resolve().parent
    search = [
        entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != directory
    ]
    existing = importlib.machinery.PathFinder.find_spec("sitecustomize", search)
    if existing is not None and existing.origin is not None:
        spec = importlib.util.spec_from_file_location(
            "_agents_existing_sitecustomize", existing.origin
        )
        assert spec is not None and spec.loader is not None
        customization = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = customization
        spec.loader.exec_module(customization)
