"""Apply reviewed hardening patches after materializing the WSL MCP bundle."""

from __future__ import annotations

from pathlib import Path


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one patch target in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_workspace() -> None:
    path = Path("src/automation/chatgpt_mcp/workspace.py")
    old = '''    def _validate_trusted_mirror(self) -> None:\n        if not self.git_dir.is_dir():\n            raise WorkspaceError(f"trusted Git mirror is missing: {self.git_dir}")\n        bare = self._checked_git(\n            ["rev-parse", "--is-bare-repository"],\n            message="trusted Git mirror is invalid",\n        )\n        if bare != "true":\n            raise WorkspaceError("trusted Git mirror must be bare")\n'''
    new = '''    def _validate_trusted_mirror(self) -> None:\n        if not self.git_dir.is_dir():\n            raise WorkspaceError(f"trusted Git mirror is missing: {self.git_dir}")\n        result = self._git(["rev-parse", "--is-bare-repository"])\n        if result.returncode != 0 or result.stdout.strip() != "true":\n            raise WorkspaceError("trusted Git mirror must be bare")\n'''
    _replace_once(path, old, new)


def patch_cli_boundary() -> None:
    path = Path("src/automation/chatgpt_mcp/cli.py")
    old_import = '''from src.automation.chatgpt_mcp.settings import GatewaySettings\nfrom src.automation.chatgpt_mcp.tunnel import QuickTunnel\n\n\ndef _state_dir() -> Path:\n'''
    new_import = '''from src.automation.chatgpt_mcp.settings import GatewaySettings\nfrom src.automation.chatgpt_mcp.tunnel import QuickTunnel\nfrom src.utils.configuration import (\n    BoundaryPathField,\n    NonHydraPathBoundary,\n    PathDirection,\n    PathKind,\n    PathResolver,\n    PathRole,\n    RuntimePathRoots,\n)\n\nPATH_BOUNDARY = NonHydraPathBoundary(\n    name="automation.chatgpt_mcp",\n    fields=(\n        BoundaryPathField(\n            "repo_root",\n            PathRole.PROJECT,\n            PathDirection.INPUT,\n            PathKind.DIRECTORY,\n            must_exist=True,\n            allow_role_root=True,\n        ),\n        BoundaryPathField(\n            "state_dir",\n            PathRole.ARTIFACT,\n            PathDirection.OUTPUT,\n            PathKind.DIRECTORY,\n            allow_role_root=True,\n        ),\n    ),\n)\n\n\ndef _state_dir() -> Path:\n'''
    _replace_once(path, old_import, new_import)

    old_dispatch = '''    arguments = parser.parse_args()\n\n    if arguments.command == "serve":\n'''
    new_dispatch = '''    arguments = parser.parse_args()\n    boundary_settings = GatewaySettings.from_env(\n        public_base_url=(\n            arguments.public_base_url if arguments.command == "serve" else None\n        ),\n        require_public_base_url=arguments.command == "serve",\n    )\n    roots = RuntimePathRoots(\n        project_root=boundary_settings.repo_root,\n        data_root=boundary_settings.repo_root,\n        checkpoint_root=boundary_settings.repo_root,\n        artifact_root=boundary_settings.state_dir,\n        output_root=boundary_settings.state_dir,\n        cache_root=boundary_settings.state_dir,\n        external_asset_root=boundary_settings.repo_root,\n    )\n    PATH_BOUNDARY.validate(\n        {\n            "repo_root": boundary_settings.repo_root,\n            "state_dir": boundary_settings.state_dir,\n        },\n        resolver=PathResolver(roots),\n    )\n\n    if arguments.command == "serve":\n'''
    _replace_once(path, old_dispatch, new_dispatch)


def patch_runtime() -> None:
    path = Path("src/automation/chatgpt_mcp/runtime.py")
    old_venv = '''    def _install_venv(self) -> Path:\n        target = self.settings.runtime_venv_root\n        link = self.settings.project_venv_link\n\n        if target.is_dir():\n            self._ensure_project_venv_link(link, target)\n        else:\n            if link.is_symlink():\n                resolved = link.resolve()\n                if resolved != target or not target.is_dir():\n                    raise RuntimeInstallError(\n                        f"project .venv symlink does not name trusted venv: {link}"\n                    )\n            elif link.is_dir():\n                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)\n                try:\n                    os.replace(link, target)\n                except OSError:\n                    shutil.move(str(link), str(target))\n                os.symlink(target, link, target_is_directory=True)\n            else:\n                raise RuntimeInstallError(\n                    f"bootstrap virtual environment is missing: {link}"\n                )\n\n        python_executable = target / "bin/python"\n        if not python_executable.exists():\n            raise RuntimeInstallError(\n                f"trusted virtual environment has no Python: {python_executable}"\n            )\n        return python_executable\n'''
    new_venv = '''    def _install_venv(self) -> Path:\n        target = self.settings.runtime_venv_root\n        link = self.settings.project_venv_link\n        if not target.is_dir():\n            raise RuntimeInstallError(\n                "trusted virtual environment must be provisioned outside tennis-lab "\n                f"before runtime installation: {target}"\n            )\n        python_executable = target / "bin/python"\n        if not python_executable.exists():\n            raise RuntimeInstallError(\n                f"trusted virtual environment has no Python: {python_executable}"\n            )\n        self._ensure_project_venv_link(link, target)\n        return python_executable\n'''
    _replace_once(path, old_venv, new_venv)

    old_release = '''            shutil.copytree(\n                source / "src/automation/chatgpt_mcp",\n                candidate / "src/automation/chatgpt_mcp",\n                dirs_exist_ok=False,\n                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),\n            )\n            os.replace(candidate, release)\n'''
    new_release = '''            shutil.copytree(\n                source / "src/automation/chatgpt_mcp",\n                candidate / "src/automation/chatgpt_mcp",\n                dirs_exist_ok=False,\n                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),\n            )\n            configuration = candidate / "src/utils/configuration"\n            configuration.mkdir(mode=0o700, parents=True)\n            (candidate / "src/utils/__init__.py").write_text(\n                '\"\"\"Minimal trusted runtime utilities.\"\"\"\\n',\n                encoding="utf-8",\n            )\n            for module_name in ("errors.py", "schema.py", "paths.py"):\n                shutil.copy2(\n                    source / "src/utils/configuration" / module_name,\n                    configuration / module_name,\n                )\n            (configuration / "__init__.py").write_text(\n                "from src.utils.configuration.paths import (\\n"\n                "    BoundaryPathField, NonHydraPathBoundary, PathDirection, PathKind,\\n"\n                "    PathResolver, PathRole, RuntimePathRoots,\\n"\n                ")\\n"\n                "__all__ = [\\n"\n                "    'BoundaryPathField', 'NonHydraPathBoundary', 'PathDirection',\\n"\n                "    'PathKind', 'PathResolver', 'PathRole', 'RuntimePathRoots',\\n"\n                "]\\n",\n                encoding="utf-8",\n            )\n            os.replace(candidate, release)\n'''
    _replace_once(path, old_release, new_release)


def patch_job_environment() -> None:
    path = Path("src/automation/chatgpt_mcp/jobs.py")
    old = '''            "--env",\n            "HOME=/tmp/tennis-mcp-home",\n            "--env",\n            "PYTHONUNBUFFERED=1",\n'''
    new = '''            "--env",\n            "HOME=/tmp/tennis-mcp-home",\n            "--env",\n            "TMPDIR=/tmp",\n            "--env",\n            "XDG_CACHE_HOME=/tennis-lab/.cache",\n            "--env",\n            "HF_HOME=/tennis-lab/.cache/huggingface",\n            "--env",\n            "TORCH_HOME=/tennis-lab/.cache/torch",\n            "--env",\n            "MPLCONFIGDIR=/tennis-lab/.cache/matplotlib",\n            "--env",\n            "WANDB_DIR=/tennis-lab/outputs/wandb",\n            "--env",\n            "PYTHONUNBUFFERED=1",\n'''
    _replace_once(path, old, new)


def patch_runtime_tests() -> None:
    path = Path("tests/unit/automation/chatgpt_mcp/test_runtime.py")
    old_fixture = '''    package = source / "src/automation/chatgpt_mcp"\n    package.mkdir(parents=True)\n    (source / "src/__init__.py").write_text("", encoding="utf-8")\n    (source / "src/automation/__init__.py").write_text("", encoding="utf-8")\n    (package / "__init__.py").write_text("", encoding="utf-8")\n    (package / "example.py").write_text("VALUE = 1\\n", encoding="utf-8")\n'''
    new_fixture = '''    package = source / "src/automation/chatgpt_mcp"\n    package.mkdir(parents=True)\n    (source / "src/__init__.py").write_text("", encoding="utf-8")\n    (source / "src/automation/__init__.py").write_text("", encoding="utf-8")\n    (package / "__init__.py").write_text("", encoding="utf-8")\n    (package / "example.py").write_text("VALUE = 1\\n", encoding="utf-8")\n    configuration = source / "src/utils/configuration"\n    configuration.mkdir(parents=True)\n    for module_name in ("errors.py", "schema.py", "paths.py"):\n        (configuration / module_name).write_text("VALUE = 1\\n", encoding="utf-8")\n'''
    _replace_once(path, old_fixture, new_fixture)

    old_settings = '''    settings.uv_python_root.mkdir()\n\n    first = RuntimeInstaller(settings).install(source)\n'''
    new_settings = '''    settings.uv_python_root.mkdir()\n    trusted_python = settings.runtime_venv_root / "bin/python"\n    trusted_python.parent.mkdir(parents=True)\n    trusted_python.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")\n    os.chmod(trusted_python, 0o700)\n\n    first = RuntimeInstaller(settings).install(source)\n'''
    _replace_once(path, old_settings, new_settings)

    old_assert = '''    assert settings.runtime_current_dir.is_symlink()\n    assert settings.runtime_current_dir.resolve() == first.release_dir\n'''
    new_assert = '''    assert settings.runtime_current_dir.is_symlink()\n    assert settings.runtime_current_dir.resolve() == first.release_dir\n    assert first.release_dir.joinpath("src/utils/configuration/paths.py").is_file()\n'''
    _replace_once(path, old_assert, new_assert)


def patch_deploy_workflow() -> None:
    path = Path(".github/workflows/deploy-wsl-mcp.yml")
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '      - ".github/workflows/deploy-wsl-mcp.yml"\n',
        '      - ".github/workflows/deploy-wsl-mcp.yml"\n'
        '      - "pyproject.toml"\n'
        '      - "uv.lock"\n',
        1,
    )
    checkout = '''      - name: Checkout reviewed main revision\n        uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6\n        with:\n          clean: true\n          persist-credentials: false\n          submodules: false\n\n'''
    setup = checkout + '''      - name: Set up trusted uv bootstrap\n        uses: astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78 # v7\n        with:\n          python-version: "3.11"\n\n'''
    if text.count(checkout) != 1:
        raise SystemExit("deploy workflow checkout patch target is missing")
    text = text.replace(checkout, setup, 1)

    old_bootstrap = '''          bootstrap_python="$PROJECT_ROOT/.venv/bin/python"\n          tunnel_id_file="$MCP_STATE_DIR/secure-tunnel/tunnel-id"\n          runtime_key_file="$MCP_STATE_DIR/secure-tunnel/runtime-api-key"\n\n          test -x "$bootstrap_python"\n          test -s "$tunnel_id_file"\n          test -s "$runtime_key_file"\n\n          tunnel_id="$(tr -d '\\r\\n' < "$tunnel_id_file")"\n'''
    new_bootstrap = '''          tunnel_id_file="$MCP_STATE_DIR/secure-tunnel/tunnel-id"\n          runtime_key_file="$MCP_STATE_DIR/secure-tunnel/runtime-api-key"\n\n          test -s "$tunnel_id_file"\n          test -s "$runtime_key_file"\n          mkdir -p "$MCP_CONTROL_DIR/venvs"\n          venv_key="$(sha256sum "$GITHUB_WORKSPACE/uv.lock" | awk '{print $1}')"\n          venv_target="$MCP_CONTROL_DIR/venvs/$venv_key"\n          if [[ ! -x "$venv_target/bin/python" ]]; then\n            rm -rf "$venv_target"\n            (\n              cd "$GITHUB_WORKSPACE"\n              UV_PROJECT_ENVIRONMENT="$venv_target" \\\n                uv sync --locked --no-install-project\n            )\n          fi\n          venv_link="$MCP_CONTROL_DIR/venv"\n          if [[ -e "$venv_link" && ! -L "$venv_link" ]]; then\n            rm -rf "$venv_link"\n          fi\n          venv_tmp="$MCP_CONTROL_DIR/.venv.${GITHUB_RUN_ID}.tmp"\n          rm -f "$venv_tmp"\n          ln -s "venvs/$venv_key" "$venv_tmp"\n          mv -Tf "$venv_tmp" "$venv_link"\n          bootstrap_python="$venv_link/bin/python"\n          test -x "$bootstrap_python"\n\n          tunnel_id="$(tr -d '\\r\\n' < "$tunnel_id_file")"\n'''
    if text.count(old_bootstrap) != 1:
        raise SystemExit("deploy workflow bootstrap patch target is missing")
    text = text.replace(old_bootstrap, new_bootstrap, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_workspace()
    patch_cli_boundary()
    patch_runtime()
    patch_job_environment()
    patch_runtime_tests()
    patch_deploy_workflow()


if __name__ == "__main__":
    main()
