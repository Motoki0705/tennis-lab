"""Tests for DINOv3 dynamic-output boundary validation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.utils.models.loading import require_dinov3_patch_tokens

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('external_namespace', [False, True])
def test_namespace_packages_require_declared_search_directories(tmp_path: Path, external_namespace: bool) -> None:
    repository = tmp_path / 'declared'
    (repository / 'dinov3').mkdir(parents=True)
    (repository / 'dinov3/backend.py').write_text('import torch\ndef make():\n    return torch.nn.Linear(1, 1)\n')
    (repository / 'hubconf.py').write_text('from dinov3.backend import make\ndef tiny(pretrained=False):\n    return make()\n')
    (tmp_path / 'outside/dinov3').mkdir(parents=True)
    script = '''from pathlib import Path
import sys
from src.utils.models.loading.dinov3 import _guarded_hub_load
root = Path(sys.argv[1]); external = sys.argv[2] == 'True'
if external:
    sys.path.insert(0, str(root/'outside'))
try:
    result = _guarded_hub_load(root/'declared', 'tiny')
except RuntimeError as error:
    assert external and 'declared' in str(error), str(error)
else:
    assert not external, 'External namespace directory was accepted'
    assert result.in_features == 1
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(external_namespace)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('resolution', ['external_package', 'external_submodule', 'unverifiable'])
def test_hub_rejects_modules_outside_the_declared_inventory(tmp_path: Path, resolution: str) -> None:
    declared, external = tmp_path / 'declared', tmp_path / 'external'
    declared.mkdir()
    (external / 'dinov3').mkdir(parents=True)
    (external / 'dinov3/__init__.py').write_text('')
    (external / 'dinov3/backend.py').write_text('import torch\ndef make():\n    return torch.nn.Linear(1, 1)\n')
    if resolution == 'external_submodule':
        (declared / 'dinov3').mkdir()
        (declared / 'dinov3/__init__.py').write_text(f'__path__.append({str(external / "dinov3")!r})\n')
    elif resolution == 'unverifiable':
        (external / 'dinov3/__init__.py').write_text('__file__ = None\n')
    (declared / 'hubconf.py').write_text('from dinov3.backend import make\ndef tiny(pretrained=False):\n    return make()\n')
    script = '''from pathlib import Path
import sys
from src.utils.models.loading.dinov3 import _guarded_hub_load
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'external'))
try:
    _guarded_hub_load(root/'declared', 'tiny')
except RuntimeError as error:
    assert 'declared' in str(error), str(error)
else:
    raise AssertionError('Unbound DINOv3 implementation was accepted')
for name in tuple(sys.modules):
    if name == 'dinov3' or name.startswith('dinov3.'):
        del sys.modules[name]
try:
    _guarded_hub_load(root/'declared', 'tiny')
except RuntimeError as error:
    assert 'fresh process' in str(error)
else:
    raise AssertionError('Unverified imports were retried in the same process')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("layout", ["package", "nested", "broken", "cycle"])
def test_unverifiable_directory_symlinks_stop_before_model_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    from src.utils.models.loading.dinov3 import (
        dinov3_source_assets,
        load_dinov3_backbone,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# valid hub stub\n")
    target = tmp_path / "implementation"
    if layout != "broken":
        target.mkdir()
        (target / "backend.py").write_text("# implementation\n")
    link = repository / "dinov3"
    if layout == "nested":
        link.mkdir()
        link = link / "hub"
    link.symlink_to(repository if layout == "cycle" else target, target_is_directory=True)
    checkpoint = tmp_path / "weights.pt"
    torch.save({}, checkpoint)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid source layout reached model construction")
    monkeypatch.setattr(torch.hub, "load", forbidden)
    with pytest.raises(ValueError, match="symlink"):
        dinov3_source_assets(repository)
    with pytest.raises(ValueError, match="symlink"):
        load_dinov3_backbone(repository_path=repository, checkpoint_path=checkpoint, backbone_name="tiny", strict=True)


@pytest.mark.parametrize("mode", ["different", "same", "same_copy", "untracked"])
def test_hub_implementation_switch_is_explicitly_rejected(tmp_path: Path, mode: str) -> None:
    for label in ("a", "b"):
        repo = tmp_path / label
        (repo / "dinov3" / "hub").mkdir(parents=True)
        (repo / "dinov3" / "__init__.py").write_text("")
        (repo / "dinov3" / "hub" / "__init__.py").write_text("")
        tag = "a" if mode in {"same", "same_copy"} else label
        (repo / "dinov3" / "hub" / "backbones.py").write_text(
            f'import torch\ndef make():\n    m = torch.nn.Linear(1, 1)\n    m.embed_dim = 1\n    m.patch_size = 1\n    m.tag = {tag!r}\n    return m\n')
        (repo / "hubconf.py").write_text('from dinov3.hub.backbones import make\ndef tiny(pretrained=False):\n    return make()\n')
        torch.save(torch.nn.Linear(1, 1).state_dict(), repo / "weights.pt")
    script = '''from pathlib import Path
import sys
import torch
from src.utils.models.loading.dinov3 import load_dinov3_backbone
root = Path(sys.argv[1])
def load(name):
    return load_dinov3_backbone(repository_path=root/name, checkpoint_path=root/name/'weights.pt', backbone_name='tiny', strict=True).module.tag
mode = sys.argv[2]
if mode == 'untracked':
    assert torch.hub.load(str(root/'a'), 'tiny', source='local', pretrained=False).tag == 'a'
else:
    assert load('a') == 'a'
try:
    result = load('a' if mode == 'same' else 'b')
except RuntimeError as error:
    assert 'implementation' in str(error) or 'declared' in str(error)
else:
    assert mode == 'same' and result == 'a', 'Unsafe implementation switch was accepted: ' + result
if mode == 'same':
    assert result == 'a'
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_require_dinov3_patch_tokens_accepts_exact_contract() -> None:
    tokens = torch.randn(2, 16, 8)

    result = require_dinov3_patch_tokens(
        {"x_norm_patchtokens": tokens},
        expected_batch_size=2,
        expected_num_tokens=16,
        expected_embed_dim=8,
    )

    assert result is tokens


@pytest.mark.parametrize(
    ("outputs", "error", "message"),
    [
        ([], TypeError, "must return a mapping"),
        ({}, KeyError, "missing required"),
        ({"x_norm_patchtokens": []}, TypeError, "must be a tensor"),
        ({"x_norm_patchtokens": torch.randn(2, 8)}, ValueError, "shape"),
        ({"x_norm_patchtokens": torch.randn(1, 16, 8)}, ValueError, "batch size"),
        ({"x_norm_patchtokens": torch.randn(2, 15, 8)}, ValueError, "token count"),
        ({"x_norm_patchtokens": torch.randn(2, 16, 7)}, ValueError, "embedding width"),
    ],
)
def test_require_dinov3_patch_tokens_rejects_contract_violations(
    outputs: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        require_dinov3_patch_tokens(
            outputs,
            expected_batch_size=2,
            expected_num_tokens=16,
            expected_embed_dim=8,
        )
