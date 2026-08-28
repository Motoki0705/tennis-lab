from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one match, found {count}: {old[:120]!r}"
        )
    write(path, text.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str, *, flags: int = 0) -> None:
    text = read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one regex match, found {count}: {pattern!r}"
        )
    write(path, updated)


def remove_line(path: str, line: str, *, count: int = 1) -> None:
    text = read(path)
    needle = line + "\n"
    actual = text.count(needle)
    if actual != count:
        raise RuntimeError(
            f"{path}: expected {count} occurrences, found {actual}: {line!r}"
        )
    write(path, text.replace(needle, "", count))


model = "src/tasks/blcs/models/blcs_multiview_axial_model.py"
new_stage = '''class _AxialAttentionStage(nn.Module):
    """One preconstructed camera/time stage using global attention on both axes."""

    def __init__(
        self,
        *,
        camera_layers: list[TransformerBlock],
        time_layers: list[TransformerBlock],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.camera_layers = nn.ModuleList(camera_layers)
        self.time_layers = nn.ModuleList(time_layers)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        camera_attention_keep_mask: Tensor,
        time_attention_keep_mask: Tensor,
        camera_frequencies: Tensor,
        time_frequencies: Tensor,
    ) -> Tensor:
        """Apply the fixed camera layers followed by global time layers."""
        batch_size, seq_len, num_cameras = x.shape[:3]
        for layer in self.camera_layers:
            camera_values = x.reshape(
                batch_size * seq_len,
                num_cameras,
                self.hidden_dim,
            )
            camera_values = layer(
                camera_values,
                freqs_cis=camera_frequencies,
                attn_mask=camera_attention_keep_mask,
            )
            x = camera_values.reshape(
                batch_size,
                seq_len,
                num_cameras,
                self.hidden_dim,
            )
        for layer in self.time_layers:
            time_values = x.permute(0, 2, 1, 3).reshape(
                batch_size * num_cameras,
                seq_len,
                self.hidden_dim,
            )
            time_values = layer(
                time_values,
                freqs_cis=time_frequencies,
                attn_mask=time_attention_keep_mask,
            )
            x = time_values.reshape(
                batch_size,
                num_cameras,
                seq_len,
                self.hidden_dim,
            ).permute(0, 2, 1, 3)
        return x


class BLCSMultiViewAxialModel'''
regex_once(
    model,
    r"class _GlobalTimeAttention\(nn\.Module\):.*?\n\nclass BLCSMultiViewAxialModel",
    new_stage,
    flags=re.DOTALL,
)
for line in (
    "        time_window_radius: int,",
    "        time_global_stage_mask: Sequence[bool],",
    "            time_global_stage_mask=time_global_stage_mask,",
    "        self.time_window_radius = int(time_window_radius)",
    "        self.time_global_stage_mask = time_global_stage_mask",
    "            time_global_stage_mask=config.time_global_stage_mask,",
    "            time_window_radius=config.time_window_radius,",
):
    remove_line(model, line)
replace_once(
    model,
    '''        time_global_stage_mask = self._normalize_stage_mask(
            time_global_stage_mask,
        )

''',
    "",
)
replace_once(
    model,
    '''        if self.time_window_radius < 0:
            raise ValueError(
                f"time_window_radius must be non-negative, got {self.time_window_radius}"
            )

''',
    "",
)
replace_once(
    model,
    '''        for camera_count, time_count, global_last in zip(
            self.camera_layers_per_stage,
            self.time_layers_per_stage,
            self.time_global_stage_mask,
            strict=True,
        ):
            time_implementations: list[nn.Module] = []
            for time_index in range(time_count):
                block = TransformerBlock(time_block_config)
                if global_last and time_index == time_count - 1:
                    time_implementations.append(
                        _GlobalTimeAttention(block, self.hidden_dim)
                    )
                else:
                    time_implementations.append(
                        _SlidingTimeAttention(block, self.hidden_dim)
                    )
            stages.append(
                _AxialAttentionStage(
                    camera_layers=[
                        TransformerBlock(camera_block_config)
                        for _ in range(camera_count)
                    ],
                    time_layers=time_implementations,
                    hidden_dim=self.hidden_dim,
                )
            )
''',
    '''        for camera_count, time_count in zip(
            self.camera_layers_per_stage,
            self.time_layers_per_stage,
            strict=True,
        ):
            stages.append(
                _AxialAttentionStage(
                    camera_layers=[
                        TransformerBlock(camera_block_config)
                        for _ in range(camera_count)
                    ],
                    time_layers=[
                        TransformerBlock(time_block_config)
                        for _ in range(time_count)
                    ],
                    hidden_dim=self.hidden_dim,
                )
            )
''',
)
remove_line(model, "        time_global_stage_mask: tuple[bool, ...],")
replace_once(
    model,
    '''        if len(time_global_stage_mask) != num_layers:
            raise ValueError(
                "time_global_stage_mask length must equal num_layers, got "
                f"{len(time_global_stage_mask)} and {num_layers}"
            )
''',
    "",
)
replace_once(
    model,
    '''    @staticmethod
    def _normalize_stage_mask(
        values: Sequence[bool],
    ) -> tuple[bool, ...]:
        return tuple(bool(value) for value in values)

''',
    "",
)
replace_once(
    model,
    '''        masks = build_axial_padding_masks(
            padding_mask,
            time_window_radius=self.time_window_radius,
        )
''',
    "        masks = build_axial_padding_masks(padding_mask)\n",
)
remove_line(model, "                masks.sliding_attention_keep_mask,")

padding = "src/tasks/blcs/models/components/padding.py"
replace_once(
    padding,
    '''from src.utils.models.components.ops.time_local import (
    build_local_attention_keep_mask,
)

''',
    "",
)
remove_line(padding, "    sliding_attention_keep_mask: Tensor")
replace_once(
    padding,
    '''def build_axial_padding_masks(
    padding_mask: Tensor,
    *,
    time_window_radius: int,
) -> AxialPaddingMasks:
''',
    '''def build_axial_padding_masks(
    padding_mask: Tensor,
) -> AxialPaddingMasks:
''',
)
replace_once(
    padding,
    '''    if type(time_window_radius) is not int:
        raise TypeError("time_window_radius must be exactly int.")
    if time_window_radius < 0:
        raise ValueError("time_window_radius must be non-negative.")
''',
    "",
)
replace_once(
    padding,
    "    time_attention_keep_mask, repaired_time_valid = build_self_attn_mask(time_valid)\n",
    "    time_attention_keep_mask, _ = build_self_attn_mask(time_valid)\n",
)
replace_once(
    padding,
    '''    sliding_attention_keep_mask = build_local_attention_keep_mask(
        repaired_time_valid,
        time_window_radius,
    )
''',
    "",
)
remove_line(
    padding,
    "        sliding_attention_keep_mask=sliding_attention_keep_mask,",
)

configuration = "src/tasks/blcs/configuration.py"
regex_once(
    configuration,
    r'\ndef _bool_sequence\(value: object, \*, path: str\) -> tuple\[bool, \.\.\.\]:\n.*?\n    return tuple\(cast\("bool", item\) for item in value\)\n',
    "\n",
    flags=re.DOTALL,
)
for line in (
    "    time_window_radius: int",
    "    time_global_stage_mask: tuple[bool, ...]",
    '            "time_window_radius",',
    '            "time_global_stage_mask",',
    '                "time_window_radius": int,',
    '                "time_global_stage_mask": (list, tuple),',
    '            time_window_radius=int(model["time_window_radius"]),',
):
    remove_line(configuration, line)
replace_once(
    configuration,
    '''            time_global_stage_mask=_bool_sequence(
                model["time_global_stage_mask"], path="model.time_global_stage_mask"
            ),
''',
    "",
)
replace_once(
    configuration,
    '''        if not (
            len(result.camera_layers_per_stage)
            == len(result.time_layers_per_stage)
            == len(result.time_global_stage_mask)
            == result.num_layers
        ):
''',
    '''        if not (
            len(result.camera_layers_per_stage)
            == len(result.time_layers_per_stage)
            == result.num_layers
        ):
''',
)
replace_once(
    configuration,
    '''        if result.num_court_tokens <= 0 or result.time_window_radius < 0:
            raise SemanticConfigurationError(
                "model.num_court_tokens must be positive and time_window_radius "
                "must be non-negative."
            )
''',
    '''        if result.num_court_tokens <= 0:
            raise SemanticConfigurationError(
                "model.num_court_tokens must be positive."
            )
''',
)

for yaml_path in (ROOT / "src/tasks/blcs/configs/model").glob(
    "*multiview_axial*.yaml"
):
    text = yaml_path.read_text(encoding="utf-8")
    text = re.sub(
        r"^time_global_stage_mask:.*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^time_window_radius:.*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    yaml_path.write_text(text, encoding="utf-8")

write(
    "src/utils/models/components/ops/loader.py",
    '''from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

MOE_EXTENSION_NAME = "src.utils.models.components.ops.moe._C"
COMPRESSED_TIME_LOCAL_EXTENSION_NAME = (
    "src.utils.models.components.ops.compressed_time_local._C"
)


@lru_cache(maxsize=1)
def get_moe_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(MOE_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=1)
def get_compressed_time_local_cuda_extension() -> ModuleType | None:
    try:
        return importlib.import_module(COMPRESSED_TIME_LOCAL_EXTENSION_NAME)
    except (ImportError, OSError):
        return None


def is_moe_cuda_available() -> bool:
    return get_moe_cuda_extension() is not None


def is_compressed_time_local_cuda_available() -> bool:
    return get_compressed_time_local_cuda_extension() is not None


def require_moe_cuda_extension() -> ModuleType:
    extension = get_moe_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "MoE CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python -m pip install -e . "
            "--no-build-isolation`, or call the API with use_cuda=False."
        )
    return extension


def require_compressed_time_local_cuda_extension() -> ModuleType:
    extension = get_compressed_time_local_cuda_extension()
    if extension is None:
        raise RuntimeError(
            "Compressed time-local CUDA extension is not available. Build it with "
            "`TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python -m pip install -e . "
            "--no-build-isolation`, or select backend='reference'."
        )
    return extension
''',
)
write(
    "src/utils/models/components/ops/__init__.py",
    '''from src.utils.models.components.ops.loader import (
    get_compressed_time_local_cuda_extension,
    get_moe_cuda_extension,
    is_compressed_time_local_cuda_available,
    is_moe_cuda_available,
    require_compressed_time_local_cuda_extension,
    require_moe_cuda_extension,
)
from src.utils.models.components.ops.moe import (
    MoEDispatchResult,
    MoEOperations,
    resolve_moe_operations,
)

__all__ = [
    "MoEDispatchResult",
    "MoEOperations",
    "get_compressed_time_local_cuda_extension",
    "get_moe_cuda_extension",
    "is_compressed_time_local_cuda_available",
    "is_moe_cuda_available",
    "require_compressed_time_local_cuda_extension",
    "require_moe_cuda_extension",
    "resolve_moe_operations",
]
''',
)

build = "src/utils/models/components/ops/build.py"
regex_once(
    build,
    r'''                CUDAExtension\(
                    name="src\.utils\.models\.components\.ops\.time_local\._C",
                    sources=\[
                        str\(build_paths\.time_local_bindings\),
                        str\(build_paths\.time_local_kernels\),
                    \],
                    extra_compile_args=common_compile_args,
                \),
''',
    "",
)

operations = "src/utils/configuration/operations.py"
for line in (
    'FORCE_TIME_LOCAL_REFERENCE = "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE"',
    'USE_TIME_LOCAL_CUDA = "TENNIS_LAB_USE_TIME_LOCAL_CUDA"',
    "    force_time_local_reference: bool",
    "    use_time_local_cuda: bool",
    '        "time_local_bindings": ConfigField.of(str),',
    '        "time_local_kernels": ConfigField.of(str),',
    "    time_local_bindings: Path",
    "    time_local_kernels: Path",
    "            self.time_local_bindings,",
    "            self.time_local_kernels,",
    '            time_local_bindings=resolve(PathRole.PROJECT, "time_local_bindings"),',
    '            time_local_kernels=resolve(PathRole.PROJECT, "time_local_kernels"),',
    '    "FORCE_TIME_LOCAL_REFERENCE",',
    '    "USE_TIME_LOCAL_CUDA",',
    "            force_time_local_reference=parsed[FORCE_TIME_LOCAL_REFERENCE],",
    "            use_time_local_cuda=parsed[USE_TIME_LOCAL_CUDA],",
):
    remove_line(operations, line)
replace_once(
    operations,
    '''_BOOLEAN_NAMES = (
    BUILD_CUDA_OPS,
    FORCE_MOE_REFERENCE,
    FORCE_TIME_LOCAL_REFERENCE,
    USE_TIME_LOCAL_CUDA,
)
_RUNTIME_BOOLEAN_NAMES = (
    FORCE_MOE_REFERENCE,
    FORCE_TIME_LOCAL_REFERENCE,
    USE_TIME_LOCAL_CUDA,
)
''',
    '''_BOOLEAN_NAMES = (
    BUILD_CUDA_OPS,
    FORCE_MOE_REFERENCE,
)
_RUNTIME_BOOLEAN_NAMES = (FORCE_MOE_REFERENCE,)
''',
)
regex_once(
    operations,
    r'''        if parsed\[FORCE_TIME_LOCAL_REFERENCE\] and parsed\[USE_TIME_LOCAL_CUDA\]:
            raise SemanticConfigurationError\(
                f"\{FORCE_TIME_LOCAL_REFERENCE\}=1 conflicts with "
                f"\{USE_TIME_LOCAL_CUDA\}=1\."
            \)
''',
    "",
)

config_init = "src/utils/configuration/__init__.py"
for line in (
    "    FORCE_TIME_LOCAL_REFERENCE,",
    "    USE_TIME_LOCAL_CUDA,",
    '    "FORCE_TIME_LOCAL_REFERENCE",',
    '    "USE_TIME_LOCAL_CUDA",',
):
    remove_line(config_init, line)

install = "scripts/colab/setup/install_cuda_ops.sh"
for line in (
    '            "time_local_bindings": "src/utils/models/components/ops/time_local/csrc/time_local.cpp",',
    '            "time_local_kernels": "src/utils/models/components/ops/time_local/csrc/time_local_cuda.cu",',
):
    remove_line(install, line)

readme = "src/utils/README.md"
regex_once(
    readme,
    r"^- \*\*`components/ops/`\*\*:.*$",
    "- **`components/ops/`**: MoE と compressed time-local attention の CUDA / reference 実装、autograd bridge、extension loader/build。MoE backend/capacity policyは`MoELayer` constructorで固定し、compressed time-local backendはCSWAの構築時に固定する。",
    flags=re.MULTILINE,
)

test_padding = "tests/unit/tasks/blcs/models/components/test_padding.py"
remove_line(test_padding, "        time_window_radius=1,")
remove_line(
    test_padding,
    "    assert masks.sliding_attention_keep_mask.any(dim=-1).all()",
)

test_axial = "tests/unit/tasks/blcs/models/test_blcs_multiview_axial_model.py"
for line in (
    '            "time_global_stage_mask": [False],',
    '            "time_window_radius": 2,',
):
    remove_line(test_axial, line)

test_contract = "tests/unit/tasks/blcs/models/test_padding_contract.py"
for line in (
    "        time_window_radius=1,",
    "        time_global_stage_mask=[False],",
):
    remove_line(test_contract, line)

test_build = "tests/unit/utils/models/components/ops/test_build.py"
for line in (
    '    "TENNIS_LAB_FORCE_TIME_LOCAL_REFERENCE",',
    '    "TENNIS_LAB_USE_TIME_LOCAL_CUDA",',
):
    remove_line(test_build, line)
regex_once(
    test_build,
    r'''        "time_local_bindings": \(
            "src/utils/models/components/ops/time_local/csrc/time_local\.cpp"
        \),
        "time_local_kernels": \(
            "src/utils/models/components/ops/time_local/csrc/time_local_cuda\.cu"
        \),
''',
    "",
)
regex_once(
    test_build,
    r"\ndef test_enabled_setup_spec_load_delegates_environment_conflicts\(\n    tmp_path: Path,\n\) -> None:\n.*?(?=\ndef test_operation_loader_rejects_preloaded_module_from_another_root)",
    "\n",
    flags=re.DOTALL,
)

test_operations = "tests/unit/utils/configuration/test_operations.py"
for line in (
    "    FORCE_TIME_LOCAL_REFERENCE,",
    "    USE_TIME_LOCAL_CUDA,",
    '        FORCE_TIME_LOCAL_REFERENCE: "0",',
    '        USE_TIME_LOCAL_CUDA: "0",',
    "        FORCE_TIME_LOCAL_REFERENCE,",
    "        USE_TIME_LOCAL_CUDA,",
):
    remove_line(test_operations, line)
regex_once(
    test_operations,
    r'''        "time_local_bindings": \(
            "src/utils/models/components/ops/time_local/csrc/time_local\.cpp"
        \),
        "time_local_kernels": \(
            "src/utils/models/components/ops/time_local/csrc/time_local_cuda\.cu"
        \),
''',
    "",
)
replace_once(
    test_operations,
    '''def test_operation_environment_rejects_unknown_and_conflicting_inputs() -> None:
    with pytest.raises(ConfigurationError, match="Unknown configuration"):
        OperationEnvironmentConfig.from_mapping({"TENNIS_LAB_TYPO": "1"})
    with pytest.raises(ConfigurationError, match="conflicts"):
        OperationEnvironmentConfig.from_mapping(
            _operation_environment(
                **{
                    FORCE_TIME_LOCAL_REFERENCE: "1",
                    USE_TIME_LOCAL_CUDA: "1",
                }
            )
        )
''',
    '''def test_operation_environment_rejects_unknown_inputs() -> None:
    with pytest.raises(ConfigurationError, match="Unknown configuration"):
        OperationEnvironmentConfig.from_mapping({"TENNIS_LAB_TYPO": "1"})
''',
)

for relative in (
    "src/utils/models/components/ops/time_local",
    "tests/unit/utils/models/components/ops/time_local",
):
    target = ROOT / relative
    if not target.is_dir():
        raise RuntimeError(f"expected directory to remove: {relative}")
    shutil.rmtree(target)
