"""Repository-wide architecture policy checks after the canonical scene migration."""

from __future__ import annotations

import ast
import re
import subprocess
from collections import Counter, deque
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BoundaryIdentityT = TypeVar("BoundaryIdentityT", bound=Hashable)
BASE_REVISION = "408791ac5697adb89c431a9e4331d173cb01a890"
CONFIG_MERGE_REVISION = "7ea4e16a48137c249763f884b831a82c70b7eb29"
REMOVED_MODULES = (
    "src.configuration_contracts",
    "src.configuration_validation",
    "src.tasks.ball_detection.evaluation.adapters",
    "src.tasks.ball_detection.models.discriminators.trajectory_discriminator",
    "src.tasks.ball_detection.models.input_adapter",
    "src.tasks.ball_detection.training.losses",
    "src.tasks.ball_detection.validation",
    "src.tasks.base.data.scene_chunk_manager",
    "src.tasks.base.inference.grad_mode",
    "src.tasks.base.preview",
    "src.tasks.blcs.models.components.court_ball_point_fusion",
    "src.tasks.blcs.models.discriminators.trajectory_discriminator",
    "src.tasks.blcs.validation",
    "src.tasks.blcs.visualization.adapters",
    "src.tasks.blcs.visualization.adapters.predict_inputs",
    "src.tasks.blcs.visualization.adapters.render_inputs",
    "src.tasks.court_detection.inference.preprocess",
    "src.tasks.court_detection.data.court_kp_dataset",
    "src.tasks.court_detection.data.court_line_dataset",
    "src.tasks.court_detection.data.court_seg_dataset",
    "src.tasks.plcs.utils",
    "src.tasks.plcs.utils.pose_geometry",
    "src.tasks.plcs.models.discriminators.pose_sequence_discriminator",
    "src.tasks.plcs.validation_matrix",
    "src.tasks.plcs.visualization.adapters.predict_inputs",
    "src.tasks.slcs.data.contract",
    "src.tasks.slcs.data.contract_writer",
    "src.tasks.slcs.data.synthetic",
    "src.tennis_scene.io",
    "src.tennis_scene.utils",
    "src.tennis_scene.utils.transforms",
    "src.utils.configuration.validation",
)
PROHIBITED_SYMBOLS = frozenset(
    {
        "CLIP_MANIFEST_NAME",
        "DATASET_INDEX_NAME",
        "DatasetContractError",
        "UnsupportedFormatVersionError",
        "ClipRef",
        "CourtKPDataset",
        "CourtSegDataset",
        "CourtLineDataset",
        "CourtKeypointModelIO",
        "CourtSegModelIO",
        "CourtLineModelIO",
        "_build_track_tensor",
        "_axis_angle_to_matrix",
        "_migrate_legacy_group_embedding_keys",
        "_net_height_at_x",
        "_normalize_valid_mask",
        "_rotation_matrix_z",
        "_sort_tracks",
    }
)
ISSUE_695_REMOVAL_PREFIXES = (
    "src.synthetic_data_generation.",
)
SUPPORTED_TASK_LOCAL_MODULES = frozenset(
    {
        "src.tasks.base.data.chunk_manager",
        "src.tasks.base.data.chunked_datamodule",
        "src.tasks.base.data.dataset_writer",
        "src.tasks.base.data.scene_dataset",
        "src.tasks.base.training.chunk_rotation_callback",
        "src.tasks.blcs.data.chunk_manager",
        "src.tasks.blcs.generate_dataset.io.dataset_io",
        "src.tasks.blcs.scripts.generate_dataset",
        "src.tasks.blcs.scripts.preview_augmentation",
        "src.tasks.blcs.scripts.visualize",
        "src.tasks.blcs.visualization.orchestrator",
        "src.tasks.plcs.data.chunk_manager",
        "src.tasks.plcs.generate_dataset.io.dataset_io",
        "src.tasks.plcs.scripts.generate_dataset",
        "src.tasks.plcs.scripts.preview_augmentation",
        "src.tasks.plcs.scripts.visualize",
        "src.tasks.plcs.visualization.orchestrator",
        "src.utils.data.scene_io",
    }
)
COURT_LINE_PREPROCESSING_CONSUMERS = {
    "src/synthetic_data_generation/alignment/evidence_source.py": (
        "predictor.adapter.spec.short_side",
        "predictor.short_side",
    ),
}
EXPECTED_DIRECT_FORWARD_VALIDATION_BOUNDARIES = {
    (
        "src.tasks.plcs.models.plcs_track_query_model."
        "PLCSTrackQueryModel.build_spatial_coordinates",
        "Python raise",
    ): 1,
    (
        "src.utils.models.architectures.transformer_sequence_discriminator."
        "TransformerSequenceDiscriminator.forward",
        "Python raise",
    ): 9,
    (
        "src.utils.models.architectures.transformer_sequence_discriminator."
        "TransformerSequenceDiscriminator.forward",
        "Python shape/value validation branch",
    ): 5,
    (
        "src.utils.models.architectures.transformer_sequence_discriminator."
        "TransformerSequenceDiscriminator.forward",
        "runtime implementation/type selection via isinstance",
    ): 2,
}
BLCS_SINGLE_VIEW_MASK_PATH = (
    "src.tasks.blcs.models.blcs_model.BLCSModel.forward",
    "src.tasks.blcs.models.components.padding.build_single_view_padding_masks",
)
BLCS_SINGLE_VIEW_COURT_COUNT_VALIDATION_PATH = (
    *BLCS_SINGLE_VIEW_MASK_PATH,
    "src.tasks.blcs.models.components.padding._validate_num_court_tokens",
)
BLCS_SINGLE_VIEW_PADDING_VALIDATION_PATH = (
    *BLCS_SINGLE_VIEW_MASK_PATH,
    "src.tasks.blcs.models.components.padding._validate_padding_mask",
)
BLCS_SINGLE_VIEW_OUTPUT_MASK_PATH = (
    "src.tasks.blcs.models.blcs_model.BLCSModel.forward",
    "src.tasks.blcs.models.components.padding.mask_trajectory_outputs",
)
BLCS_AXIAL_MASK_PATH = (
    "src.tasks.blcs.models.blcs_multiview_axial_model."
    "BLCSMultiViewAxialModel.forward",
    "src.tasks.blcs.models.components.padding.build_axial_padding_masks",
)
BLCS_AXIAL_PADDING_VALIDATION_PATH = (
    *BLCS_AXIAL_MASK_PATH,
    "src.tasks.blcs.models.components.padding._validate_padding_mask",
)
BLCS_AXIAL_OUTPUT_MASK_PATH = (
    "src.tasks.blcs.models.blcs_multiview_axial_model."
    "BLCSMultiViewAxialModel.forward",
    "src.tasks.blcs.models.components.padding.mask_trajectory_outputs",
)
BLCS_MULTIVIEW_MASK_PATH = (
    "src.tasks.blcs.models.blcs_multiview_model.BLCSMultiViewModel.forward",
    "src.tasks.blcs.models.components.padding.build_multiview_padding_masks",
)
BLCS_MULTIVIEW_COURT_COUNT_VALIDATION_PATH = (
    *BLCS_MULTIVIEW_MASK_PATH,
    "src.tasks.blcs.models.components.padding._validate_num_court_tokens",
)
BLCS_MULTIVIEW_PADDING_VALIDATION_PATH = (
    *BLCS_MULTIVIEW_MASK_PATH,
    "src.tasks.blcs.models.components.padding._validate_padding_mask",
)
BLCS_MULTIVIEW_OUTPUT_MASK_PATH = (
    "src.tasks.blcs.models.blcs_multiview_model.BLCSMultiViewModel.forward",
    "src.tasks.blcs.models.components.padding.mask_trajectory_outputs",
)
BLCS_FIXED_QUERY_MASK_PATH = (
    "src.tasks.blcs.models.blcs_track_query_model.BLCSTrackQueryModel.forward",
    "src.utils.models.multiview_padding.build_fixed_query_padding_masks",
)
PLCS_SPATIAL_COORDINATE_VALIDATION_PATH = (
    "src.tasks.plcs.models.plcs_track_query_model.PLCSTrackQueryModel.forward",
    "src.tasks.plcs.models.plcs_track_query_model."
    "PLCSTrackQueryModel.build_spatial_coordinates",
)
PLCS_FIXED_QUERY_MASK_PATH = (
    "src.tasks.plcs.models.plcs_track_query_model.PLCSTrackQueryModel.forward",
    "src.utils.models.multiview_padding.build_fixed_query_padding_masks",
)
SLCS_MASK_PATH = (
    "src.tasks.slcs.models.slcs_model.SLCSFusionModel.forward",
    "src.tasks.slcs.models.components.padding.build_slcs_padding_masks",
)
SLCS_PADDING_VALIDATION_PATH = (
    *SLCS_MASK_PATH,
    "src.tasks.slcs.models.components.padding._validate_padding_mask",
)
TRANSFORMER_SEQUENCE_DISCRIMINATOR_PATH = (
    "src.utils.models.architectures.transformer_sequence_discriminator."
    "TransformerSequenceDiscriminator.forward",
)
EXPECTED_TRANSITIVE_FORWARD_VALIDATION_BOUNDARIES_BY_PATH = {
    BLCS_SINGLE_VIEW_MASK_PATH: {
        "forward validation helper _validate_num_court_tokens": 1,
        "forward validation helper _validate_padding_mask": 1,
    },
    BLCS_SINGLE_VIEW_COURT_COUNT_VALIDATION_PATH: {
        "Python raise": 2,
        "Python shape/value validation branch": 1,
        "runtime implementation/type selection via type": 1,
    },
    BLCS_SINGLE_VIEW_PADDING_VALIDATION_PATH: {
        "Python raise": 4,
        "Python shape/value validation branch": 3,
        "runtime implementation/type selection via isinstance": 1,
    },
    BLCS_SINGLE_VIEW_OUTPUT_MASK_PATH: {
        "Python raise": 1,
        "Python shape/value validation branch": 1,
    },
    BLCS_AXIAL_MASK_PATH: {
        "Python raise": 2,
        "Python shape/value validation branch": 1,
        "forward validation helper _validate_padding_mask": 1,
        "runtime implementation/type selection via type": 1,
    },
    BLCS_AXIAL_PADDING_VALIDATION_PATH: {
        "Python raise": 4,
        "Python shape/value validation branch": 3,
        "runtime implementation/type selection via isinstance": 1,
    },
    BLCS_AXIAL_OUTPUT_MASK_PATH: {
        "Python raise": 1,
        "Python shape/value validation branch": 1,
    },
    BLCS_MULTIVIEW_MASK_PATH: {
        "forward validation helper _validate_num_court_tokens": 1,
        "forward validation helper _validate_padding_mask": 1,
    },
    BLCS_MULTIVIEW_COURT_COUNT_VALIDATION_PATH: {
        "Python raise": 2,
        "Python shape/value validation branch": 1,
        "runtime implementation/type selection via type": 1,
    },
    BLCS_MULTIVIEW_PADDING_VALIDATION_PATH: {
        "Python raise": 4,
        "Python shape/value validation branch": 3,
        "runtime implementation/type selection via isinstance": 1,
    },
    BLCS_MULTIVIEW_OUTPUT_MASK_PATH: {
        "Python raise": 1,
        "Python shape/value validation branch": 1,
    },
    BLCS_FIXED_QUERY_MASK_PATH: {
        "Python raise": 6,
        "Python shape/value validation branch": 4,
        "runtime implementation/type selection via isinstance": 1,
        "runtime implementation/type selection via type": 1,
    },
    PLCS_SPATIAL_COORDINATE_VALIDATION_PATH: {"Python raise": 1},
    PLCS_FIXED_QUERY_MASK_PATH: {
        "Python raise": 6,
        "Python shape/value validation branch": 4,
        "runtime implementation/type selection via isinstance": 1,
        "runtime implementation/type selection via type": 1,
    },
    SLCS_MASK_PATH: {
        "Python raise": 2,
        "Python shape/value validation branch": 2,
        "forward validation helper _validate_padding_mask": 2,
        "runtime implementation/type selection via type": 2,
    },
    SLCS_PADDING_VALIDATION_PATH: {
        "Python raise": 5,
        "Python shape/value validation branch": 4,
        "runtime implementation/type selection via isinstance": 1,
    },
    TRANSFORMER_SEQUENCE_DISCRIMINATOR_PATH: {
        "Python raise": 9,
        "Python shape/value validation branch": 5,
        "runtime implementation/type selection via isinstance": 2,
    },
}
EXPECTED_TRANSITIVE_FORWARD_VALIDATION_BOUNDARIES = {
    (call_path, violation): count
    for call_path, expected in (
        EXPECTED_TRANSITIVE_FORWARD_VALIDATION_BOUNDARIES_BY_PATH.items()
    )
    for violation, count in expected.items()
}
COMPATIBILITY_RETENTION_PATTERN = re.compile(
    r"\b(?:retain(?:ed|ing)?|ke(?:ep|pt)|preserv(?:e|ed|ing))\b"
    r"[^.!?\n]{0,100}\bfor compatibility\b",
    flags=re.IGNORECASE,
)


def _git_paths(*arguments: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(path for path in completed.stdout.splitlines() if path)


def _repository_python_files() -> tuple[Path, ...]:
    paths = _git_paths(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src/**/*.py",
    )
    return tuple(
        REPOSITORY_ROOT / path
        for path in paths
        if not path.startswith("src/submodules/vendor/")
        and (REPOSITORY_ROOT / path).is_file()
    )


def _repository_consumer_files() -> tuple[Path, ...]:
    paths = _git_paths(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        ".spin",
        "experiments",
    )
    suffixes = {".json", ".py", ".pyi", ".sh", ".toml", ".yaml", ".yml"}
    return tuple(
        REPOSITORY_ROOT / path
        for path in paths
        if not path.startswith("src/submodules/vendor/")
        and (REPOSITORY_ROOT / path).is_file()
        and (REPOSITORY_ROOT / path).suffix in suffixes
    )


def _repository_consumer_python_files() -> tuple[Path, ...]:
    return tuple(
        path for path in _repository_consumer_files() if path.suffix == ".py"
    )


def _repository_reference_python_files() -> tuple[Path, ...]:
    paths = _git_paths(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        ".spin",
        "experiments",
        "tests",
    )
    return tuple(
        REPOSITORY_ROOT / path
        for path in paths
        if path.endswith(".py")
        and not path.startswith("src/submodules/vendor/")
        and (REPOSITORY_ROOT / path).is_file()
    )


def _module_name_from_source_path(path: str) -> str:
    source = Path(path).with_suffix("")
    parts = source.parts[:-1] if source.name == "__init__" else source.parts
    return ".".join(parts)


def _source_module_name(path: Path) -> str:
    return _module_name_from_source_path(str(path.relative_to(REPOSITORY_ROOT)))


def _deleted_repository_modules() -> frozenset[str]:
    paths = _git_paths(
        "diff",
        "--no-renames",
        "--diff-filter=D",
        "--name-only",
        BASE_REVISION,
        "--",
        "src",
    )
    modules = frozenset(
        _module_name_from_source_path(path)
        for path in paths
        if path.endswith(".py")
    )
    return frozenset(module for module in modules if _module_path(module) is None)


def _source_reference(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(REPOSITORY_ROOT)}:{getattr(node, 'lineno', 0)}"


def _qualified_name(node: ast.expr | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name(node: ast.Call) -> str:
    return _qualified_name(node.func)


def _declaration_retains_compatibility(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    docstring = ast.get_docstring(node, clean=False)
    return docstring is not None and COMPATIBILITY_RETENTION_PATTERN.search(
        docstring
    ) is not None


def _function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    arguments = node.args
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return frozenset(names)


def _is_passthrough_argument(node: ast.expr, parameters: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in parameters
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.Starred) and _is_passthrough_argument(
        node.value,
        parameters,
    )


def _is_delegating_call(
    node: ast.expr,
    *,
    member_name: str,
    parameters: frozenset[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    target = _qualified_name(node.func)
    if not target or target.rsplit(".", maxsplit=1)[-1] == member_name:
        return False
    return all(
        _is_passthrough_argument(argument, parameters) for argument in node.args
    ) and all(
        _is_passthrough_argument(keyword.value, parameters)
        for keyword in node.keywords
    )


def _attribute_root(node: ast.Attribute) -> ast.expr:
    value: ast.expr = node
    while isinstance(value, ast.Attribute):
        value = value.value
    return value


def _is_direct_forward_value(
    node: ast.expr,
    *,
    member_name: str,
    parameters: frozenset[str],
) -> bool:
    if _is_delegating_call(
        node,
        member_name=member_name,
        parameters=parameters,
    ):
        return True
    if isinstance(node, ast.Name):
        return node.id in parameters
    if isinstance(node, ast.Attribute):
        root = _attribute_root(node)
        return isinstance(root, ast.Name) and root.id in {"self", "cls"}
    return False


def _assigned_names(node: ast.expr) -> frozenset[str]:
    if isinstance(node, ast.Name):
        return frozenset({node.id})
    if isinstance(node, (ast.Tuple, ast.List)):
        return frozenset(
            name
            for element in node.elts
            for name in _assigned_names(element)
        )
    return frozenset()


def _is_binding_projection(node: ast.expr, bindings: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in bindings
    if isinstance(node, ast.Attribute):
        return _is_binding_projection(node.value, bindings)
    if isinstance(node, ast.Subscript):
        return _is_binding_projection(node.value, bindings)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_binding_projection(element, bindings) for element in node.elts)
    return False


def _is_private_non_dunder(name: str) -> bool:
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def _is_pure_forwarding_member(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if not _is_private_non_dunder(node.name):
        return False
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    parameters = _function_parameters(node)
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value:
        return _is_direct_forward_value(
            body[0].value,
            member_name=node.name,
            parameters=parameters,
        )
    if len(body) != 2 or not isinstance(body[1], ast.Return) or body[1].value is None:
        return False
    assignment = body[0]
    if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1:
        target = assignment.targets[0]
        value = assignment.value
    elif isinstance(assignment, ast.AnnAssign) and assignment.value is not None:
        target = assignment.target
        value = assignment.value
    else:
        return False
    bindings = _assigned_names(target)
    return (
        bool(bindings)
        and _is_delegating_call(
            value,
            member_name=node.name,
            parameters=parameters,
        )
        and _is_binding_projection(body[1].value, bindings)
    )


def _referenced_member_names(trees: Iterable[ast.Module]) -> frozenset[str]:
    references: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                references.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                references.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                references.add(node.value)
    return frozenset(references)


def _unused_private_forwarders(
    tree: ast.Module,
    referenced_names: frozenset[str],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        member
        for class_node in ast.walk(tree)
        if isinstance(class_node, ast.ClassDef)
        for member in class_node.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and member.name not in referenced_names
        and _is_pure_forwarding_member(member)
    )


def _is_main_guard(node: ast.If) -> bool:
    return (
        isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_file = REPOSITORY_ROOT / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = REPOSITORY_ROOT / relative / "__init__.py"
    if package_file.is_file():
        return package_file
    return None


def _changed_or_untracked_paths(scope: str) -> set[str]:
    changed = set(
        _git_paths("diff", "--name-only", BASE_REVISION, "--", scope)
    )
    changed.update(
        _git_paths("ls-files", "--others", "--exclude-standard", "--", scope)
    )
    return changed


def test_branch_contains_configuration_contract_merge() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CONFIG_MERGE_REVISION, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_reserved_vendor_scope_contains_only_the_public_nht_gitlink() -> None:
    assert _changed_or_untracked_paths("third_party") == {"third_party/nht"}
    mode = _git_paths("ls-files", "-s", "--", "third_party/nht")
    assert len(mode) == 1 and mode[0].startswith("160000 ")
    assert not _changed_or_untracked_paths("src/submodules/vendor")


def test_court_line_preprocessing_size_has_one_public_surface() -> None:
    predictor_path = (
        REPOSITORY_ROOT
        / "src/tasks/court_detection/inference/mask_predictor.py"
    )
    predictor_tree = ast.parse(
        predictor_path.read_text(encoding="utf-8"),
        filename=str(predictor_path),
    )
    predictor_class = next(
        node
        for node in predictor_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CourtLinePredictor"
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "short_side"
        for node in predictor_class.body
    ), "CourtLinePredictor.short_side must not be restored as a compatibility shim"

    for relative, (canonical_surface, stale_surface) in (
        COURT_LINE_PREPROCESSING_CONSUMERS.items()
    ):
        path = REPOSITORY_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        attribute_names = {
            _qualified_name(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        assert canonical_surface in attribute_names, (
            f"{relative} must consume CourtModelSpec.short_side through the "
            "selected bundle-aware CourtModelIOAdapter"
        )
        assert stale_surface not in attribute_names, (
            f"{relative} still consumes the removed CourtLinePredictor.short_side"
        )


def test_library_main_guards_are_absent() -> None:
    violations: list[str] = []
    for path in _repository_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(REPOSITORY_ROOT)
        allowed = "scripts" in relative.parts or path.name == "__main__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_main_guard(node) and not allowed:
                violations.append(_source_reference(path, node))
    assert not violations, "library main guards:\n" + "\n".join(violations)


def _reachable_forward_functions(
    tree: ast.Module,
) -> Iterable[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forward = methods.get("forward")
        if forward is None:
            continue
        pending = deque([forward])
        visited: set[str] = set()
        while pending:
            function = pending.popleft()
            identity = f"{class_node.name}.{function.name}"
            if identity in visited:
                continue
            visited.add(identity)
            yield function, identity
            for child in ast.walk(function):
                if not isinstance(child, ast.Call):
                    continue
                call = _call_name(child)
                target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
                if call.startswith("self."):
                    target = methods.get(call.removeprefix("self."))
                elif "." not in call:
                    target = module_functions.get(call)
                if target is not None:
                        pending.append(target)


def _absolute_import_module(
    *,
    current_module: str,
    current_path: Path,
    imported_module: str | None,
    level: int,
) -> str:
    if level == 0:
        return imported_module or ""
    current_parts = current_module.split(".")
    package_parts = (
        current_parts
        if current_path.name == "__init__.py"
        else current_parts[:-1]
    )
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        return ""
    suffix = [] if imported_module is None else imported_module.split(".")
    return ".".join([*package_parts[:keep], *suffix])


def _repository_function_index() -> tuple[
    dict[
        tuple[str, str | None, str],
        tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef],
    ],
    dict[str, dict[str, str]],
]:
    functions: dict[
        tuple[str, str | None, str],
        tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef],
    ] = {}
    imports: dict[str, dict[str, str]] = {}
    for path in _repository_python_files():
        module = _source_module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_imports: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[(module, None, node.name)] = (path, node)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions[(module, node.name, child.name)] = (path, child)
            elif isinstance(node, ast.ImportFrom):
                imported_module = _absolute_import_module(
                    current_module=module,
                    current_path=path,
                    imported_module=node.module,
                    level=node.level,
                )
                if imported_module.startswith("src"):
                    for alias in node.names:
                        if alias.name != "*":
                            module_imports[alias.asname or alias.name] = (
                                f"{imported_module}.{alias.name}"
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src") and alias.asname:
                        module_imports[alias.asname] = alias.name
        imports[module] = module_imports
    return functions, imports


def _repository_function_target(
    qualified_symbol: str,
    *,
    functions: dict[
        tuple[str, str | None, str],
        tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef],
    ],
    imports: dict[str, dict[str, str]],
    seen: frozenset[str] = frozenset(),
) -> tuple[str, str | None, str] | None:
    if qualified_symbol in seen:
        return None
    parts = qualified_symbol.split(".")
    modules = set(imports)
    for split_at in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:split_at])
        remaining = parts[split_at:]
        if module not in modules or len(remaining) != 1:
            continue
        symbol = remaining[0]
        direct = (module, None, symbol)
        if direct in functions:
            return direct
        reexport = imports[module].get(symbol)
        if reexport is not None:
            return _repository_function_target(
                reexport,
                functions=functions,
                imports=imports,
                seen=seen | {qualified_symbol},
            )
        return None
    return None


def _transitive_forward_functions() -> Iterable[
    tuple[
        Path,
        ast.FunctionDef | ast.AsyncFunctionDef,
        tuple[str, ...],
    ]
]:
    functions, imports = _repository_function_index()
    roots: list[tuple[str, str | None, str]] = []
    for identity, (path, _) in functions.items():
        module, class_name, function_name = identity
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        module_owner = (
            relative.startswith("src/utils/models/")
            or relative.startswith("src/utils/losses/")
            or "/models/" in relative
            or "/training/" in relative
        )
        if class_name is not None and function_name == "forward" and module_owner:
            roots.append(identity)

    for root in roots:
        module, class_name, _ = root
        if class_name is None:
            continue
        pending: deque[
            tuple[tuple[str, str | None, str], tuple[str, ...]]
        ] = deque([(root, (f"{module}.{class_name}.forward",))])
        visited: set[tuple[str, str | None, str]] = set()
        while pending:
            identity, call_path = pending.popleft()
            if identity in visited:
                continue
            visited.add(identity)
            path, function = functions[identity]
            yield path, function, call_path
            module, class_name, _ = identity
            module_imports = imports[module]
            for child in ast.walk(function):
                if not isinstance(child, ast.Call):
                    continue
                call = _call_name(child)
                target: tuple[str, str | None, str] | None = None
                if class_name is not None and call.startswith("self."):
                    method = call.removeprefix("self.")
                    if "." not in method:
                        candidate = (module, class_name, method)
                        target = candidate if candidate in functions else None
                elif "." not in call:
                    local = (module, None, call)
                    if local in functions:
                        target = local
                    elif call in module_imports:
                        target = _repository_function_target(
                            module_imports[call],
                            functions=functions,
                            imports=imports,
                        )
                else:
                    prefix, _, suffix = call.partition(".")
                    qualified = (
                        f"{module_imports[prefix]}.{suffix}"
                        if prefix in module_imports
                        else call
                    )
                    if qualified.startswith("src."):
                        target = _repository_function_target(
                            qualified,
                            functions=functions,
                            imports=imports,
                        )
                if target is not None:
                    target_name = ".".join(
                        part for part in target if part is not None
                    )
                    pending.append((target, (*call_path, target_name)))


def _condition_uses_python_validation(node: ast.expr) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in {"ndim", "shape"}:
            return True
        if isinstance(child, ast.Call):
            call = _call_name(child)
            if call in {"bool", "float", "int", "isinstance", "len", "type"}:
                return True
            if call.endswith((".dim", ".item")):
                return True
    return False


def _forward_violation(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assert):
        return "Python assert"
    if isinstance(node, ast.Raise):
        return "Python raise"
    if isinstance(node, (ast.If, ast.IfExp, ast.While)) and _condition_uses_python_validation(
        node.test
    ):
        return "Python shape/value validation branch"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets: list[ast.expr] = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            for target in targets
        ):
            return "module-state mutation"
    if not isinstance(node, ast.Call):
        return None
    call = _call_name(node)
    leaf = call.rsplit(".", maxsplit=1)[-1]
    if call in {"callable", "getattr", "hasattr", "isinstance", "setattr", "type"}:
        return f"runtime implementation/type selection via {call}"
    if leaf.startswith(("check_", "ensure_", "require_", "validate_")) or leaf.startswith(
        ("_check", "_ensure", "_require", "_validate")
    ):
        return f"forward validation helper {call}"
    if call in {
        "open",
        "print",
        "subprocess.Popen",
        "subprocess.run",
        "torch.load",
        "torch.save",
    } or call.startswith(("logger.", "logging.", "requests.")):
        return f"forward side effect {call}"
    if call.startswith(("nn.", "torch.nn.")) and leaf[:1].isupper():
        return f"module construction {call}"
    return None


def _assert_exact_forward_validation_boundaries(
    actual: Counter[tuple[BoundaryIdentityT, str]],
    expected: Mapping[tuple[BoundaryIdentityT, str], int],
    *,
    heading: str,
) -> None:
    expected_counter = Counter(expected)
    unexpected = actual - expected_counter
    missing = expected_counter - actual
    details = [
        *(
            f"unexpected {identity}: {violation} ({count})"
            for (identity, violation), count in sorted(
                unexpected.items(), key=lambda item: repr(item[0])
            )
        ),
        *(
            f"missing {identity}: {violation} ({count})"
            for (identity, violation), count in sorted(
                missing.items(), key=lambda item: repr(item[0])
            )
        ),
    ]
    assert not details, f"{heading}:\n" + "\n".join(details)


def test_repository_owned_forwards_are_computation_only() -> None:
    violations: Counter[tuple[str, str]] = Counter()
    for path in _repository_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function, identity in _reachable_forward_functions(tree):
            for node in ast.walk(function):
                violation = _forward_violation(node)
                if violation is not None:
                    qualified_identity = f"{_source_module_name(path)}.{identity}"
                    violations[(qualified_identity, violation)] += 1
    _assert_exact_forward_validation_boundaries(
        violations,
        EXPECTED_DIRECT_FORWARD_VALIDATION_BOUNDARIES,
        heading="forward contract violations",
    )


def test_transitive_repository_forward_helpers_are_computation_only() -> None:
    violations: Counter[tuple[tuple[str, ...], str]] = Counter()
    for _path, function, call_path in _transitive_forward_functions():
        for node in ast.walk(function):
            violation = _forward_violation(node)
            if violation is not None:
                violations[(call_path, violation)] += 1
    _assert_exact_forward_validation_boundaries(
        violations,
        EXPECTED_TRANSITIVE_FORWARD_VALIDATION_BOUNDARIES,
        heading="transitive forward contract violations",
    )


def test_forward_validation_boundary_controls_freeze_paths_and_counts() -> None:
    reason = "Python raise"
    expected_path: tuple[str, ...] = (
        "src.example.Model.forward",
        "src.example.validate_input",
    )
    expected: dict[tuple[tuple[str, ...], str], int] = {
        (expected_path, reason): 1
    }
    mutations: tuple[Counter[tuple[tuple[str, ...], str]], ...] = (
        Counter(
            {
                (
                    ("src.other.Model.forward", "src.example.validate_input"),
                    reason,
                ): 1
            }
        ),
        Counter(
            {
                (
                    (
                        "src.example.Model.forward",
                        "src.example.bridge",
                        "src.example.validate_input",
                    ),
                    reason,
                ): 1
            }
        ),
        Counter({(expected_path, reason): 2}),
    )

    for mutation in mutations:
        try:
            _assert_exact_forward_validation_boundaries(
                mutation,
                expected,
                heading="mutation",
            )
        except AssertionError as error:
            assert "unexpected" in str(error)
        else:
            raise AssertionError("rerouted or recounted validation was accepted")


def test_transitive_forward_inventory_keeps_shared_helper_roots_distinct() -> None:
    shared_helper = (
        "src.utils.models.multiview_padding.build_fixed_query_padding_masks"
    )
    discovered = {
        call_path
        for _, _, call_path in _transitive_forward_functions()
        if call_path[-1] == shared_helper
    }

    assert discovered == {
        BLCS_FIXED_QUERY_MASK_PATH,
        PLCS_FIXED_QUERY_MASK_PATH,
    }


def test_transitive_forward_inventory_crosses_repository_modules() -> None:
    call_paths = {
        call_path for _, _, call_path in _transitive_forward_functions()
    }
    expected_cross_module_paths = {
        (
            "src.tasks.ball_detection.models.conv_next_unet.StemLayer.forward",
            "src.utils.tensor_utils.flatten_time_to_batch",
        ),
        (
            "src.utils.models.components.attention.MultiHeadSelfAttention.forward",
            "src.utils.models.components.attention.MultiHeadSelfAttention._apply_rope",
            "src.utils.models.components.rope.apply_rotary_emb",
        ),
    }
    assert expected_cross_module_paths <= call_paths


def test_removed_modules_have_no_forwarding_path_or_owned_reference() -> None:
    deleted = _deleted_repository_modules()
    original = frozenset(REMOVED_MODULES)
    assert original <= deleted
    unexpected = {
        module
        for module in deleted - original
        if not module.startswith(ISSUE_695_REMOVAL_PREFIXES)
    }
    assert not unexpected, f"deletions outside the canonical migration: {unexpected}"

    missing = [module for module in REMOVED_MODULES if _module_path(module) is not None]
    assert not missing, f"removed modules still exist: {missing}"

    stale: list[str] = []
    for path in _repository_consumer_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for module in REMOVED_MODULES:
            references = (module, module.replace(".", "/"))
            if any(reference in text for reference in references):
                stale.append(f"{path.relative_to(REPOSITORY_ROOT)}: {module}")
    assert not stale, "stale removed-module references:\n" + "\n".join(stale)


def test_task_local_generation_and_chunk_consumers_remain_supported() -> None:
    missing = sorted(
        module for module in SUPPORTED_TASK_LOCAL_MODULES if _module_path(module) is None
    )

    assert not missing, "supported task-local modules are missing:\n" + "\n".join(
        missing
    )


def test_compatibility_symbols_are_not_defined_or_reexported() -> None:
    violations: list[str] = []
    for path in _repository_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            bindings: set[str] = set()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bindings.update(alias.asname or alias.name.rsplit(".", maxsplit=1)[-1] for alias in node.names)
            elif isinstance(node, ast.Assign):
                bindings.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bindings.add(node.target.id)
            for binding in bindings & PROHIBITED_SYMBOLS:
                violations.append(f"{_source_reference(path, node)}: {binding}")
    assert not violations, "compatibility symbols:\n" + "\n".join(violations)


def test_compatibility_declarations_and_unused_private_forwarders_are_absent() -> None:
    reference_trees = tuple(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in _repository_reference_python_files()
    )
    referenced_names = _referenced_member_names(reference_trees)
    violations: list[str] = []
    for path in _repository_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ) and _declaration_retains_compatibility(node):
                violations.append(
                    f"{_source_reference(path, node)} {node.name}: "
                    "declaration explicitly retains compatibility"
                )
        for member in _unused_private_forwarders(tree, referenced_names):
            violations.append(
                f"{_source_reference(path, member)} {member.name}: "
                "unused private pure-forwarding member"
            )
    assert not violations, "compatibility declarations/forwarders:\n" + "\n".join(
        violations
    )


def test_compatibility_forwarder_policy_controls_are_bounded() -> None:
    positive = ast.parse(
        '''
class LegacyAdapter:
    @property
    def _legacy_value(self):
        """Value preserved as the old spelling for compatibility."""
        return self.canonical_value

    def _legacy_call(self, value, **kwargs):
        return self.canonical_call(value, **kwargs)
'''
    )
    positive_references = _referenced_member_names((positive,))
    assert {
        member.name
        for member in _unused_private_forwarders(positive, positive_references)
    } == {"_legacy_call", "_legacy_value"}
    assert any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_legacy_value"
        and _declaration_retains_compatibility(node)
        for node in ast.walk(positive)
    )
    for docstring in (
        "Retaining this old adapter for compatibility.",
        "Kept for compatibility.",
        "Preserved under its former name for compatibility.",
    ):
        declaration = ast.parse(
            f'def legacy():\n    """{docstring}"""\n    pass\n'
        ).body[0]
        assert isinstance(declaration, ast.FunctionDef)
        assert _declaration_retains_compatibility(declaration)

    allowed = ast.parse(
        '''
class ActiveAdapters:
    def _invoked_adapter(self, value):
        return self.canonical_call(value)

    def _transforming_adapter(self, value):
        """Transforms input without retaining a compatibility fallback."""
        return self.canonical_call(value + 1)

    def run(self, value):
        return self._invoked_adapter(value)
'''
    )
    allowed_references = _referenced_member_names((allowed,))
    assert not _unused_private_forwarders(allowed, allowed_references)
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and _declaration_retains_compatibility(node)
        for node in ast.walk(allowed)
    )


def test_static_src_import_modules_exist() -> None:
    missing: list[str] = []
    for path in _repository_consumer_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = (node.module,)
            for module in modules:
                if module == "src" or not module.startswith("src."):
                    continue
                if _module_path(module) is None:
                    missing.append(f"{_source_reference(path, node)}: {module}")
    assert not missing, "unresolved repository imports:\n" + "\n".join(missing)


def test_repository_cli_and_dynamic_target_modules_exist() -> None:
    patterns = (
        re.compile(r"\bpython(?:3)?\s+-m\s+(src(?:\.[A-Za-z_]\w*)+)"),
        re.compile(r"(?m)^\s*_target_:\s*['\"]?(src(?:\.[A-Za-z_]\w*)+)"),
    )
    discovered: list[tuple[Path, str]] = []
    missing: list[str] = []
    for path in _repository_consumer_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            for match in pattern.finditer(text):
                module = match.group(1)
                discovered.append((path, module))
                if _module_path(module) is None:
                    missing.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}: {module}"
                    )

    assert discovered, "consumer scan found no CLI or dynamic-target modules"
    assert not missing, "unresolved CLI/dynamic modules:\n" + "\n".join(missing)
