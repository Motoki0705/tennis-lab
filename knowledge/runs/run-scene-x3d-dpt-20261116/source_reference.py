"""Load the verbatim training DPT class for a same-device paired comparison.

The reference exists only in the experiment output directory. Production keeps
one encoder/model implementation. Helpers used below are separately checked
against their training source hashes in source-parity.json.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess

from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig


def build_reference(saved, directory: Path):
    original = subprocess.check_output([
        "git", "show", "13dff57d4:src/tasks/ball_detection/models/mdd/detectors.py"
    ], text=True)
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MDDDPTDetector")
    body = ast.get_source_segment(original, node)
    header = '''from __future__ import annotations
from torch import Tensor, nn
from src.tasks.ball_detection.models.mdd.config import MDDPretrainConfig
from src.tasks.ball_detection.models.mdd.cnn_encoder import build_mdd_encoder
from src.tasks.ball_detection.preprocessing.rgb_mdd import build_input_transform
from src.tasks.ball_detection.models.mdd.execution import decode_frame_heatmaps
from src.utils.models.dpt import DPTDecoder
'''
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_reference_dpt.py"
    path.write_text(header + "\n" + body + "\n")
    spec = importlib.util.spec_from_file_location("training_reference_dpt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = MDDHeatmapConfig(**saved["model_config"])
    reference = module.MDDDPTDetector(config, mdd_a=saved["input_contract"]["mdd_a"], mdd_b=saved["input_contract"]["mdd_b"])
    reference.load_state_dict(saved["state_dict"], strict=True)
    return reference, dict(training_commit="13dff57d4", class_source_sha256=hashlib.sha256(body.encode()).hexdigest(),
                           generated_source=str(path), note="Same local GPU/input/runtime; not the historical L4 output comparison.")
