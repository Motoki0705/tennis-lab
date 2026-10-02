"""Replay GPU generation with archived settings and an explicitly rebuilt operator."""
import json
import sys
from pathlib import Path
from src.tennis_scene.chat_annotation.player_pose.selection import initialize
from src.tennis_scene.chat_annotation.player_pose.generation import generate_clip
from src.tennis_scene.chat_annotation.player_pose.storage import digest
replay = Path(sys.argv[1])
repository = Path(sys.argv[2])
bundle = Path(__file__).parent
saved = json.loads((bundle / "campaign-inputs/config.json").read_text())
for filename, expected in saved["store_hashes"].items():
    if digest(Path(saved["store"]) / filename) != expected:
        raise ValueError("Original ball store changed")
for name, expected in saved["asset_hashes"].items():
    if name != "dino_extension" and digest(Path(saved["assets"][name])) != expected:
        raise ValueError(f"Original model weight changed: {name}")
config = {k:v for k,v in saved.items() if k not in ("code_hashes", "asset_hashes", "asset_stats", "store_hashes")}
config.update(project_root=str(repository), dataset=str(replay / "dataset"), python=str(repository / ".venv/bin/python"))
config["assets"]["dino_extension"] = str(replay / "operator/lib/MultiScaleDeformableAttention.so")
campaign = replay / "campaign"
initialize(campaign, config)
generate_clip(campaign, 34)
print("Replay uses the recorded model settings and a newly fingerprinted CUDA operator; byte equality is not claimed.")
