"""Content-bound review predictions, separate from immutable training runs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ball_refiner_3d.review_predictions.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_root(outputs_root: Path) -> Path:
    return outputs_root / "review" / "predictions"


def bundle_path(root: Path, checkpoint_hash: str, profile: dict[str, Any]) -> Path:
    recipe = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return root / checkpoint_hash / recipe


def read_receipt(directory: Path, checkpoint_hash: str, manifest_hash: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Never infer prediction provenance from a filename, step, or current hash."""
    try:
        receipt = json.loads((directory / "receipt.json").read_text())
        expected = {"schema": SCHEMA, "checkpoint_sha256": checkpoint_hash,
                    "manifest_sha256": manifest_hash, "evaluation_profile": profile}
        if any(receipt[key] != value for key, value in expected.items()):
            raise ValueError("保存済み予測とcheckpoint・dataset・評価条件の対応が一致しません")
        if receipt["predictions_sha256"] != sha256(directory / "pred_test.npz"):
            raise ValueError("保存済み予測の内容hashが一致しません")
        return dict(receipt)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("生成元を検証できる保存済み予測がありません。再推論または表示用予測の生成を実行してください") from exc


def write_bundle(directory: Path, predictions: dict[str, np.ndarray], *, checkpoint: Path,
                 checkpoint_hash: str, manifest_hash: str, profile: dict[str, Any]) -> None:
    """Publish only freshly evaluated predictions; caller hashes before loading."""
    if directory.exists():
        raise FileExistsError(f"Prediction bundle already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=directory.parent) as temporary:
        staged = Path(temporary)
        np.savez_compressed(staged / "pred_test.npz", allow_pickle=False, **predictions)
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("予測生成中にcheckpointが変更されました")
        receipt = {"schema": SCHEMA, "checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_hash,
                   "evaluation_profile": profile, "predictions_sha256": sha256(staged / "pred_test.npz"),
                   "device": "cpu", "source": "fresh_checkpoint_inference"}
        (staged / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
        staged.rename(directory)
