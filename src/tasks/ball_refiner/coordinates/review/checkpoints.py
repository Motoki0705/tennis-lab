"""Discover local coordinate checkpoints without constructing GPU models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.tasks.ball_refiner.coordinates.config import ModelConfig, parse_section
from src.tasks.ball_refiner.coordinates.inference import CHECKPOINT_SCHEMA
from src.tasks.ball_refiner.coordinates.review.artifacts import (
    bundle_path,
    cache_root,
    read_receipt,
)
from src.tasks.ball_refiner.coordinates.review.contracts import evaluation_profile


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    info: dict[str, Any]
    run: Path | None
    predictions: Path | None = None


def describe(path: Path, identifier: str, manifest_hash: str, fps: float, prediction_root: Path) -> Checkpoint:
    info: dict[str, Any] = {"id": identifier, "label": identifier, "filename": path.name,
                            "compatible": False, "reason": None, "recommended": False}
    run: Path | None = None
    predictions = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("座標Refinerのcheckpoint形式ではありません")
        model = parse_section(ModelConfig, payload["model_config"])
        if payload["manifest_sha256"] != manifest_hash:
            raise ValueError("学習時の共通データとmanifestが一致しません")
        if not math.isclose(float(payload["fps"]), fps, abs_tol=1e-6, rel_tol=0):
            raise ValueError("学習時とデータのFPSが一致しません")
        score = float(payload["validation_rmse"])
        if not math.isfinite(score) or score < 0:
            raise ValueError("validation RMSEが不正です")
        method = "flow" if model.architecture == "flow" else "gan" if payload.get("discriminator") is not None else "regression"
        info.update(dimensions=model.dimensions, method=method, model_config=payload["model_config"],
                    step=int(payload["step"]), validation_rmse=score, unit="px" if model.dimensions == 2 else "m",
                    train_event_probability=None, evaluation_profile=None, saved_available=False,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), manifest_sha256=manifest_hash)
        # Native training layout. A copied, standalone checkpoint still supports
        # real inference, but has no invented training labels or saved predictions.
        if path.parent.name == "checkpoints" and path.parent.parent.name == "version_0" and path.parent.parent.parent.name == "logs":
            candidate = path.parents[3]
            config_path = candidate / "config.yaml"
            if config_path.is_file():
                config = yaml.safe_load(config_path.read_text())
                if config["model"] != payload["model_config"]:
                    raise ValueError("隣接するconfigとcheckpointのモデル設定が一致しません")
                if bool(config["training"]["gan_weight"]) != (method == "gan"):
                    raise ValueError("隣接するconfigとcheckpointのGAN設定が一致しません")
                run = candidate
                info.update(train_event_probability=float(config["corruption"]["event_probability"]),
                            evaluation_profile=evaluation_profile(config), run_name=candidate.parent.name)
                if path.name == "best.ckpt":
                    predictions = bundle_path(prediction_root, info["sha256"], info["evaluation_profile"])
        rate = info["train_event_probability"]
        suffix = "学習条件不明" if rate is None else f"イベント {rate:.0%}"
        info.update(compatible=True, label=f"{model.dimensions}D · {method.upper()} · {suffix} · {path.name} · val {score:.3f} {info['unit']}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, AssertionError) as exc:
        info["reason"] = str(exc)
    # weights_only may reject legacy custom pickle classes. Keep that visible.
    except Exception as exc:
        info["reason"] = f"checkpoint読込失敗: {type(exc).__name__}: {exc}"
    return Checkpoint(path, info, run, predictions)


class CheckpointCatalog:
    def __init__(self, outputs: Path, curated: Path, manifest_hash: str, fps: float) -> None:
        self.roots = {"outputs": outputs.resolve(), "ckpt": curated.resolve()}
        self.prediction_root = cache_root(outputs.resolve())
        self.manifest_hash, self.fps = manifest_hash, fps
        self.entries: dict[str, Checkpoint] = {}
        self._cache: dict[str, tuple[tuple[int, int, int, int], Checkpoint]] = {}

    def refresh(self) -> list[dict[str, Any]]:
        entries: dict[str, Checkpoint] = {}
        for prefix, root in self.roots.items():
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.ckpt")):
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    continue
                identifier = f"{prefix}:{path.relative_to(root).as_posix()}"
                stat = path.stat()
                sidecar = path.parents[3] / "config.yaml" if len(path.parents) > 3 else path
                side = sidecar.stat() if sidecar.is_file() else None
                revision = (stat.st_size, stat.st_mtime_ns, side.st_size if side else 0, side.st_mtime_ns if side else 0)
                old = self._cache.get(identifier)
                entry = old[1] if old and old[0] == revision else describe(resolved, identifier, self.manifest_hash, self.fps, self.prediction_root)
                self._cache[identifier] = (revision, entry)
                entry.info["recommended"] = False
                if entry.info["compatible"]:
                    entry.info.update(saved_available=False, saved_unavailable_reason="生成元を検証できる保存済み予測がありません")
                    if entry.predictions is not None:
                        try:
                            read_receipt(entry.predictions, entry.info["sha256"], self.manifest_hash, entry.info["evaluation_profile"])
                            entry.info.update(saved_available=True, saved_unavailable_reason=None)
                        except ValueError as exc:
                            entry.info["saved_unavailable_reason"] = str(exc)
                entries[identifier] = entry
        self.entries = entries
        # Compare validation scores only within an identical evaluation recipe.
        for dimensions in (2, 3):
            candidates = [item for item in entries.values() if item.info["compatible"] and item.info["dimensions"] == dimensions
                          and item.info["filename"] == "best.ckpt" and item.info["evaluation_profile"] is not None]
            groups: dict[str, list[Checkpoint]] = {}
            for item in candidates:
                groups.setdefault(json.dumps(item.info["evaluation_profile"], sort_keys=True), []).append(item)
            for group in groups.values():
                min(group, key=lambda item: (item.info["validation_rmse"], item.info["id"])).info["recommended"] = True
        return [item.info.copy() for item in entries.values()]

    def get(self, identifier: str, dimensions: int, expected_hash: str | None = None) -> Checkpoint:
        if identifier not in self.entries:
            raise ValueError("カタログにないcheckpointです。一覧を更新してください")
        entry = self.entries[identifier]
        if not entry.info["compatible"] or entry.info["dimensions"] != dimensions:
            raise ValueError("この次元・データに対応しないcheckpointです")
        actual = hashlib.sha256(entry.path.read_bytes()).hexdigest()
        if actual != entry.info["sha256"] or (expected_hash is not None and actual != expected_hash):
            raise RuntimeError("checkpointが変更されています。一覧を更新してください")
        return entry
