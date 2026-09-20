"""Non-destructive snapshots, atomic manifests, and serialized writers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .contracts import FrameRecord, Manifest, VariantConfig
from .sampling import sample_indices

TRANSFER_PROMPT = (
    "Image 1 is the fixed appearance reference. Image 2 is the edit target and the sole "
    "authority for geometry and composition. Change only the tennis playing surfaces and "
    "surrounding court ground to red clay matching Image 1. Preserve Image 2’s exact "
    "viewpoint, perspective, lens distortion, framing, court boundaries, white-line "
    "positions and widths, nets, posts, fences, buildings, vegetation, sky, lighting, and "
    "shadows. Do not copy Image 1’s layout. Add or remove no objects. Return one "
    "1920×1080 image without cropping, padding, text, or watermark."
)
REFERENCE_PROMPT = (
    "Use case: style-transfer. Edit the supplied photograph to create the fixed appearance "
    "reference for a multi-view tennis scene. Change only the tennis playing surfaces and "
    "surrounding court ground to natural terracotta red clay with fine matte clay texture. "
    "Preserve the exact viewpoint, perspective, lens distortion, framing, court boundaries, "
    "white-line positions and widths, nets, posts, fences, buildings, vegetation, sky, "
    "lighting, and shadows. Add or remove no objects. Return one 1920×1080 image without "
    "cropping, padding, text, or watermark."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_manifest(root: Path) -> Manifest:
    manifest: Manifest = Manifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    return manifest


def save_manifest(root: Path, manifest: Manifest) -> None:
    write_json(root / "manifest.json", manifest.model_dump(mode="json"))


@contextmanager
def variant_lock(root: Path) -> Iterator[None]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Not a prepared variant directory: {root}")
    with (root / ".writer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def source_inventory(config: VariantConfig, names: list[str]) -> dict[str, str]:
    source = config.source_workspace
    paths = [
        source / name
        for name in (
            "run.json",
            "resolved-config.yaml",
            "frames/frames.json",
            "sfm/reconstruction.json",
        )
    ]
    model = source / "sfm/model"
    if not model.is_dir():
        raise FileNotFoundError(model)
    paths.extend(sorted(p for p in model.rglob("*") if p.is_file()))
    paths.extend(source / "frames/images" / name for name in sorted(set(names)))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source artifacts must be regular files, not symbolic links")
    return {str(path.relative_to(source)): sha256(path) for path in paths}


def prepare(config: VariantConfig) -> Manifest:
    """Publish a complete input snapshot, or verify the existing same experiment."""
    root = config.output_root
    if root.exists():
        with variant_lock(root):
            manifest = load_manifest(root)
            if manifest.config != config:
                raise ValueError("Existing variant has a different configuration")
            from .validation import validate_integrity

            validate_integrity(root, manifest, verify_source=True)
            return manifest
    source = config.source_workspace
    source_config = yaml.safe_load((source / "resolved-config.yaml").read_text())
    test_every = source_config["nht_training"]["test_every"]
    if not isinstance(test_every, int) or test_every < 2:
        raise ValueError("Source test_every must permit both training and validation")
    frame_metadata = json.loads((source / "frames/frames.json").read_text())
    accepted = sorted(
        row["filename"] for row in frame_metadata["frames"] if row["accepted"]
    )
    names = [
        f"frame_{index:06d}.jpg"
        for index in (
            config.frame_indices
            if config.frame_indices is not None
            else sample_indices(
                config.first_index, config.last_index, config.sample_count
            )
        )
    ]
    reference_name = f"frame_{config.reference_index:06d}.jpg"
    if not set([*names, reference_name]).issubset(accepted):
        raise ValueError("Requested frames must be accepted source images")
    inventory = source_inventory(config, [*names, reference_name])
    frames = [
        FrameRecord(
            name=name,
            source_index=int(name[6:12]),
            original_sorted_index=accepted.index(name),
            split="validation" if accepted.index(name) % test_every == 0 else "train",
            source_sha256=inventory[f"frames/images/{name}"],
        )
        for name in names
    ]
    if {frame.split for frame in frames} != {"train", "validation"}:
        raise ValueError(
            "Selection must include both original train and validation splits"
        )
    with Image.open(source / "frames/images" / names[0]) as im:
        size = im.size
    for name in [*names, reference_name]:
        with Image.open(source / "frames/images" / name) as im:
            if im.size != size:
                raise ValueError("All input images must have identical dimensions")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for directory in (
            "inputs/targets",
            "reference",
            "prompts",
            "generation/attempts",
            "generation/accepted",
            "provenance/source",
            "review",
        ):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        for frame in frames:
            shutil.copyfile(
                source / "frames/images" / frame.name,
                staging / "inputs/targets" / frame.name,
            )
        shutil.copyfile(
            source / "frames/images" / reference_name,
            staging / "inputs/reference-source.jpg",
        )
        for relative in (
            "run.json",
            "resolved-config.yaml",
            "frames/frames.json",
            "sfm/reconstruction.json",
        ):
            target = staging / "provenance/source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        prompts = {"reference": REFERENCE_PROMPT, "transfer": TRANSFER_PROMPT}
        if config.generation_size is not None:
            width, height = config.generation_size
            prompts = {
                key: value.replace("1920×1080", f"{width}×{height}")
                for key, value in prompts.items()
            }
        for name, prompt in prompts.items():
            (staging / "prompts" / f"{name}.txt").write_text(prompt, encoding="utf-8")
        (staging / "variant.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        manifest = Manifest(
            provider_session=os.environ.get("CODEX_THREAD_ID")
            or os.environ.get("CODEX_SESSION_ID"),
            config=config,
            config_sha256=sha256(staging / "variant.yaml"),
            source_files=inventory,
            prompt_sha256={
                name: text_sha256(prompt) for name, prompt in prompts.items()
            },
            image_size=size,
            frames=frames,
            reference_source_sha256=inventory[f"frames/images/{reference_name}"],
        )
        save_manifest(staging, manifest)
        (staging / "generation/requests.jsonl").touch()
        staging.rename(root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def revise_generation_size(
    root: Path, size: tuple[int, int], *, reason: str
) -> Manifest:
    """Record an explicit input-protocol revision before accepting any target."""
    from .validation import validate_integrity

    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest, verify_source=True)
        if manifest.pending or any(frame.accepted_attempt for frame in manifest.frames):
            raise ValueError(
                "Input protocol can change only before accepting targets, with no pending request"
            )
        if not reason.strip():
            raise ValueError("An explicit revision reason is required")
        revised_config = VariantConfig.model_validate(
            {**manifest.config.model_dump(), "generation_size": size}
        )
        archive = root / "provenance/revisions" / f"{manifest.workflow_revision:04d}"
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "variant.yaml"):
            shutil.copyfile(root / name, archive / name)
        shutil.copytree(root / "prompts", archive / "prompts")
        write_json(
            archive / "revision.json",
            {"reason": reason, "next_generation_size": list(size)},
        )
        manifest.config = revised_config
        manifest.workflow_revision += 1
        manifest.status = "generating"
        (root / "variant.yaml").write_text(
            yaml.safe_dump(revised_config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        manifest.config_sha256 = sha256(root / "variant.yaml")
        prompt = TRANSFER_PROMPT.replace("1920×1080", f"{size[0]}×{size[1]}")
        (root / "prompts/transfer.txt").write_text(prompt, encoding="utf-8")
        manifest.prompt_sha256["transfer"] = text_sha256(prompt)
        save_manifest(root, manifest)
        return manifest


def configure_api_key_file(root: Path, key_file: Path) -> None:
    """Change credential location without changing generation parameters or prompts."""
    from .validation import validate_integrity

    if not key_file.is_absolute():
        raise ValueError("API key file must be absolute")
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_integrity(root, manifest)
        if manifest.config.api is None:
            raise ValueError("Variant has no API configuration")
        old = manifest.config.api.api_key_file
        manifest.config.api.api_key_file = key_file
        if manifest.provider_session is None:
            manifest.provider_session = os.environ.get(
                "CODEX_THREAD_ID"
            ) or os.environ.get("CODEX_SESSION_ID")
        config_path = root / "variant.yaml"
        config_path.write_text(
            yaml.safe_dump(manifest.config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        manifest.config_sha256 = sha256(config_path)
        history = root / "provenance/credential-locations.jsonl"
        with history.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "previous_file": str(old),
                        "current_file": str(key_file),
                        "secret_persisted": False,
                    }
                )
                + "\n"
            )
        save_manifest(root, manifest)
