"""Paired model comparison with byte-identical inputs and a fixed prompt."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts import GenerationRequest, VariantConfig
from .openai_api import execute_request, load_api_key, parameters
from .workspace import TRANSFER_PROMPT, sha256, text_sha256, variant_lock, write_json

MODELS = {
    "sunburst": "gpt-image-2.5-sunburst-2026-09-08",
    "flare": "gpt-image-2.5-flare-2026-09-08",
}


def compare_models(
    config: VariantConfig,
    root: Path,
    reference: Path,
    target_index: int,
    *,
    retry_failed_request: bool = False,
) -> dict[str, Any]:
    if config.api is None or config.generation_size is None:
        raise ValueError("Model comparison requires API configuration")
    key = load_api_key(config.api.api_key_file)
    target = config.source_workspace / "frames/images" / f"frame_{target_index:06d}.jpg"
    size = config.generation_size
    prompt = TRANSFER_PROMPT.replace("1920×1080", f"{size[0]}×{size[1]}")
    expected = {
        "models": MODELS,
        "quality": config.api.quality,
        "size": list(size),
        "reference_source": str(reference),
        "reference_source_sha256": sha256(reference),
        "target_source": str(target),
        "target_source_sha256": sha256(target),
        "prompt_sha256": text_sha256(prompt),
    }
    root.mkdir(parents=True, exist_ok=True)
    with variant_lock(root):
        manifest_path = root / "comparison.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != expected:
                raise ValueError("Existing comparison has different settings or inputs")
        else:
            inputs = root / "inputs"
            inputs.mkdir(exist_ok=True)
            shutil.copyfile(reference, inputs / "reference-original.png")
            shutil.copyfile(target, inputs / "target-original.jpg")
            for source, name in ((reference, "reference.png"), (target, "target.png")):
                with Image.open(source) as image:
                    image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(
                        inputs / name
                    )
            (root / "prompt.txt").write_text(prompt, encoding="utf-8")
            write_json(manifest_path, expected)
        input_paths = [root / "inputs/reference.png", root / "inputs/target.png"]
        input_hashes = [sha256(path) for path in input_paths]
        requests = {}
        for label, model in MODELS.items():
            api = config.api.model_copy(update={"model": model})
            model_config = config.model_copy(update={"api": api})
            request = GenerationRequest(
                request_id=f"comparison-{label}-01",
                target=target.name,
                attempt=1,
                prompt=prompt,
                prompt_sha256=text_sha256(prompt),
                referenced_image_paths=[str(path) for path in input_paths],
                input_sha256=input_hashes,
                provider="openai_api",
                api_parameters=parameters(model_config),
                input_geometry={
                    "generation_size": list(size),
                    "preprocessing": "full-image LANCZOS resize before API",
                },
            )
            directory = root / label
            directory.mkdir(exist_ok=True)
            request_path = directory / "request.json"
            if (
                request_path.exists()
                and GenerationRequest.model_validate_json(request_path.read_text())
                != request
            ):
                raise ValueError("Comparison request changed since previous execution")
            write_json(request_path, request.model_dump(mode="json"))
            requests[label] = (directory, request, api)
        results = {}
        failures = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    execute_request,
                    directory,
                    request,
                    api,
                    key,
                    retry_failed_request=retry_failed_request,
                ): label
                for label, (directory, request, api) in requests.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                    print(
                        json.dumps(
                            {
                                "model": label,
                                "status": "generated",
                                "seconds": results[label]["metadata"][
                                    "elapsed_seconds"
                                ],
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    failures[label] = str(error)
                    print(
                        json.dumps(
                            {"model": label, "status": "failed", "error": str(error)}
                        ),
                        flush=True,
                    )
        write_json(root / "results.json", {"results": results, "failures": failures})
        if failures:
            raise RuntimeError(
                f"Model comparison has failed requests; see {root / 'results.json'}"
            )
        _contact_sheet(root, size)
        return {
            "root": str(root),
            "comparison_image": str(root / "comparison.png"),
            "results": results,
        }


def _contact_sheet(root: Path, size: tuple[int, int]) -> None:
    width, height = size
    header = 44
    canvas = Image.new("RGB", (2 * width, 2 * (height + header)), "#20242a")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=26)
    items = [
        ("Original target", root / "inputs/target.png"),
        ("Fixed clay reference", root / "inputs/reference.png"),
        ("Sunburst / high", root / "sunburst/api-result.png"),
        ("Flare / high", root / "flare/api-result.png"),
    ]
    for index, (title, path) in enumerate(items):
        x, y = (index % 2) * width, (index // 2) * (height + header)
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Comparison output dimensions differ: {path}")
            canvas.paste(image.convert("RGB"), (x, y + header))
        draw.text((x + 14, y + 8), title, fill="white", font=font)
    canvas.save(root / "comparison.png")
