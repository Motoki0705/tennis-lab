"""Real JPEG ordering, CUDA stream lifetimes and prefetch error propagation."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torchvision.io as image_io
from torchvision.io import ImageReadMode, decode_jpeg

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.training.coordinate_images import coordinate_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-producer", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output")
    torch.set_num_threads(2)
    dataset = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder="nvjpeg")
    selected: dict[tuple[str, int], int] = {}
    for i, (record, window) in enumerate(dataset.windows):
        selected.setdefault((dataset.records[record]["clip"]["source"], window.frame_step), i)
    indices = [selected[key] for key in sorted(selected)]
    batches = [collate_coordinate_windows([dataset[i] for i in indices[start:start+2]])
               for start in range(0, len(indices), 2)]
    device = torch.device("cuda")
    references = [torch.stack(decode_jpeg(list(b["jpeg"].split(b["jpeg_lengths"])),
                    mode=ImageReadMode.RGB, device=device)).view(b["image_shape"]) for b in batches]
    if args.delay_producer:
        original_decode = image_io.decode_jpeg

        def delayed_decode(*positional: Any, **keywords: Any) -> Any:
            result = original_decode(*positional, **keywords)
            # Keep caller-stream stack outstanding while the following decode
            # could otherwise reuse its intermediate output storage.
            torch.cuda._sleep(10_000_000)
            return result

        image_io.decode_jpeg = delayed_decode
    for original, expected, decoded in zip(batches, references,
                                          coordinate_batches(batches, device, prefetch=True), strict=True):
        torch.testing.assert_close(decoded["rgb"], expected, rtol=0, atol=0)
        torch.testing.assert_close(decoded["timestamps"].cpu(), original["timestamps"], rtol=0, atol=0)
        for key in ("uv", "position_valid", "frame_indices"):
            torch.testing.assert_close(decoded[key], original[key], rtol=0, atol=0)
        assert original["clip_id"] == decoded["clip_id"]

    def broken() -> Iterator[dict[str, Any]]:
        yield batches[0]
        raise RuntimeError("reader_test_exception")

    try:
        list(coordinate_batches(broken(), device, prefetch=True))
    except RuntimeError as error:
        assert str(error) == "reader_test_exception"
    else:
        raise AssertionError("Prefetch swallowed a reader error")
    iterator = coordinate_batches(batches, device, prefetch=True)
    next(iterator)
    iterator.close()
    assert not any(t.name.startswith("coordinate-jpeg") for t in threading.enumerate())
    result = dict(status="ok", windows=len(indices), batch_sizes=[len(b["clip_id"]) for b in batches],
                  sources=sorted({source for source, _ in selected}), frame_steps=[1, 2, 4],
                  rgb_bit_identical=True, teachers_and_order_unchanged=True,
                  reader_error_propagated=True, early_exit_joined=True,
                  delayed_producer=args.delay_producer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
