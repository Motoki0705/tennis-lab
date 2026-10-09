"""Explicit JPEG decoder boundary outside the compiled RGB-input model."""

from __future__ import annotations

import time
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from torch import Tensor

JPEG_DECODERS = ("opencv", "nvjpeg")


def coordinate_batches(batches: Iterable[dict[str, Any]], device: torch.device,
                       *, prefetch: bool = False) -> Generator[dict[str, Any], None, None]:
    """Overlap next-batch JPEG decode with the current model on a CUDA stream.

    One producer preserves sampler order. Events and record_stream keep tensors
    alive across streams, including the CPU encoded buffer used by nvJPEG.
    Exceptions propagate through the future; there is no decoder fallback.
    """
    if not prefetch:
        yield from batches
        return
    if device.type != "cuda":
        raise ValueError("Image prefetch requires CUDA nvJPEG input")
    iterator = iter(batches)
    stream = torch.cuda.Stream(device=device)

    def load() -> tuple[dict[str, Any], torch.cuda.Event] | None:
        begin = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return None
        arrived = time.perf_counter()
        with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
            rgb = decode_coordinate_jpegs(batch, device)
            result = {key: value for key, value in batch.items() if key not in {"jpeg", "jpeg_lengths", "image_shape"}}
            result.update(rgb=rgb, _encoded_keepalive=batch["jpeg"])
            for key in ("timestamps", "pose", "pose_valid"):
                if key in result:
                    result[key] = result[key].to(device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
            result.update(_reader_wait_seconds=arrived - begin,
                          _image_prepare_seconds=time.perf_counter() - arrived)
            return result, event

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="coordinate-jpeg") as pool:
        future = pool.submit(load)
        try:
            while (ready := future.result()) is not None:
                batch, event = ready
                current = torch.cuda.current_stream(device)
                current.wait_event(event)
                for value in batch.values():
                    if isinstance(value, Tensor) and value.is_cuda:
                        value.record_stream(current)
                future = pool.submit(load)
                yield batch
        finally:
            # Join the producer before releasing encoded buffers on early exit.
            pool.shutdown(wait=True)
            stream.synchronize()


def decode_coordinate_jpegs(batch: dict[str, Any], device: torch.device) -> Tensor:
    if device.type != "cuda":
        raise ValueError("nvJPEG coordinate input requires CUDA; no CPU fallback")
    from torchvision.io import ImageReadMode, decode_jpeg

    packed = batch["jpeg"]
    # DataLoader's pin-memory traversal converts plain tuples into lists.
    lengths = tuple(batch["jpeg_lengths"])
    shape = tuple(batch["image_shape"])
    if ("rgb" in batch or packed.dtype != torch.uint8 or packed.device.type != "cpu"
            or packed.ndim != 1 or sum(lengths) != packed.numel()
            or len(shape) != 5 or shape[1:3] != (32, 3) or len(lengths) != shape[0] * 32):
        raise ValueError("Expected CPU packed JPEGs with declared B,32,3,H,W")
    images = decode_jpeg(list(packed.split(lengths)), mode=ImageReadMode.RGB, device=device,
                         apply_exif_orientation=False)
    if any(tuple(image.shape) != shape[2:] or image.dtype != torch.uint8 for image in images):
        raise ValueError("JPEG dimensions differ from the frozen image geometry")
    rgb: Tensor = torch.stack(images).view(shape)
    return rgb


def jpeg_decoder_contract(decoder: str) -> dict[str, Any]:
    if decoder == "opencv":
        import cv2

        return dict(decoder="opencv", version=cv2.__version__, color_order="BGR_to_RGB", device="cpu")
    if decoder != "nvjpeg":
        raise ValueError("Unknown JPEG decoder")
    import torchvision

    return dict(decoder="torchvision.io.decode_jpeg/nvjpeg", version=torchvision.__version__,
                color_order="RGB", device="cuda", apply_exif_orientation=False,
                cuda_version=torch.version.cuda, pixels_identical_to_opencv=False)
