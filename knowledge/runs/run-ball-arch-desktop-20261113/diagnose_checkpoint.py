"""Isolate numerical and compiler axes before accepting a memory policy."""

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from guard import configure, host_guard
from rebuild_dataset import write
from test_state import multiscale_config


def main(root):
    import torch

    from src.tasks.ball_detection.models.mdd.multiscale_query import (
        DeepMDDMultiScaleQueryDetector,
    )
    from src.tasks.ball_detection.training.runtime.compilation import (
        compile_coordinate_model,
        coordinate_compile_scope,
    )

    configure(root)
    torch.set_num_threads(2)
    results = []
    with host_guard(root / "checkpoint-diagnostic-host.json"):
        for dropout in (0.0, 0.1):
            for changed_chunk in (False, True):
                for scope in ("msda", "encoder_msda"):
                    torch.manual_seed(1859)
                    config = multiscale_config(
                        frames=3,
                        dropout=dropout,
                        msda_levels=(8, 16, 32, 64),
                        msda_chunk_size=6,
                    )
                    plain = DeepMDDMultiScaleQueryDetector(config).cuda()
                    replay = DeepMDDMultiScaleQueryDetector(
                        replace(
                            config,
                            activation_checkpointing=True,
                            checkpoint_scope=scope,
                            msda_chunk_size=1 if changed_chunk else 6,
                        )
                    ).cuda()
                    replay.load_state_dict(plain.state_dict(), strict=True)
                    rgb = torch.randint(
                        256, (2, 3, 3, 257, 385), dtype=torch.uint8, device="cuda"
                    )
                    times = torch.tensor(
                        [[0.0, 0.03, 0.07], [1.0, 1.04, 1.09]], device="cuda"
                    )
                    probe = torch.tensor(
                        [[[0.2, 0.9], [1.3, -0.7], [0.3, 0.6]]], device="cuda"
                    )
                    rng = torch.cuda.get_rng_state()
                    a = plain(rgb, times)
                    (a * probe).sum().backward()
                    torch.cuda.set_rng_state(rng)
                    b = replay(rgb, times)
                    (b * probe).sum().backward()
                    differences = []
                    for (name, p), (_, q) in zip(
                        plain.named_parameters(), replay.named_parameters(), strict=True
                    ):
                        if p.grad is None or q.grad is None:
                            differences.append(dict(name=name, missing=True))
                            continue
                        if not torch.allclose(p.grad, q.grad, rtol=5e-3, atol=2e-5):
                            differences.append(
                                dict(
                                    name=name,
                                    max_abs=float((p.grad - q.grad).abs().max()),
                                    norm=float(p.grad.norm()),
                                    relative_l2=float(
                                        (p.grad - q.grad).norm()
                                        / (p.grad.norm() + 1e-20)
                                    ),
                                )
                            )
                    record = dict(
                        axis="gradient",
                        scope=scope,
                        dropout=dropout,
                        changed_chunk=changed_chunk,
                        max_output_difference=float((a - b).abs().max()),
                        differences=differences,
                    )
                    results.append(record)
                    write(root / "checkpoint-diagnostic.json", results)
                    print(json.dumps(record), flush=True)
                    del plain, replay, rgb, a, b
                    torch.cuda.empty_cache()
        for scope in ("msda", "encoder_msda"):
            torch._dynamo.reset()
            config = multiscale_config(
                frames=3,
                dropout=0.0,
                activation_checkpointing=True,
                checkpoint_scope=scope,
                msda_levels=(8, 16, 32, 64),
                msda_chunk_size=1,
            )
            model = DeepMDDMultiScaleQueryDetector(config).cuda()
            rgb = torch.randint(
                256, (1, 3, 3, 65, 97), dtype=torch.uint8, device="cuda"
            )
            times = torch.tensor([[0.0, 0.03, 0.07]], device="cuda")
            record = dict(axis="compile", scope=scope)
            try:
                compile_coordinate_model(model, mode="default", fullgraph=False)
                with coordinate_compile_scope(model):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(rgb, times)
                    output.square().mean().backward()
                record["status"] = "passed"
            except BaseException as error:
                record.update(
                    status="failed", error=str(error), traceback=traceback.format_exc()
                )
            results.append(record)
            write(root / "checkpoint-diagnostic.json", results)
            print(
                json.dumps({k: v for k, v in record.items() if k != "traceback"}),
                flush=True,
            )
            del model, rgb
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root.resolve())
