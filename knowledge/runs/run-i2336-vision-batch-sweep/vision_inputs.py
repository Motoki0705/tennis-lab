"""Use the same fixed inputs, model identities and decoders as #2303 and #2304."""

import json
import subprocess

import numpy as np
from batching import batch_profile_points, batch_slices
from benchmark_dino import (
    CHECKPOINT_SHA256 as DINO_CHECKPOINT_SHA256,
)
from benchmark_dino import EXTENSION_SHA256, check_extension, compare, verify_inputs
from fixed_inputs import CHECKPOINT as DINO_CHECKPOINT
from fixed_inputs import LOCK as DINO_LOCK
from fixed_inputs import REPOSITORY, load_sample, rows_for
from inputs import sha256_file
from measure_dino import INPUT_NAME, OUTPUT_SHAPES
from vit_models import HMR2, MODEL_ASSETS, VITPOSE, VitAdapter

DINO = "dino_4scale_swin_l"
BUCKET = "750x1333"
MODELS = (VITPOSE, HMR2, DINO)
BATCHES = {VITPOSE: (1, 4, 8, 16, 32), HMR2: (1, 4, 8, 16, 32), DINO: (1, 2, 4)}
OPT_BATCH = {VITPOSE: 16, HMR2: 16, DINO: 4}
CONDITIONS = {
    VITPOSE: ("0", "1", "2"),
    HMR2: ("0", "1", "2", "3a", "3b"),
    DINO: ("0", "1", "2", "2-plugin"),
}
PTQ_RECIPES = {"3a": "attention-out-max", "3b": "linear-max"}
DINO_REVISION = "d84a491d41898b3befd8294d1cf2614661fc0953"


def read(path):
    return json.loads(path.read_text())


class PinnedInputs:
    def __init__(self, model, directory, *, verify=False):
        self.model, self.directory = model, directory
        if model == DINO:
            self.adapter = None
            self.checkpoint = DINO_CHECKPOINT
            if verify:
                lock = verify_inputs(directory, BUCKET, self.checkpoint)
                check_extension()
                revision = subprocess.check_output(
                    ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
                ).strip()
                if revision != DINO_REVISION or subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(REPOSITORY),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    text=True,
                ):
                    raise ValueError(
                        "DINO checkout differs from the pinned #2304 source"
                    )
            else:
                lock = read(DINO_LOCK)
            self.rows = rows_for(lock, BUCKET, "eval")
            self.addresses = [row["file"] for row in self.rows]
            representative = lock["buckets"][BUCKET]["representative"]
            self.input_shape = tuple(self.rows[0]["shape"][1:])
            self.identity = {
                "model": model,
                "bucket": BUCKET,
                "checkpoint_sha256": DINO_CHECKPOINT_SHA256,
                "tensor_set_sha256": lock["tensor_set_sha256"],
                "extension_sha256": EXTENSION_SHA256,
                "dino_revision": DINO_REVISION,
            }
        else:
            self.adapter = VitAdapter(model)
            self.checkpoint, digest = MODEL_ASSETS[model]
            lock, self.addresses = self.adapter.verify_inputs(directory)
            representative = tuple(lock["profile_address"])
            self.input_shape = self.adapter.batch_shape[1:]
            self.identity = {
                "model": model,
                "checkpoint_sha256": digest,
                "tensors": lock["tensors"],
                **self.adapter.identity(),
            }
            if verify and sha256_file(self.checkpoint) != digest:
                raise ValueError(
                    f"{model} checkpoint differs from the pinned #2303 weights"
                )
        self.profile_index = self.addresses.index(representative)
        self.identity["profile_anchor"] = (
            representative if model == DINO else list(representative)
        )

    def batches(self, batch_size):
        slices = batch_slices(len(self.addresses), batch_size)
        if self.model == DINO:
            return [
                np.concatenate(
                    [load_sample(self.directory, row) for row in self.rows[span]]
                )
                for span in slices
            ]
        data = np.load(self.directory / "eval.npy", allow_pickle=False)
        return [data[span] for span in slices]

    def compare(self, actual, baseline):
        if self.model == DINO:
            return compare(baseline, actual, self.rows)
        return self.adapter.accuracy(actual, baseline, self.addresses)

    @property
    def output_shapes(self):
        if self.model == DINO:
            return {name: shape[1:] for name, shape in OUTPUT_SHAPES.items()}
        return {"features": self.adapter.output_shape}

    @property
    def multiplier(self):
        return 2 if self.model == VITPOSE else 1

    def profile(self, optimum, maximum):
        return {
            INPUT_NAME: [
                [batch * self.multiplier, *self.input_shape]
                for batch in batch_profile_points(optimum, maximum)
            ]
        }

    def representative(self, batch_size):
        span = batch_slices(len(self.addresses), batch_size)[
            self.profile_index // batch_size
        ]
        if span.stop - span.start != batch_size:
            raise ValueError(
                "The pinned profile anchor is outside the complete batches"
            )
        return [
            list(row) if isinstance(row, tuple) else row for row in self.addresses[span]
        ]

    @property
    def prediction_name(self):
        return "detections.json" if self.model == DINO else "features.npy"

    def predictions(self, directory):
        path = directory / self.prediction_name
        return read(path) if self.model == DINO else np.load(path, allow_pickle=False)


def dino_method(condition):
    return {"2": "grid", "2-plugin": "plugin"}[condition]


def load_conversion(data, condition, directory):
    conversion = read(directory / "conversion.json")
    dino = data.model == DINO
    plan = directory / ("dino.plan" if dino else f"{data.model}.plan")
    digest_key = "plan_sha256" if dino else "engine_sha256"
    matches = (
        conversion["tensor_set_sha256"] == data.identity["tensor_set_sha256"]
        and conversion["method"] == dino_method(condition)
        and conversion["dino_revision"] == data.identity["dino_revision"]
        and conversion["extension_sha256"] == data.identity["extension_sha256"]
        if dino
        else conversion["identity"] == data.identity
        and conversion["precision_manifest_sha256"]
        == sha256_file(directory / "precision.json")
    )
    profile = conversion["input_profile"][INPUT_NAME]
    optimum, maximum = (shape[0] // data.multiplier for shape in profile[1:])
    if (
        not matches
        or conversion["condition"] != condition.split("-")[0]
        or conversion["checkpoint_sha256"] != data.identity["checkpoint_sha256"]
        or conversion[digest_key] != sha256_file(plan)
        or conversion["input_profile"] != data.profile(optimum, maximum)
        or set(conversion["compute_precision"]) != {"min", "opt", "max"}
    ):
        raise ValueError(f"Conversion differs from the requested engine: {directory}")
    return plan, conversion, conversion[digest_key]
