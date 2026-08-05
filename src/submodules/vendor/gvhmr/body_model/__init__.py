"""Vendored SMPL-X body models (trimmed from hmr4d.utils.smplx_utils).

Only the model types used by GVHMR inference are provided. All variants
require the licensed ``SMPLX_NEUTRAL.npz`` (see :mod:`smplx_lite`); the small
regressor assets are bundled under ``data/``.
"""

from src.submodules.configuration import BundledModelAssetPaths, require_absolute_path

from .body_model_smplx import BodyModelSMPLX
from .smplx_lite import (
    SmplxLite,
    SmplxLiteCoco17,
    SmplxLiteSmplN24,
    SmplxLiteV437Coco17,
    resolve_smplx_model_file,
)


def make_smplx(type, *, model_path, bundled_assets, **kwargs):
    """Build a body model used by GVHMR (trimmed to inference variants).

    - ``supermotion``: full SMPL-X (``BodyModelSMPLX``), predicts vertices.
    - ``supermotion_v437coco17``: 437 verts + COCO17 joints (EnDecoder FK).
    - ``supermotion_coco17`` / ``supermotion_smpl24``: joints-only variants.
    """
    model_path = require_absolute_path(model_path, name="SMPL-X body-model directory")
    if not isinstance(bundled_assets, BundledModelAssetPaths):
        raise TypeError("bundled_assets must be BundledModelAssetPaths.")
    bundled_assets.require_files()
    if type == "supermotion":
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        # Fail early with an actionable message if the licensed file is absent.
        resolve_smplx_model_file(model_path / "smplx", bm_kwargs["gender"])
        return BodyModelSMPLX(model_path=str(model_path), **bm_kwargs)
    if type == "supermotion_v437coco17":
        return SmplxLiteV437Coco17(
            model_path=model_path / "smplx",
            bundled_assets=bundled_assets,
            **kwargs,
        )
    if type == "supermotion_coco17":
        return SmplxLiteCoco17(
            model_path=model_path / "smplx",
            bundled_assets=bundled_assets,
            **kwargs,
        )
    if type == "supermotion_smpl24":
        return SmplxLiteSmplN24(
            model_path=model_path / "smplx",
            bundled_assets=bundled_assets,
            **kwargs,
        )
    raise NotImplementedError(f"Unknown body model type: {type}")


__all__ = [
    "BodyModelSMPLX",
    "SmplxLite",
    "SmplxLiteCoco17",
    "SmplxLiteSmplN24",
    "SmplxLiteV437Coco17",
    "load_smpl_faces",
    "make_smplx",
    "resolve_smplx_model_file",
]


def load_smpl_faces(path):
    """Load SMPL triangle faces (13776, 3) from an .npz/.pkl body-model file.

    SMPL and SMPL-H share the same mesh topology, so e.g.
    ``data/smplh/neutral/model.npz`` works as a faces source.
    """
    import pickle

    import numpy as np

    path = require_absolute_path(path, name="SMPL faces asset")
    if not path.exists():
        raise FileNotFoundError(
            f"SMPL body-model file not found: {path} (needed for mesh faces)"
        )
    if path.suffix == ".npz":
        data = np.load(path)
        return data["f"].astype("int64")
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        return np.asarray(data["f"], dtype="int64")
    raise ValueError(f"Unsupported body-model file type: {path.suffix}")
