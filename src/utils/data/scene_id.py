"""Helpers for resolving scene IDs from dataset samples."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset, Subset


def resolve_scene_id(dataset: Dataset, idx: int) -> int:
    """Resolve a scene ID for the given dataset/sample index.

    The function supports datasets that expose:
    - scene_id_for_sample(idx)
    - index[idx][0]
    - scenes (list of paths)
    - scene_paths (list of paths)

    Args:
        dataset: Dataset or Subset.
        idx: Sample index within the dataset.

    Returns:
        Integer scene ID.

    Raises:
        AttributeError: If the dataset does not expose scene ID information.
    """

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        base_idx = dataset.indices[idx]
        return resolve_scene_id(base_dataset, int(base_idx))

    scene_id = _resolve_scene_id_impl(dataset, idx)
    return int(scene_id)


def _resolve_scene_id_impl(dataset: Dataset, idx: int) -> int:
    """Resolve scene ID for concrete dataset types.

    Args:
        dataset: Dataset instance.
        idx: Sample index.

    Returns:
        Integer scene ID.
    """

    if hasattr(dataset, "scene_id_for_sample"):
        return int(getattr(dataset, "scene_id_for_sample")(idx))
    if hasattr(dataset, "index"):
        entry = getattr(dataset, "index")[idx]
        if isinstance(entry, (tuple, list)) and entry:
            return int(entry[0])
    if hasattr(dataset, "scene_paths"):
        return int(idx)
    if hasattr(dataset, "scenes"):
        return int(idx)
    raise AttributeError("Dataset must expose scene_id_for_sample, index, scenes, or scene_paths")


if __name__ == "__main__":
    class _DummyDataset(Dataset[Any]):
        def __init__(self) -> None:
            self.index = [(0, 0), (1, 0), (1, 1)]

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int) -> int:
            return idx

    dummy = _DummyDataset()
    assert resolve_scene_id(dummy, 0) == 0
    assert resolve_scene_id(dummy, 1) == 1
    print("common.data.scene_id smoke ok")
