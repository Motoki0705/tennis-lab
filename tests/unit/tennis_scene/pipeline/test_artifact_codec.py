import sys

import numpy as np
import pytest

from src.tennis_scene.pipeline.storage.codec import restore_type

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


def test_pep695_array_alias_accepts_numeric_arrays_and_rejects_json_lists():
    alias = TypeAliasType("alias", np.ndarray[tuple[int, ...], np.dtype[np.float64]])
    array = np.array([[3., 7.], [11., 19.]])
    np.testing.assert_array_equal(restore_type(array, alias), [[3., 7.], [11., 19.]])
    with pytest.raises(TypeError, match="NumPy"):
        restore_type([[3., 7.], [11., 19.]], alias)
