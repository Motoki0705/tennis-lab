"""Configuration users must not require optional visualization libraries."""
from __future__ import annotations

import subprocess
import sys


def test_configuration_import_does_not_load_opencv() -> None:
    subprocess.run([sys.executable, "-c", """
import sys
class RejectOpenCV:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cv2':
            raise ImportError('OpenCV deliberately unavailable')
sys.meta_path.insert(0, RejectOpenCV())
from src.utils.configuration import NonHydraPathBoundary
assert 'src.utils.rendering' not in sys.modules
"""], check=True)


def test_rendering_remains_available_on_explicit_access() -> None:
    subprocess.run([sys.executable, "-c", """
import src.utils
assert src.utils.rendering is src.utils.rendering
assert src.utils.rendering.__name__ == 'src.utils.rendering'
try:
    src.utils.nonexistent
except AttributeError:
    pass
else:
    raise AssertionError('unknown attributes must raise')
"""], check=True)
