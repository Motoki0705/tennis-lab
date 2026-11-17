"""Check frozen numerical sources, allowing only the recorded mypy comment."""
import ast
import hashlib
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_sources(frozen):
    for name,digest in frozen['numeric_sources'].items():
        path=Path(name)
        if sha(path)==digest:
            continue
        assert name=='src/tasks/court_detection/geometry/video_drift/motion.py', name
        evaluated=Path(__file__).parent/'frozen-motion.py.txt'
        assert sha(evaluated)==digest
        original=evaluated.read_text()
        final=path.read_text()
        assert final==original.replace('return np.asarray(centre + np.quantile(along, [.1, .9])[:, None] * vh[0], dtype=np.float64)',
            'return np.asarray(centre + np.quantile(along, [.1, .9])[:, None] * vh[0], dtype=np.float64)  # type: ignore[index]')
        assert ast.dump(ast.parse(original))==ast.dump(ast.parse(final))
