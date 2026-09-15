import json

from src.tasks.ball_detection.visualization.review.datasets import BallDatasetCatalog


def test_canonical_sibling_image_reference_stays_supported(tmp_path, make_web_store):
    data = tmp_path / "data"
    root = make_web_store(data)
    source = root.parent / "source"
    source.mkdir()
    original = root / "stills" / "positive.jpg"
    original.rename(source / "positive.jpg")
    strings_path = root / "index_strings.json"
    strings = json.loads(strings_path.read_text())
    strings["paths"][0] = "../source/positive.jpg"
    strings_path.write_text(json.dumps(strings))
    catalog = BallDatasetCatalog(data)
    entry = next(item for item in catalog.entries() if item.spec.id == "web_static")
    assert entry.available
    frame = catalog.resolve("web_static", "0").read_rgb(0)
    assert frame.shape == (48, 64, 3)
