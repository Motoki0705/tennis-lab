"""Bind all user-supplied photos to immutable, byte-identical paper inputs."""

from __future__ import annotations

import shutil

from common import REPO, ROOT, sha256, write_json
from PIL import Image, ImageOps


def main() -> None:
    directory = REPO / "data/samples/tennis_court"
    paths = sorted(p for p in directory.iterdir() if p.is_file())
    if not paths:
        raise ValueError(f"No input photographs: {directory}")
    records = []
    for index, path in enumerate(paths, 1):
        ident = f"local{index:02d}"
        with Image.open(path) as image:
            size = ImageOps.exif_transpose(image).size
        dest = ROOT / "images" / (ident + path.suffix.lower())
        shutil.copyfile(path, dest)
        records.append(
            {
                "id": ident,
                "filename": path.name,
                "supplied_path": str(path.relative_to(REPO)),
                "paper_path": str(dest.relative_to(ROOT)),
                "sha256": sha256(dest),
                "width": size[0],
                "height": size[1],
            }
        )
    write_json(
        ROOT / "evidence/inputs.json",
        {
            "schema": "court_paper_inputs_v1",
            "selection": "All files supplied in data/samples/tennis_court, filename order; no prediction-based selection.",
            "source": "User-supplied images; author, capture date and original publication URL were not supplied.",
            "preprocessing": "Byte-identical copy; EXIF orientation is applied identically at inference and display time.",
            "images": records,
        },
    )
    print(f"Bound {len(records)} supplied photographs.")


if __name__ == "__main__":
    main()
