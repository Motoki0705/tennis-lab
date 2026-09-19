"""One read-only ZIP CRC pass over all published RGB token archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=False)
    records = json.loads((root / "dataset.json").read_text())["clips"]
    rows = []
    for record in records:
        directory = root / record["path"] / "annotations/dino_v3"
        archives = sorted(directory.glob("*.npz"))
        if len(archives) != record["num_cameras"]:
            raise ValueError(f"Archive inventory mismatch: {directory}")
        for path in archives:
            row = {"path": str(path), "bytes": path.stat().st_size, "bad_member": None}
            try:
                with ZipFile(path) as archive:
                    row["bad_member"] = archive.testzip()
                row["passed"] = row["bad_member"] is None
            except Exception as error:
                row.update(passed=False, error=f"{type(error).__name__}: {error}")
            rows.append(row)
            if not row["passed"]:
                print(json.dumps(row), flush=True)
            elif len(rows) % 25 == 0:
                print(f"Checked {len(rows)} archives", flush=True)
    failures = [row for row in rows if not row["passed"]]
    result = {
        "dataset_root": str(root),
        "archives": len(rows),
        "failed": len(failures),
        "rows": rows,
        "interpretation": "One later read only; a pass does not erase an earlier read failure or establish environmental stability.",
    }
    (out / "crc_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"archives": len(rows), "failed": failures}), flush=True)


if __name__ == "__main__":
    main()
