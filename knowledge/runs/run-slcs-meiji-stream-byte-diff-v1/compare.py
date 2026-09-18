"""One finite CPU comparison of saved streams, never a live checkpoint read.

Hash both providers and compare bytes on the same scan. ZIP inspection reads only
central/local headers; it never unpickles, decompresses, or loads a model.
Observed bit differences identify bytes, not the hardware/software root cause.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

from src.utils.checksum import dual_sha256

MAIN = Path("/home/kamimura/projects/tennis-lab")
CHUNK = 1024 * 1024
PAGE = 4096
CAP = 4096


def stat(path: Path) -> dict[str, int]:
    return stat_fields(path.stat())


def stat_fields(value: os.stat_result) -> dict[str, int]:
    return {
        k: int(getattr(value, f"st_{k}"))
        for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def scan(
    left: Path, right: Path, expected: tuple[str, str], *, cap: int = CAP
) -> dict[str, Any]:
    paths = (left, right)
    primary = [hashlib.sha256(), hashlib.sha256()]
    secondary = [
        importlib.import_module("_sha256").sha256(),
        importlib.import_module("_sha256").sha256(),
    ]
    report: dict[str, Any] = {
        "sources": [{"path": str(p), "stat_before": stat(p)} for p in paths],
        "differing_bytes": 0,
        "length_only_bytes": 0,
        "xor_distribution": {},
        "bit_flip_counts_lsb0": [0] * 8,
        "flipped_bits_total": 0,
        "first_difference": None,
        "last_difference": None,
        "details": [],
        "detail_cap": cap,
        "difference_ranges_inclusive": [],
        "different_chunks": [],
    }
    lengths = [0, 0]
    offset = 0
    with left.open("rb") as a, right.open("rb") as b:
        readers = (a, b)
        for index, reader in enumerate(readers):
            report["sources"][index]["descriptor_before"] = stat_fields(
                os.fstat(reader.fileno())
            )
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if not x and not y:
                break
            for i, chunk in enumerate((x, y)):
                primary[i].update(chunk)
                secondary[i].update(chunk)
                lengths[i] += len(chunk)
            if x != y:
                report["different_chunks"].append(offset // CHUNK)
                for i in range(max(len(x), len(y))):
                    u, v = (
                        (x[i] if i < len(x) else None),
                        (y[i] if i < len(y) else None),
                    )
                    if u == v:
                        continue
                    position = offset + i
                    report["differing_bytes"] += 1
                    if report["first_difference"] is None:
                        report["first_difference"] = position
                    report["last_difference"] = position
                    ranges = report["difference_ranges_inclusive"]
                    if ranges and ranges[-1][1] + 1 == position:
                        ranges[-1][1] = position
                    else:
                        ranges.append([position, position])
                    xor = None if u is None or v is None else u ^ v
                    if xor is None:
                        report["length_only_bytes"] += 1
                    else:
                        key = f"0x{xor:02x}"
                        report["xor_distribution"][key] = (
                            report["xor_distribution"].get(key, 0) + 1
                        )
                        report["flipped_bits_total"] += xor.bit_count()
                        for bit in range(8):
                            report["bit_flip_counts_lsb0"][bit] += (xor >> bit) & 1
                    if len(report["details"]) < cap:
                        report["details"].append(
                            {
                                "offset_zero_based": position,
                                "left_byte": u,
                                "right_byte": v,
                                "xor": xor,
                                "chunk_index": position // CHUNK,
                                "chunk_offset": position % CHUNK,
                                "page_size": PAGE,
                                "page_index": position // PAGE,
                                "page_offset": position % PAGE,
                            }
                        )
            offset += max(len(x), len(y))
        for index, reader in enumerate(readers):
            report["sources"][index]["descriptor_after"] = stat_fields(
                os.fstat(reader.fileno())
            )
    for i, path in enumerate(paths):
        source = report["sources"][i]
        source.update(
            stat_after=stat(path),
            hashlib_sha256=primary[i].hexdigest(),
            cpython_sha256=secondary[i].hexdigest(),
            bytes_read=lengths[i],
            expected_sha256=expected[i],
        )
        source["gates"] = {
            "stat_unchanged": source["stat_before"]
            == source["descriptor_before"]
            == source["descriptor_after"]
            == source["stat_after"],
            "length_matches_stat": lengths[i] == source["stat_before"]["size"],
            "providers_and_expected_match": source["hashlib_sha256"]
            == source["cpython_sha256"]
            == expected[i],
        }
    report["details_truncated"] = report["differing_bytes"] > len(report["details"])
    report["all_offsets_retained_as_inclusive_ranges"] = True
    report["gates_passed"] = all(
        all(source["gates"].values()) for source in report["sources"]
    )
    report["byte_equal"] = report["differing_bytes"] == 0
    return report


def archive_entries(path: Path, ranges: list[list[int]]) -> dict[str, Any]:
    """Locate difference ranges using only ZIP headers, including payload offsets."""
    result: dict[str, Any] = {"is_zipfile": zipfile.is_zipfile(path), "entries": []}
    if not result["is_zipfile"]:
        return result
    try:
        with zipfile.ZipFile(path) as archive, path.open("rb") as raw:
            for info in archive.infolist():
                raw.seek(info.header_offset)
                header = raw.read(30)
                if len(header) != 30 or header[:4] != b"PK\x03\x04":
                    raise ValueError("Invalid local ZIP header")
                name_len, extra_len = struct.unpack_from("<HH", header, 26)
                start = info.header_offset + 30 + name_len + extra_len
                end = start + info.compress_size
                overlap = sum(
                    max(0, min(b + 1, end) - max(a, start)) for a, b in ranges
                )
                if overlap:
                    result["entries"].append(
                        {
                            "filename": info.filename,
                            "local_header_offset": info.header_offset,
                            "payload_start": start,
                            "payload_end_exclusive": end,
                            "compressed_size": info.compress_size,
                            "uncompressed_size": info.file_size,
                            "compression_type": info.compress_type,
                            "differing_payload_bytes": overlap,
                            "pickle_loaded": False,
                        }
                    )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        result["header_parse_error"] = repr(error)
    return result


def run(good_metadata: Path, bad_metadata: Path, output: Path) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CUDA_VISIBLE_DEVICES= for CPU-only comparison")
    if output.exists() or not output.is_relative_to(MAIN / "outputs"):
        raise ValueError("Output must be new under main outputs")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "failed",
        "command": sys.argv,
        "script_sha256": dual_sha256(Path(__file__)),
        "limitations": __doc__,
    }
    try:
        metadata_hashes = {
            str(p): dual_sha256(p) for p in (good_metadata, bad_metadata)
        }
        report["metadata_sha256_before"] = metadata_hashes
        rows = (
            json.loads(good_metadata.read_text())["passes"][0],
            json.loads(bad_metadata.read_text())["captures"][0],
        )
        expected_hashes = (
            "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
            "a8e786c151eedd1dca57ea5b108b901b6a6cb8731a3990b213bdffa418751905",
        )
        if tuple(row["hashlib_sha256"] for row in rows) != expected_hashes:
            raise ValueError(
                "Metadata does not identify the requested good/bad saved streams"
            )
        snapshots = tuple(Path(row["snapshot"]) for row in rows)
        for path, row in zip(snapshots, rows, strict=True):
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(MAIN / "outputs")
                or path.suffix != ".bin"
            ):
                raise ValueError(
                    f"Refuse anything other than an output snapshot: {path}"
                )
            if row["hashlib_sha256"] != row["cpython_sha256"]:
                raise ValueError("Metadata providers disagree")
            if output.is_relative_to(path.parent):
                raise ValueError(
                    "Report output must be separate from source snapshot directories"
                )
        report["comparison"] = scan(
            snapshots[0],
            snapshots[1],
            (rows[0]["hashlib_sha256"], rows[1]["hashlib_sha256"]),
        )
        if any(
            source["bytes_read"] != row["bytes_read"]
            or row["bytes_read"] != row["bytes_written"]
            for source, row in zip(report["comparison"]["sources"], rows, strict=True)
        ):
            raise ValueError("Snapshot length differs from recorded capture")
        if not report["comparison"]["gates_passed"]:
            raise ValueError("Saved snapshot stat/length/hash gate failed")
        report["archives"] = [
            archive_entries(path, report["comparison"]["difference_ranges_inclusive"])
            for path in snapshots
        ]
        for path, source in zip(
            snapshots, report["comparison"]["sources"], strict=True
        ):
            if stat(path) != source["stat_after"]:
                raise ValueError("Snapshot changed during header inspection")
        report["metadata_sha256_after"] = {
            str(p): dual_sha256(p) for p in (good_metadata, bad_metadata)
        }
        if report["metadata_sha256_after"] != metadata_hashes:
            raise ValueError("Metadata changed")
        report["status"] = "compared"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        with (output / "comparison.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("good-metadata", "bad-metadata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.good_metadata.resolve(), args.bad_metadata.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()
