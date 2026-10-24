"""Exercise all frozen requests on CPU, stopping immediately before subprocess."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

from render_absolute import absolute_scratch, load_inputs, probe

from src.synthetic_data_generation.dataset.court.scale.publication import write_content
from src.synthetic_data_generation.dataset.court.scale.runtime import (
    _NHTRenderer,
    renderer_fingerprint,
)
from src.synthetic_data_generation.rendering.nht import client


class BeforeSubprocess(Exception):
    """The real request and scene validation reached the intercepted boundary."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--adopted-manifest", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    assets = args.scratch / "assets"
    assets.mkdir()
    assert probe.file_sha256(args.adopted_manifest) == probe.ASSET_SHA
    manifest = json.loads(args.adopted_manifest.read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        assert relative.parts[0] == "scenes" and ".." not in relative.parts
        source = args.source_root / relative.relative_to("scenes")
        assert source.stat().st_size == entry["size_bytes"]
        assert probe.file_sha256(source) == entry["sha256"]
        destination = assets / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(args.adopted_manifest, assets / "assets.json")
    selection = probe.load_selection(args.selection)
    inputs = load_inputs(assets)
    frozen = probe.reconstruct(selection, inputs)
    checked = []
    with absolute_scratch() as scratch:
        backend = _NHTRenderer(
            scratch, fingerprint=renderer_fingerprint(), batch_size=1
        )
        for scenario, records in frozen:
            data = inputs[scenario["scene"]]
            asset_path, asset_hash = write_content(scratch, "assets", data["asset"])
            scene = {
                "scene_id": scenario["scene"],
                "asset_manifest_path": asset_path,
                "asset_manifest_sha256": asset_hash,
            }
            reached: list[dict[str, Any]] = []

            def intercept(
                argv: list[str],
                *,
                expected_records: list[dict[str, Any]] = records,
                expected_scenario: dict[str, Any] = scenario,
                result: list[dict[str, Any]] = reached,
                **kwargs: Any,
            ) -> NoReturn:
                assert kwargs["shell"] is False and kwargs["check"] is True
                assert "--without-source-images" in argv
                assert argv[argv.index("--batch-size") + 1] == "1"
                for flag in ("--scene", "--cameras", "--output"):
                    assert Path(argv[argv.index(flag) + 1]).is_absolute()
                payload = json.loads(
                    Path(argv[argv.index("--cameras") + 1]).read_text()
                )
                assert payload["schema"] == "nht_render_request_v1"
                assert [c["camera_id"] for c in payload["cameras"]] == [
                    record["sample_id"] for record in expected_records
                ]
                result.append(
                    {
                        "variant": expected_scenario["variant"],
                        "scene": expected_scenario["scene"],
                        "camera_count": len(payload["cameras"]),
                        "request_sha256": probe.digest(payload),
                        "absolute_paths": True,
                        "scene_validated": True,
                        "request_json_written": True,
                    }
                )
                raise BeforeSubprocess

            with patch.object(client.subprocess, "run", side_effect=intercept):
                try:
                    list(backend(scene, records))
                except BeforeSubprocess:
                    pass
                else:
                    raise AssertionError("The real NHT boundary was not reached")
            assert len(reached) == 1
            checked.extend(reached)
    assert len(checked) == 16
    assert sum(row["camera_count"] for row in checked) == 192
    result = {
        "schema": "court_d2_nht_preflight_v1",
        "selection_sha256": selection["selection_sha256"],
        "asset_manifest_sha256": probe.ASSET_SHA,
        "copied_verified_input_files": len(manifest["files"]),
        "checked_scenarios": checked,
        "checked_cameras": 192,
        "nht_subprocesses_started": 0,
        "gpu_calls": 0,
        "outcome": "passed",
        "boundary": "Existing _NHTRenderer and NHTRenderClient through request construction, scene validation and JSON write; subprocess.run intercepted before execution.",
    }
    probe.write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
