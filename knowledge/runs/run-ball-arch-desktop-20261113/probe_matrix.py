"""Sequential exact-shape qualification; OOM candidates never become training."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rebuild_dataset import write

SCOPES = ("msda", "encoder_msda")


def archive_probe(root, case, number, scope):
    import tempfile

    from desktop import REMOTE, publish_index, stage_name
    from rebuild_dataset import sha

    from src.utils.artifact_store import RcloneSnapshotStore
    directory = root / "probes" / scope / stage_name(case, number)
    if not directory.exists():
        return None
    if not json.loads((root / "archive-acceptance.json").read_text())["complete"]:
        raise ValueError("Original Drive archive must finish before diagnostic publication")
    store = RcloneSnapshotStore(directory, remote="gdrive",
            remote_root=REMOTE+"/diagnostics/"+scope+"/"+directory.name,
            config_path=Path(os.environ["RCLONE_CONFIG"]))
    generation="probe-"+str(time.time_ns())
    receipt=store.publish(generation)
    with tempfile.TemporaryDirectory(prefix="probe-readback-",dir=root) as temporary:
        verification=RcloneSnapshotStore(Path(temporary)/"verified",remote="gdrive",remote_root=store.remote_root,
                        config_path=Path(os.environ["RCLONE_CONFIG"])).restore(generation,manifest_sha256=receipt["manifest_sha256"])
    path=root/"probe-receipts"/(case+"-"+scope+".json")
    retired=[dict(path=str(p.relative_to(root)),size_bytes=p.stat().st_size,sha256=sha(p)) for p in directory.glob("*.pt")]
    write(path,dict(receipt=receipt,readback=verification,local_retired=retired,
                   reason="discarded diagnostic updates; immutable verified Drive copy retained"))
    publish_index(root,path,Path(os.environ["RCLONE_CONFIG"]))
    for record in retired:
        (root/record["path"]).unlink()
    return str(path)


def probe(root, case, scope):
    number = 1 if case == "T1" else 0
    stop = 5503 if case == "T1" else 3
    command = [sys.executable, str(Path(__file__).with_name("desktop.py")), "stage", "--root", str(root),
               "--case", case, "--number", str(number), "--scope", scope, "--stop", str(stop)]
    if case == "T2":
        command += ["--joint"]
    log = root / "probe-logs" / (case + "-" + scope + ".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("xb", buffering=0) as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        failure = json.loads((root / ("probe-FAILED-" + case + "-" + scope + ".json")).read_text())
        if failure.get("kind") != "oom":
            raise RuntimeError("Probe failed beyond an expected allocator OOM: " + str(log))
        receipt=archive_probe(root,case,number,scope)
        return dict(case=case, scope=scope, status="oom", failure=failure, log=str(log), persistence=receipt)
    record = json.loads((root / f"probe-environment-{case}-{number}-{scope}-result.json").read_text())
    samples = record["samples"]
    receipt=archive_probe(root,case,number,scope)
    return dict(case=case, scope=scope, status="passed", samples=samples,
                warm_update_seconds=statistics.median(r["seconds"] for r in samples[1:]),
                peak_allocated_bytes=max(r["peak_vram_bytes"] for r in samples),
                peak_reserved_bytes=max(r["peak_reserved_vram_bytes"] for r in samples), log=str(log), persistence=receipt)


def main(root):
    if not os.environ.get("TENNIS_RUN_ID"):
        raise ValueError("Shared queue required")
    if not json.loads((root/"native-qualification.json").read_text())["complete"]:
        raise ValueError("Native and full-shape qualification must pass before probes")
    path = root / "probe-matrix.json"
    if path.exists():
        raise ValueError("Probe matrix already exists; no automatic retry")
    results = []
    for scope in SCOPES:
        for case in ("T1", "T2"):
            record = probe(root, case, scope)
            results.append(record)
            write(path, dict(complete=False, results=results, updated_at=time.time()))
    passing = [s for s in SCOPES if all(any(r["case"] == c and r["scope"] == s and r["status"] == "passed" for r in results) for c in ("T1", "T2"))]
    if not passing:
        raise RuntimeError("No common checkpoint scope fits T1 and T2 under the allocator cap")
    scope = min(passing, key=lambda s: sum(r["warm_update_seconds"] for r in results if r["scope"] == s))
    c1 = probe(root, "C1", scope)
    results.append(c1)
    if c1["status"] != "passed":
        raise RuntimeError("Unchanged C1 recipe does not fit the allocator cap")
    write(path, dict(complete=True, results=results, updated_at=time.time()))
    write(root / "checkpoint-policy.json", dict(schema_version=1, confirmed=True, scope=scope,
          t1_t2_identical=True, production_mutable=False, selected_by="minimum summed warm median update seconds among common passing scopes",
          allocator_fraction=.80, scientific_shape_unchanged=True, cnn_cases_checkpointing=False,
          evidence="probe-matrix.json"))
    write(root / "probe-acceptance.json", dict(complete=True, scope=scope, evidence="probe-matrix.json",
          scientific_updates_consumed=0, t1_resume_model_adamw_rng_sampler_audited=True,
          t2_unfrozen_diagnostic_only=True, c1_recipe_unchanged=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root.resolve())
