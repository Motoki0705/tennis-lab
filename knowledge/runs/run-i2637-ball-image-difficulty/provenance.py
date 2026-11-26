"""Verify the archive and the later CLI reannotation using only relevant log fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

from measure import LABEL_DIR, sha

REVISION = f"{LABEL_DIR}_rev1"
REVISED = ("clip_012", "clip_015", "clip_026")


def downloads(history: Path, archive: Path) -> list[dict]:
    # Do not read other visits, log private conversation IDs, or modify the history DB.
    with sqlite3.connect(history.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute(
            """select target_path, tab_url, referrer, start_time, received_bytes
               from downloads where target_path like ?""",
            ("%" + archive.name,),
        ).fetchall()
    records = [
        dict(tab_host=urlsplit(tab).hostname, referrer_host=urlsplit(referrer).hostname,
             start_utc=(datetime(1601, 1, 1, tzinfo=UTC) + timedelta(microseconds=start)).isoformat(),
             bytes=size)
        for path, tab, referrer, start, size in rows
        if path.replace("\\", "/").split("/")[-1] == archive.name and size == archive.stat().st_size
    ]
    if not records:
        raise ValueError("No browser download record matches archive basename and size")
    return records


def original(root: Path, archive: Path) -> dict:
    labels = root / LABEL_DIR
    files = {f"clip_{i:03d}_ball_positions.json" for i in range(69)}
    if {f.name for f in labels.glob("*.json")} != files:
        raise ValueError("Unexpected set of original annotations")
    entries = []
    metadata_keys = set()
    with ZipFile(archive) as source:
        members = {Path(name).name: name for name in source.namelist() if name.endswith(".json")}
        if len(members) != 69 or set(members) != files:
            raise ValueError("Archive member names differ")
        for filename in sorted(files):
            payload = (labels / filename).read_bytes()
            if payload != source.read(members[filename]):
                raise ValueError(f"Local annotation differs from archive: {filename}")
            metadata = json.loads(payload)["metadata"]
            metadata_keys.update(metadata)
            entries.append(dict(file=filename, sha256=hashlib.sha256(payload).hexdigest(),
                                embedded_created_utc=metadata["created_utc"]))
    return dict(
        folder=LABEL_DIR, archive=archive.name, bytes=archive.stat().st_size, archive_sha256=sha(archive),
        embedded_time_min=min(e["embedded_created_utc"] for e in entries),
        embedded_time_max=max(e["embedded_created_utc"] for e in entries),
        declared_model=None, recorded_execution=None, metadata_keys=sorted(metadata_keys), members=entries,
    )


def revision(root: Path, campaign: Path, session_directory: Path) -> dict:
    launcher = campaign / "launch.sh"
    events_path = campaign / "events.log"
    events = [line.split() for line in events_path.read_text().splitlines()]
    runs = []
    for clip in REVISED:
        basename = f"{clip}_ball_positions.json"
        worker = campaign / clip
        event = worker / "codex.jsonl"
        with event.open() as stream:
            thread = json.loads(stream.readline())
        if thread["type"] != "thread.started":
            raise ValueError("Missing CLI thread start")
        (rollout,) = session_directory.glob(f"*{thread['thread_id']}.jsonl")
        with rollout.open("rb") as stream:
            for line in stream:
                context = json.loads(line)
                if context["type"] == "turn_context":
                    break
            else:
                raise ValueError(f"No turn context for {clip}")
        original_digest = sha(root / LABEL_DIR / basename)
        revised_digest = sha(root / REVISION / basename)
        if (sha(worker / "input/original_ball_positions.json") != original_digest
                or sha(worker / "revised" / basename) != revised_digest):
            raise ValueError(f"Campaign input or output differs for {clip}")
        (started,) = [e[1] for e in events if e[0] == "START" and e[2] == clip]
        (finished,) = [e[1] for e in events if e[0] == "END" and e[2:] == [clip, "rc=0"]]
        actual = context["payload"]
        if (actual["model"], actual["effort"]) != ("gpt-6.2-sol", "max"):
            raise ValueError(f"Unexpected effective model and effort for {clip}")
        runs.append(dict(
            clip=clip, start=started, finish=finished, exit_code=0,
            context_time_utc=context["timestamp"], effective_model=actual["model"], effort=actual["effort"],
            turn_context_line_sha256=hashlib.sha256(line).hexdigest(),
            codex_jsonl_sha256=sha(event),
            original_sha256=original_digest, revised_sha256=revised_digest,
        ))
    return dict(
        campaign=campaign.name, launcher_sha256=sha(launcher), events_sha256=sha(events_path),
        runs=runs, publication_folder=REVISION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "archive", "chrome-history", "campaign", "session-directory", "output"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    report = dict(
        original=original(args.root, args.archive),
        matching_download_records=downloads(args.chrome_history, args.archive),
        cli_revision=revision(args.root, args.campaign, args.session_directory),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
