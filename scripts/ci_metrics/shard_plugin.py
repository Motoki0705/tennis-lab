"""Partition after CPU selection, preserving explicitly selected fixture groups."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from .lanes import partition_nodeids, read_plan
from .pytest_plugin import record_cpu_universe
from .shards import assign_groups, group_key, read_timing_profile


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ci-shard-index", type=int)
    parser.addoption("--ci-shard-count", type=int)
    parser.addoption("--ci-shard-weights", type=Path)
    parser.addoption("--ci-lane-plan", type=Path)


@pytest.hookimpl(wrapper=True, tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> Generator[None, Any, Any]:
    result = yield
    index = config.getoption("--ci-shard-index")
    count = config.getoption("--ci-shard-count")
    try:
        if type(index) is not int or type(count) is not int or not 0 <= index < count:
            raise ValueError("both a valid shard index and count are required")
        lane_plan = read_plan(config.getoption("--ci-lane-plan"), root=Path.cwd())
        record_cpu_universe(items)
        universe = [item.nodeid for item in items]
        nodeids, deferred = partition_nodeids(universe, lane_plan)
        requested_items = [item for item in items if item.nodeid not in deferred]
        profile, profile_digest = read_timing_profile(
            config.getoption("--ci-shard-weights")
        )
        split_files = profile.get("split_files", [])
        for item in requested_items:
            if item.nodeid.split("::", 1)[0] not in split_files:
                continue
            if not isinstance(item, pytest.Function):
                raise ValueError(
                    "only Python function items may use case-level sharding"
                )
            shared = [
                name
                for name, definitions in item._fixtureinfo.name2fixturedefs.items()
                if definitions and definitions[-1].scope in {"module", "class"}
            ]
            if shared:
                raise ValueError(
                    f"cannot split {item.nodeid}: shared fixtures {sorted(shared)}"
                )
        plan = assign_groups(
            nodeids,
            count,
            profile["file_seconds"],
            split_files=split_files,
            test_weights=profile.get("test_seconds", {}),
        )
    except (OSError, ValueError, TypeError) as error:
        # Send a normal collection failure to the xdist controller. Raising a
        # UsageError inside a worker can otherwise hide its diagnostic message.
        config.hook.pytest_collectreport(
            report=pytest.CollectReport(
                nodeid="CPU shard assignment",
                outcome="failed",
                longrepr=str(error),
                result=[],
            )
        )
        items.clear()
        return result
    selected: list[pytest.Item] = []
    excluded: list[pytest.Item] = []
    for item in items:
        if item.nodeid in deferred:
            excluded.append(item)
            continue
        target = plan["assignments"][group_key(item.nodeid, plan["split_files"])]
        (selected if target == index else excluded).append(item)
    destination = Path(config.getoption("--ci-metrics-dir"))
    role = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    with (destination / f"shard-{role}.json").open("x") as stream:
        json.dump(
            {
                "schema_version": 2,
                "index": index,
                "count": count,
                "profile_sha256": profile_digest,
                "plan_id": lane_plan["plan_id"],
                "policy_sha256": lane_plan["policy_sha256"],
                "scope": lane_plan["scope"],
                "universe_nodeids": universe,
                "requested_nodeids": nodeids,
                "deferred": deferred,
                "selected_nodeids": [item.nodeid for item in selected],
                **plan,
            },
            stream,
        )
    items[:] = selected
    config.hook.pytest_deselected(items=excluded)
    return result
