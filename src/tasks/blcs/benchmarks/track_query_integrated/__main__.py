"""CLI boundary for the integrated BLCS track-query benchmark package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from src.tasks.blcs.benchmarks.contracts import (
    benchmark_path_resolver,
    resolve_benchmark_cli_path,
)
from src.tasks.blcs.benchmarks.track_query_integrated import (
    _WORKER_PREFIX,
    BENCHMARK_CASES,
    CANDIDATES,
    COMPONENT,
    DTYPES,
    _run_worker,
    execute,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="blcs.track_query_integrated",
    fields=(
        BoundaryPathField(
            "evidence",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "runtime_result",
            PathRole.PROJECT,
            PathDirection.OUTPUT,
            PathKind.FILE,
        ),
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--runtime-result", type=Path)
    parser.add_argument(
        "--record-evidence",
        action="store_true",
        help="Explicitly replace stable evidence with this fresh result.",
    )
    parser.add_argument("--worker-candidate", choices=CANDIDATES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-case",
        choices=tuple(case.name for case in BENCHMARK_CASES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-dtype", choices=DTYPES, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker_candidate is not None:
        if args.worker_case is None or args.worker_dtype is None:
            raise SystemExit(
                "--worker-case and --worker-dtype are required with "
                "--worker-candidate"
            )
        print(
            _WORKER_PREFIX
            + json.dumps(
                _run_worker(
                    args.worker_candidate,
                    args.worker_case,
                    args.worker_dtype,
                ),
                allow_nan=False,
            )
        )
        return 0
    if args.evidence is None or args.runtime_result is None:
        raise SystemExit("--evidence and --runtime-result are required")
    resolver = benchmark_path_resolver()
    paths = PATH_BOUNDARY.validate(
        {
            "evidence": resolve_benchmark_cli_path(args.evidence, resolver=resolver),
            "runtime_result": resolve_benchmark_cli_path(
                args.runtime_result,
                resolver=resolver,
            ),
        },
        resolver=resolver,
    )
    evidence_path = cast(Path, paths["evidence"])
    runtime_result_path = cast(Path, paths["runtime_result"])
    result = execute(
        evidence_path=evidence_path,
        runtime_result_path=runtime_result_path,
        record_evidence=args.record_evidence,
    )
    decision = cast(Mapping[str, Any], result["decision"])
    print(
        json.dumps(
            {
                "component": COMPONENT,
                "decision": decision["status"],
                "complete_same_shape_triplets": decision[
                    "complete_same_shape_triplets"
                ],
                "runtime_result": str(runtime_result_path),
                "recorded_evidence": args.record_evidence,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
