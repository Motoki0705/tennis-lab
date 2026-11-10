"""Read immutable review evidence; recovery never starts an agent or edits its source."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from parent.policy import digest, object_value
from parent_review_policy import (
    execution_stages,
    load_json,
    output_object,
    parse_output,
    request_manifest,
    validate_result,
)


def read_json(path: Path) -> dict[str, Any]:
    return object_value(load_json(path.read_text(encoding="utf-8")))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_final(run_dir: Path) -> Path:
    """Both copies must agree. A failed runtime may only have final.md."""
    paths = [
        p
        for p in (run_dir / "agent-runtime/final.md", run_dir / "agent_output.md")
        if p.exists()
    ]
    if not paths:
        raise ValueError("saved final is missing")
    if len({p.read_bytes() for p in paths}) != 1:
        raise ValueError("saved final copies disagree")
    return paths[0]


def runtime_evidence(run_dir: Path) -> dict[str, Any]:
    runtime = read_json(run_dir / "agent-runtime/result.json")
    contract = object_value(runtime.get("contract"))
    if (
        runtime.get("agent_spawned") is not True
        or not isinstance(runtime.get("result"), str)
        or runtime.get("result") not in {"completed", "failed"}
        or runtime.get("cleanup_complete") is not True
        or type(runtime.get("exit_code")) is not int
        or type(runtime.get("cli_exit_code")) is not int
        or type(runtime.get("finished_at")) not in {int, float}
        or not math.isfinite(runtime["finished_at"])
        or runtime["finished_at"] <= 0
    ):
        raise ValueError("runtime exit/cleanup is not confirmed")
    if any(
        not isinstance(runtime.get(key), str)
        or Path(runtime[key]).resolve() != path.resolve()
        for key, path in (
            ("run_dir", run_dir / "agent-runtime"),
            ("output_path", run_dir / "agent_output.md"),
            ("final_path", run_dir / "agent-runtime/final.md"),
        )
    ) or (
        runtime.get("prompt_sha256") != file_digest(run_dir / "prompt.md")
        or contract.get("json_schema_sha256")
        != file_digest(run_dir / "agent-runtime/output-schema.json")
    ):
        raise ValueError("runtime belongs to another run/prompt/schema")
    return runtime


def parse_final(run_dir: Path, request: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate the exact saved schema as well as the semantic review contract."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError, ValidationError

    final = saved_final(run_dir)
    text = final.read_text(encoding="utf-8")
    parsed = parse_output(text, [p["number"] for p in request["prs"]])
    schema = read_json(run_dir / "agent-runtime/output-schema.json")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(output_object(text))
    except (SchemaError, ValidationError) as exc:
        raise ValueError(
            f"saved final violates its output schema: {exc.message}"
        ) from exc
    return final, parsed


def source_evidence(state_dir: Path, run: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not re.fullmatch(r"parent-[1-9][0-9]*-[0-9a-f]{32}", run):
        raise ValueError("invalid source run ID")
    directory = state_dir / "reviews" / run
    request = request_manifest(read_json(directory / "request.json"), instructed=False)
    execution = read_json(directory / "execution.json")
    result = validate_result(read_json(directory / "result.json"))
    if (
        request["review_run_id"] != run
        or result["request"] != request
        or result["kind"] != "result"
        or execution.get("request_sha256") != digest(request)
        or type(execution.get("agent_invocations")) is not int
        or execution["agent_invocations"] != 1
        or execution.get("status") not in {"published", "publication_failed"}
        or (
            "authorization" in request
            and execution.get("authorization") != request["authorization"]
        )
    ):
        raise ValueError("source run execution/result/request mismatch")
    runtime = runtime_evidence(directory)
    cwd = runtime.get("contract", {}).get("cwd")
    if (
        not isinstance(cwd, str)
        or Path(cwd).name != "review-" + run
        or Path(cwd).exists()
    ):
        raise ValueError("review workspace cleanup is not confirmed")
    context = read_json(directory / "context.json")
    final, parsed = parse_final(directory, request)
    output_path = directory / "output.json"
    envelope = (
        validate_result(read_json(output_path)) if output_path.exists() else result
    )
    if (
        envelope["kind"] != "result"
        or envelope["request"] != request
        or envelope["started_at"] != execution["started_at"]
    ):
        raise ValueError("saved output run mismatch")
    candidate = {
        **envelope,
        **parsed,
        "status": "succeeded" if parsed["verdict"] != "incomplete" else "failed",
        "error": None
        if parsed["verdict"] != "incomplete"
        else "未確認の対象があります。",
    }
    candidate["prs"] = [
        {
            **p,
            "child_issues": next(
                c["child_issues"] for c in context["prs"] if c["number"] == p["number"]
            ),
        }
        for p in parsed["prs"]
    ]
    validate_result(candidate)
    if (
        result["status"] == "succeeded"
        or any(p["coverage"] == "reviewed" for p in result["prs"])
    ) and any(
        candidate[k] != result[k]
        for k in ("verdict", "summary", "cross_findings", "prs")
    ):
        raise ValueError("saved final contradicts the source result")
    for pr in request["prs"]:
        if file_digest(directory / f"pr-{pr['number']}.diff") != pr["diff_sha256"]:
            raise ValueError("source frozen diff digest mismatch")
    # The recorded candidate is evidence, never a replacement for parsing final.
    if output_path.exists() and read_json(output_path) != candidate:
        raise ValueError("saved output contradicts final")
    stages = execution_stages(execution)
    if stages is not None and (
        stages.get("agent", {}).get("cleanup_complete") is not True
        or stages["agent"].get("status") != "finished"
        or stages["agent"].get("runtime_sha256") != digest(runtime)
        or stages["agent"].get("exit_code") != runtime["exit_code"]
        or stages["agent"].get("workspace_cleanup_complete") is not True
        or stages.get("output", {}).get("status") != "validated"
        or stages["output"].get("final_sha256") != file_digest(final)
        or stages["output"].get("output_sha256") != digest(candidate)
    ):
        raise ValueError("source stage evidence is incomplete or changed")
    validation = {
        "source_request_sha256": digest(request),
        "source_execution_sha256": digest(execution),
        "source_result_sha256": digest(result),
        "runtime_sha256": digest(runtime),
        "final_sha256": file_digest(final),
        "output_sha256": digest(candidate),
        "current_prs": request["prs"],
    }
    return candidate, validation


def verify_adoption(state_dir: Path, adoption: dict[str, Any]) -> None:
    validate_result(adoption)
    candidate, validation = source_evidence(state_dir, adoption["source_run"])
    if validation != adoption["validation"]:
        raise ValueError("adoption source evidence changed")
    source = candidate["request"]
    target = adoption["request"]
    if any(source[k] != target[k] for k in ("repo", "parent_issue", "prs")):
        raise ValueError("recover requires the source target snapshots")
    for key in ("status", "verdict", "summary", "error", "cross_findings", "prs"):
        if adoption[key] != candidate[key]:
            raise ValueError("adoption contradicts the verified final")
