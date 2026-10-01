"""Explicit, read-only ChatGPT delegation manifests and request text."""

from __future__ import annotations

import json
from typing import Any

from .catalog import TARGETS, Catalog
from .metrics import missing_summary


def make_handoff(
    catalog: Catalog, revision: str, selections: list[dict[str, Any]], view: str
) -> dict[str, Any]:
    if not 1 <= len(selections) <= 20:
        raise ValueError("一度に選択できる動画は1〜20件です")
    catalog.check_revision(revision)
    items: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    seen: set[str] = set()
    for selection in selections:
        clip_id = selection["clip_id"]
        if clip_id in seen:
            raise ValueError("同じ動画が重複して選択されています")
        seen.add(clip_id)
        clip = catalog.clips[clip_id]
        catalog.check_clip_current(clip)
        if clip.errors or not clip.video:
            raise ValueError(f"{clip_id}: 動画/manifestのエラーを先に確認してください")
        if clip.active:
            raise ValueError(f"{clip_id}: 稼働中の担当があります")
        targets = selection.get("targets", list(TARGETS))
        if not targets or any(t not in TARGETS for t in targets):
            raise ValueError("対象はball/playerを選択してください")
        target_items = {}
        for target in targets:
            if "versions" in selection:
                version = catalog.select(
                    clip_id, target, selection["versions"].get(target)
                )
            else:
                version = catalog.default_version(clip, target, view)
            if version is None and clip.versions[target]:
                raise ValueError(
                    f"{clip_id}/{target}: 未採用の候補があります。詳細で版を選択してください"
                )
            if version:
                _, stats, _ = catalog.read_version(version)
            else:
                stats = missing_summary(len(clip.manifest.frames))
            if stats["state"] == "invalid":
                raise ValueError(f"{clip_id}/{target}: 選択JSONに検証エラーがあります")
            pending = stats["unreviewed_ranges"]
            if not pending and not selection.get("refine", False):
                continue
            if not pending and not stats["uncertain_ranges"] and not stats["issues"]:
                continue
            used_targets.add(target)
            target_items[target] = {
                "task": "review_unreviewed" if pending else "inspect_uncertainty",
                "frame_ranges_half_open": pending or stats["uncertain_ranges"],
                "reviewed_frames_to_preserve": stats["reviewed"],
                "uncertain_ranges": stats["uncertain_ranges"],
                "issues": stats["issues"],
                "annotation": catalog.version_payload(version, statistics=False)
                if version
                else None,
            }
        if target_items:
            items.append(
                {
                    "clip_id": clip.id,
                    "video_filename": clip.manifest.filename,
                    "video_path": str(clip.video),
                    "video_sha256": clip.manifest.sha256,
                    "manifest_path": str(clip.manifest_path),
                    "manifest_sha256": clip.manifest_sha,
                    "width": clip.manifest.width,
                    "height": clip.manifest.height,
                    "frame_count": len(clip.manifest.frames),
                    "time_base": clip.manifest.time_base,
                    "fps": clip.manifest.nominal_fps,
                    "targets": target_items,
                    "warnings": sorted(set(clip.warnings)),
                }
            )
    if not items:
        if all(item.get("refine") for item in selections):
            raise ValueError("選択版に再確認対象の不確実性は記録されていません")
        raise ValueError(
            "選択範囲に未確認フレームがありません。品質確認は詳細の再確認用依頼を使ってください"
        )
    payload = {
        "schema_version": "chat_annotation_handoff.v1",
        "catalog_revision": revision,
        "view": view,
        "range_convention": "zero-based half-open display frame indices, including context",
        "clips": items,
        "automatic_submission": False,
    }
    introduction = [
        "# 動画アノテーションの委託",
        "",
        "以下の元動画と、存在する場合は選択した既存注釈JSONを添付します。動画のファイル名は保持してください。",
        "対象はクリップ内の全表示フレーム（参考区間を含む）です。指定した未確認範囲を確認し、確認済みの行は維持してください。",
        "範囲は0始まりの半開区間[start, stop)です。下書き・未採用ZIPを採用済みと混同しないでください。",
        "確認済みの不確実性を再確認する場合も、画像から判別できない値を推測で埋めないでください。",
    ]
    request_sources = []
    requests: dict[str, dict[str, Any]] = {}
    for target in sorted(used_targets):
        path = catalog.safe(
            catalog.root / "project_kits" / f"{target}_detection" / "REQUEST.txt"
        )
        if not path.is_file():
            raise ValueError(
                f"委託用REQUESTがありません: project_kits/{target}_detection/REQUEST.txt"
            )
        request_sources.append(catalog.relative(path))
        target_payload = dict(
            payload,
            clips=[
                dict(item, targets={target: item["targets"][target]})
                for item in items
                if target in item["targets"]
            ],
        )
        text = [
            *introduction,
            "",
            f"この依頼の対象は {target} のみです。",
            "## 添付ファイル・作業範囲",
            "```json",
            json.dumps(target_payload, ensure_ascii=False, indent=2),
            "```",
            "",
            f"## {target} の正式な依頼文",
            path.read_text(encoding="utf-8"),
        ]
        requests[target] = {
            "text": "\n".join(text),
            "manifest": target_payload,
            "request_source": catalog.relative(path),
        }
    return {
        "manifest": payload,
        "text": next(iter(requests.values()))["text"],
        "requests": requests,
        "request_sources": request_sources,
    }
