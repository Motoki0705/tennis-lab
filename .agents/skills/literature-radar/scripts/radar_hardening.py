#!/usr/bin/env python
"""Harden the legacy literature-radar ingester without changing old records.

New hourly payloads are validated against explicit scoring and topic contracts.
Canonicalization treats DataCite arXiv DOIs as arXiv aliases, deduplicates by all
known identifiers, enforces collector/topic/backlog quotas, repairs the legacy
digest preamble, and publishes a cheap preflight status file for schedules.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import radar_ingest as legacy

EVIDENCE_RANK = {
    "abstract": 0,
    "fulltext": 1,
    "fulltext-code": 2,
    "fulltext-code-data": 3,
}
SCORE_KEYS = (
    "task_fit",
    "repo_fit",
    "evidence_quality",
    "experiment_quality",
    "adoption_feasibility",
)
SAFE_DIGEST_PREAMBLE = (
    "# Literature Radar â€” {date}\n\n"
    "ã“ã®æ—¥æ¬¡ãƒ€ã‚¤ã‚¸ã‚§ã‚¹ãƒˆã®è‡ªå‹•åŽé›†åŒºé–“ã¯GitHub ActionsãŒæ›´æ–°ã—ã¾ã™ã€‚\n"
    "æ—¥æ¬¡curatorã¯è‡ªå‹•åŒºé–“ã®å¤–å´ã ã‘ã‚’ç·¨é›†ã—ã¦ãã ã•ã„ã€‚\n\n"
)


def _normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalise_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), path, "", ""))


def _arxiv_from_datacite_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    doi = legacy._normalise_doi(value)
    match = re.fullmatch(r"10\.48550/arxiv\.(.+)", doi, flags=re.IGNORECASE)
    return legacy._normalise_arxiv(match.group(1)) if match else None


def _publisher_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if _arxiv_from_datacite_doi(value):
        return None
    return legacy._normalise_doi(value)


def _paper_identifiers(paper: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    identifiers = paper.get("identifiers") or {}
    doi = _publisher_doi(identifiers.get("doi"))
    arxiv = identifiers.get("arxiv")
    if isinstance(arxiv, str) and arxiv.strip():
        arxiv = legacy._normalise_arxiv(arxiv)
    else:
        arxiv = _arxiv_from_datacite_doi(identifiers.get("doi"))
    openreview = identifiers.get("openreview")
    if isinstance(openreview, str) and openreview.strip():
        openreview = legacy._normalise_openreview(openreview)
    else:
        openreview = None
    return doi, arxiv, openreview


def canonical_paper_id(payload: dict[str, Any]) -> str:
    paper = payload["paper"]
    doi, arxiv, openreview = _paper_identifiers(paper)
    if doi:
        return f"paper-doi-{legacy._slug(doi)}"
    if arxiv:
        return f"paper-arxiv-{legacy._slug(arxiv)}"
    if openreview:
        return f"paper-openreview-{legacy._slug(openreview)}"
    title = _normalise_title(str(paper["title"]))
    year = int(paper["year"])
    digest = hashlib.sha256(f"{title}\n{year}".encode()).hexdigest()[:16]
    return f"paper-title-{year}-{digest}"


def paper_aliases(paper: dict[str, Any]) -> set[str]:
    doi, arxiv, openreview = _paper_identifiers(paper)
    aliases: set[str] = set()
    if doi:
        aliases.add(f"doi:{doi}")
    if arxiv:
        aliases.add(f"arxiv:{arxiv}")
    if openreview:
        aliases.add(f"openreview:{openreview}")
    title = _normalise_title(str(paper.get("title", "")))
    year = paper.get("year")
    if title and isinstance(year, int):
        aliases.add(f"title:{year}:{title}")
    primary = (paper.get("urls") or {}).get("primary")
    if isinstance(primary, str) and primary.strip():
        aliases.add(f"url:{_normalise_url(primary)}")
    return aliases


def _sanitise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(payload)
    identifiers = clean["paper"]["identifiers"]
    alias_arxiv = _arxiv_from_datacite_doi(identifiers.get("doi"))
    if alias_arxiv:
        identifiers["doi"] = None
        if not identifiers.get("arxiv"):
            identifiers["arxiv"] = alias_arxiv
    return clean


def validate_hardened_candidate(
    payload: dict[str, Any],
    repo_root: Path,
    config: dict[str, Any],
    expected_collector: str | None = None,
) -> None:
    legacy.validate_raw_candidate(payload, repo_root, config, expected_collector)
    collector_id = str(payload["collector_id"])
    collector = config["collectors"][collector_id]
    screening = payload["screening"]

    tasks = {str(item) for item in screening.get("tasks", [])}
    allowed_tasks = {str(item) for item in collector.get("tasks", [])}
    if not tasks <= allowed_tasks:
        raise legacy.CandidateError(
            f"screening.tasks exceed collector {collector_id!r} responsibility: "
            f"{sorted(tasks - allowed_tasks)}"
        )

    topic = screening.get("topic")
    if topic not in collector.get("topics", []):
        raise legacy.CandidateError(
            f"screening.topic {topic!r} is not configured for {collector_id!r}"
        )

    breakdown = screening.get("score_breakdown")
    if not isinstance(breakdown, dict) or set(breakdown) != set(SCORE_KEYS):
        raise legacy.CandidateError(
            f"screening.score_breakdown must contain exactly {list(SCORE_KEYS)}"
        )
    maxima = config["ingestion"]["score_weights"]
    for key in SCORE_KEYS:
        value = breakdown.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise legacy.CandidateError(f"screening.score_breakdown.{key} must be an integer")
        if not 0 <= value <= int(maxima[key]):
            raise legacy.CandidateError(
                f"screening.score_breakdown.{key} must be 0..{maxima[key]}"
            )
    relevance = int(screening["relevance_score"])
    if sum(int(breakdown[key]) for key in SCORE_KEYS) != relevance:
        raise legacy.CandidateError(
            "screening.relevance_score must equal the score_breakdown sum"
        )

    evidence = str(screening["evidence_level"])
    minimum = str(config["ingestion"].get("minimum_evidence_level", "fulltext"))
    if EVIDENCE_RANK[evidence] < EVIDENCE_RANK[minimum:
        raise legacy.CandidateError(
            f"screening.evidence_level {evidence'ÜŸH\È™[ÝÈZ[š[][HÛZ[š[][H\ŸH‚ˆ
BˆØ\H[
ÛÛ™šYÖÈš[™Ù\Ý[Ûˆ—VÈ™]šY[˜ÙWÜØÛÜ™WØØ\È—VÙ]šY[˜ÙWJBˆYˆ[
œ™XZÙÝÛ–È™]šY[˜ÙWÜ]X[]H—JHˆØ\‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆˆ™]šY[˜ÙWÜ]X[]H^ÙYYÈÙ]šY[˜ÙH\ŸHØ\ØØ\H‚ˆ
B‚ˆ[ÝÙYÚÚ[™ÈHÙ]
ÛÛ™šYË™Ù]
˜[ÝÙYÜÛÝ\˜ÙWÚÚ[™ÈŠHÜˆ×JBˆÛÝ\˜ÙWÚÚ[™ÈHÜÝŠÛÝ\˜ÙK™Ù]
šÚ[™ŠJH›ÜˆÛÝ\˜ÙH[ˆ^[ØYÈœÛÝ\˜Ù\È—_BˆYˆÛÝ\˜ÙWÚÚ[™ÈH[ÝÙYÚÚ[™Î‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆˆœÛÝ\˜Ù\ÈÛÛZ[ˆ[œÝ\ÜYÚ[™ÎˆÜÛÜY
ÛÝ\˜ÙWÚÚ[™ÈH[ÝÙYÚÚ[™Ê_H‚ˆ
BˆYˆ]šY[˜ÙH[ˆÈ™[^XÛÙH‹™[^XÛÙKY]HŸH[™˜ÛÙHˆ›Ý[ˆÛÝ\˜ÙWÚÚ[™Î‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆžÙ]šY[˜Ù_H™\]Z\™\È[ˆÙ™šXÚX[ÛÙHÛÝ\˜ÙHŠBˆYˆ]šY[˜ÙHOH™[^XÛÙKY]Hˆ[™™]\Ù]ˆ›Ý[ˆÛÝ\˜ÙWÚÚ[™Î‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠ™[^XÛÙKY]H™\]Z\™\È[ˆÙ™šXÚX[]\Ù]ÛÝ\˜ÙHŠB‚‚™YˆÜ™XÛÜ™Ù]J™XÛÜ™ˆXÝÜÝ‹[žWK[Y^›Û™NˆÝŠHOˆÝŽ‚ˆ™]\›ˆYØXÞK—Ù]WÛÙ—Ú\ÛÊÝŠ™XÛÜ™È™š\œÝÜÙY[ˆ—JK[Y^›Û™JB‚‚™YˆÜ™XÛÜ™Ú\×ØÛÛXÝÜŠ™XÛÜ™ˆXÝÜÝ‹[žWKÛÛXÝÜ—ÚYˆÝŠHOˆ›ÛÛ‚ˆ™]\›ˆ[žJˆ\Ú[œÝ[˜ÙJ][KXÝ
H[™][K™Ù]
˜ÛÛXÝÜ—ÚYŠHOHÛÛXÝÜ—ÚYˆ›Üˆ][H[ˆ™XÛÜ™™Ù]
™\ØÛÝ™\šY\È‹×JBˆ
B‚‚™YˆÜ™XÛÜ™ÝÜXÜÊ™XÛÜ™ˆXÝÜÝ‹[žWJHOˆÙ]ÜÝ—N‚ˆ™]\›ˆÂˆÝŠ][VÈœØÜ™Y[š[™È—VÈÜXÈ—JBˆ›Üˆ][H[ˆ™XÛÜ™™Ù]
™\ØÛÝ™\šY\È‹×JBˆYˆ\Ú[œÝ[˜ÙJ][KXÝ
Bˆ[™\Ú[œÝ[˜ÙJ][K™Ù]
œØÜ™Y[š[™ÈŠKXÝ
Bˆ[™\Ú[œÝ[˜ÙJ][VÈœØÜ™Y[š[™È—K™Ù]
ÜXÈŠKÝŠBˆB‚‚™Yˆ][ÝWØ[ÝÜÊˆ™XÛÜ™Îˆ\ÝÙXÝÜÝ‹[žWWKˆ^[ØYˆXÝÜÝ‹[žWKˆÛÛ™šYÎˆXÝÜÝ‹[žWKŠHOˆ\VØ›ÛÛÝ—N‚ˆÙ][™ÜÈHÛÛ™šYÖÈš[™Ù\Ý[Ûˆ—Bˆ[Y^›Û™HHÝŠÛÛ™šYË™Ù]
[Y^›Û™H‹\ÚXKÕÚÞ[ÈŠJBˆ]HHYØXÞK›ØØ[Ù]J^[ØYÛÛ™šYÊBˆØ[YWÙ^HHÜ™XÛÜ™›Üˆ™XÛÜ™[ˆ™XÛÜ™ÈYˆÜ™XÛÜ™Ù]J™XÛÜ™[Y^›Û™JHOH]WBˆÛÛXÝÜ—ÚYHÝŠ^[ØYÈ˜ÛÛXÝÜ—ÚY—JBˆÛÛXÝÜ—ØÛÝ[HÝ[JÜ™XÛÜ™Ú\×ØÛÛXÝÜŠ™XÛÜ™ÛÛXÝÜ—ÚY
H›Üˆ™XÛÜ™[ˆØ[YWÙ^JBˆÛÛXÝÜ—Û[Z]H[
Ù][™ÜÖÈ›X^ØØ[™Y]\×Ü\—ØÛÛXÝÜ—Ü\—Ù^H—JBˆYˆÛÛXÝÜ—ØÛÝ[HÛÛXÝÜ—Û[Z]‚ˆ™]\›ˆ˜[ÙKˆ˜ÛÛXÝÜˆZ[H][ÝH™XXÚY
ØÛÛXÝÜ—ØÛÝ[KÞØÛÛXÝÜ—Û[Z]JH‚‚ˆÜXÈHÝŠ^[ØYÈœØÜ™Y[š[™È—VÈÜXÈ—JBˆÜX×ØÛÝ[HÝ[JÜXÈ[ˆÜ™XÛÜ™ÝÜXÜÊ™XÛÜ™
H›Üˆ™XÛÜ™[ˆØ[YWÙ^JBˆÜX×Û[Z]H[
Ù][™ÜÖÈ›X^ØØ[™Y]\×Ü\—ÝÜX×Ü\—Ù^H—JBˆYˆÜX×ØÛÝ[HÜX×Û[Z]‚ˆ™]\›ˆ˜[ÙKˆÜXÈZ[H][ÝH™XXÚY
ÝÜX×ØÛÝ[KÞÝÜX×Û[Z]JH›ÜˆÝÜXßH‚‚ˆZ[WÛ[Z]H[
Ù][™ÜÖÈ›X^ØØ[™Y]\×ÝÝ[Ü\—Ù^H—JBˆYˆ[ŠØ[YWÙ^JHHZ[WÛ[Z]‚ˆ™]\›ˆ˜[ÙKˆ™ÛØ˜[Z[H][ÝH™XXÚY
Û[ŠØ[YWÙ^J_KÞÙZ[WÛ[Z]JH‚‚ˆÜ[—ØÛÝ[HÝ[J™XÛÜ™™Ù]
œÝ]HŠHOHš[˜›Þˆ›Üˆ™XÛÜ™[ˆ™XÛÜ™ÊBˆÜ[—Û[Z]H[
Ù][™ÜÖÈ›X^ÛÜ[—ØØ[™Y]\È—JBˆYˆÜ[—ØÛÝ[HÜ[—Û[Z]‚ˆ™]\›ˆ˜[ÙKˆ›Ü[ˆØ[™Y]H˜XÚÛÙÈ[Z]™XXÚY
ÛÜ[—ØÛÝ[KÞÛÜ[—Û[Z]JH‚ˆ™]\›ˆYKˆ‚‚‚™YˆÙš[™ÛØØ[ÛX]Ú
ˆ™XÛÜ™Îˆ]\˜X›VÙXÝÜÝ‹[žWWK^[ØYˆXÝÜÝ‹[žWBŠHOˆXÝÜÝ‹[žWH›Û™N‚ˆ[˜ÛÛZ[™ÈH\\—Ø[X\Ù\Ê^[ØYÈœ\\ˆ—JBˆX]Ú\ÈHÜ™XÛÜ™›Üˆ™XÛÜ™[ˆ™XÛÜ™ÈYˆ[˜ÛÛZ[™È	ˆ\\—Ø[X\Ù\Ê™XÛÜ™Èœ\\ˆ—JWBˆYˆ[ŠX]Ú\ÊHˆN‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆˆ˜Ø[™Y]H[X\Ù\ÈX]Ú][\H™XÛÜ™ÎˆÖÚ][VÉÚY	×H›Üˆ][H[ˆX]Ú\×_H‚ˆ
Bˆ™]\›ˆX]Ú\ÖÌHYˆX]Ú\È[ÙH›Û™B‚‚™Yˆ[X\Ù\×Ú[—ÙÚ]Ü™YœÊˆ™\×Ü›ÛÝˆ]ˆ™Y—Ü™Yš^ˆÝˆ›Û™Kˆ\™Ù]Ù]NˆÝ‹ˆÚ[™Ý×Ù^\Îˆ[ŠHOˆÙ]ÜÝ—N‚ˆYˆ›Ý™Y—Ü™Yš^‚ˆ™]\›ˆÙ]

BˆÝ]]HÝXœ›ØÙ\ÜËœ[ŠˆÈ™Ú]‹™›Ü‹YXXÚ\™Yˆ‹‹KY›Ü›X]IJ™Y›˜[YJH‹™Y—Ü™Yš^KˆÝÙ\™\×Ü›ÛÝˆÚXÚÏUYKˆØ\\™WÛÝ]]UYKˆ^UYKˆ
KœÝÝ]ˆ\™Ù]H]][YKœÝœ[YJ\™Ù]Ù]K‰VKI[KIYŠK™]J
Bˆ[X\Ù\ÎˆÙ]ÜÝ—HHÙ]

Bˆ›Üˆ™Yˆ[ˆ
[™KœÝš\

H›Üˆ[™H[ˆÝ]]œÜ][™\Ê
HYˆ[™KœÝš\

JN‚ˆžN‚ˆœ˜[˜ÚÙ]HH]][YKœÝœ[YJ™Y‹œœÜ]
‹È‹JVËLWK‰VKI[KIYŠK™]J
Bˆ^Ù\˜[YQ\œ›ÜŽ‚ˆÛÛ[YBˆYˆXœÊ
œ˜[˜ÚÙ]HH\™Ù]
K™^\ÊHˆÚ[™Ý×Ù^\Î‚ˆÛÛ[YBˆ]ÈHÝXœ›ØÙ\ÜËœ[ŠˆÈ™Ú]‹›Ë]™YH‹‹\ˆ‹‹K[˜[YK[Û›H‹™Y‹‹KH‹šÛ›ÝÛYÙKÛ]\˜]\™KØØ[™Y]\È—KˆÝÙ\™\×Ü›ÛÝˆÚXÚÏUYKˆØ\\™WÛÝ]]UYKˆ^UYKˆ
KœÝÝ]ˆ›Üˆ][ˆ
[™KœÝš\

H›Üˆ[™H[ˆ]ËœÜ][™\Ê
HYˆ[™K™[™ÝÚ]
‹šœÛÛˆŠJN‚ˆžN‚ˆÛÛ[HÝXœ›ØÙ\ÜËœ[ŠˆÈ™Ú]‹œÚÝÈ‹ˆžÜ™YŸNžÜ]H—KˆÝÙ\™\×Ü›ÛÝˆÚXÚÏUYKˆØ\\™WÛÝ]]UYKˆ^UYKˆ
KœÝÝ]ˆ™XÛÜ™HœÛÛ‹›ØYÊÛÛ[
Bˆ^Ù\
ÝXœ›ØÙ\ÜËØ[Y›ØÙ\ÜÑ\œ›Ü‹œÛÛ‹’”ÓÓ‘XÛÙQ\œ›ÜŠN‚ˆÛÛ[YBˆYˆ\Ú[œÝ[˜ÙJ™XÛÜ™XÝ
H[™\Ú[œÝ[˜ÙJ™XÛÜ™™Ù]
œ\\ˆŠKXÝ
N‚ˆ[X\Ù\Ë\]J\\—Ø[X\Ù\Ê™XÛÜ™Èœ\\ˆ—JJBˆ™]\›ˆ[X\Ù\Â‚‚™Yˆ[™Ù\ÝÛÛ™Jˆ[œ]Ü]ˆ]ˆ™\×Ü›ÛÝˆ]ˆÛÛ™šYÎˆXÝÜÝ‹[žWKˆ^\›˜[Ø[X\Ù\ÎˆÙ]ÜÝ—H›Û™HH›Û™Kˆ^XÝYØÛÛXÝÜŽˆÝˆ›Û™HH›Û™KŠHOˆYØXÞK’[™Ù\Ý™\Ý[‚ˆ^[ØYHYØXÞKœ™XYÚœÛÛŠ[œ]Ü]
Bˆ˜[Y]WÚ\™[™YØØ[™Y]J^[ØY™\×Ü›ÛÝÛÛ™šYË^XÝYØÛÛXÝÜŠBˆ^[ØYHÜØ[š]\ÙWÜ^[ØY
^[ØY
Bˆ™XÛÜ™ÈHYØXÞK˜Ø[™Y]WÜ™XÛÜ™Ê™\×Ü›ÛÝ
BˆX]ÚHÙš[™ÛØØ[ÛX]Ú
™XÛÜ™Ë^[ØY
BˆYˆX]Ú\È›Ý›Û™N‚ˆÝ]]H™\×Ü›ÛÝÈšÛ›ÝÛYÙKÛ]\˜]\™KØØ[™Y]\ÈˆÈˆžÛX]ÚÉÚY	ßKšœÛÛˆ‚ˆÚ[™ÙYHYØXÞK›Y\™ÙWÜ™XÛÜ™
X]Ú^[ØY
BˆYˆÚ[™ÙY‚ˆYØXÞK˜[Y]WÜ™XÛÜ™
X]ÚÝ]]
BˆYØXÞKÜš]WÚœÛÛŠÝ]]X]Ú
Bˆ™]\›ˆYØXÞK’[™Ù\Ý™\Ý[
ÝŠX]ÚÈšY—JK›Y\™ÙY‹Ý]]˜YY[X\Ë[X]ÚY\ØÛÝ™\žHŠBˆ™]\›ˆYØXÞK’[™Ù\Ý™\Ý[
ÝŠX]ÚÈšY—JK››ËXÚ[™ÙH‹Ý]]œØÚY[H[ˆ[™XYH[™Ù\ÝYŠB‚ˆ[X\Ù\ÈH\\—Ø[X\Ù\Ê^[ØYÈœ\\ˆ—JBˆYˆ^\›˜[Ø[X\Ù\È[™[X\Ù\È	ˆ^\›˜[Ø[X\Ù\Î‚ˆ™]\›ˆYØXÞK’[™Ù\Ý™\Ý[
ˆØ[›ÛšXØ[Ü\\—ÚY
^[ØY
Kˆ™\XØ]H‹ˆ›Û™Kˆ˜Ø[™Y]H[X\È[™XYH^\ÝÈÛˆ[›Ý\ˆ˜Y\ˆœ˜[˜Ú‹ˆ
B‚ˆ[ÝÙY™X\ÛÛˆH][ÝWØ[ÝÜÊ™XÛÜ™Ë^[ØYÛÛ™šYÊBˆYˆ›Ý[ÝÙY‚ˆ™]\›ˆYØXÞK’[™Ù\Ý™\Ý[
Ø[›ÛšXØ[Ü\\—ÚY
^[ØY
Kœ][ÝK\™Z™XÝY‹›Û™K™X\ÛÛŠB‚ˆ\\—ÚYHØ[›ÛšXØ[Ü\\—ÚY
^[ØY
BˆÝ]]H™\×Ü›ÛÝÈšÛ›ÝÛYÙKÛ]\˜]\™KØØ[™Y]\ÈˆÈˆžÜ\\—ÚYKšœÛÛˆ‚ˆ™XÛÜ™HYØXÞK›™]×Ü™XÛÜ™
^[ØY\\—ÚY
BˆYØXÞK˜[Y]WÜ™XÛÜ™
™XÛÜ™Ý]]
BˆYØXÞKÜš]WÚœÛÛŠÝ]]™XÛÜ™
Bˆ™]\›ˆYØXÞK’[™Ù\Ý™\Ý[
\\—ÚY˜Ü™X]Y‹Ý]]›™]ÈØ[›ÛšXØ[Ø[™Y]HŠB‚‚™Yˆ\]WÙZ[WÙYÙ\Ý
™\×Ü›ÛÝˆ]]NˆÝ‹ÛÛ™šYÎˆXÝÜÝ‹[žWJHOˆ]‚ˆ[Y^›Û™HHÝŠÛÛ™šYË™Ù]
[Y^›Û™H‹\ÚXKÕÚÞ[ÈŠJBˆ™XÛÜ™ÈHÂˆ™XÛÜ™›Üˆ™XÛÜ™[ˆYØXÞK˜Ø[™Y]WÜ™XÛÜ™Ê™\×Ü›ÛÝ
BˆYˆÜ™XÛÜ™Ù]J™XÛÜ™[Y^›Û™JHOH]BˆBˆ]]ÈH—ˆ‹š›Ú[ŠˆÂˆYØXÞKUU×ÔÕT•ˆˆÈÈ:!ê¹båycãºfá¹`&z(ç‹ˆˆ‹ˆˆ¹`&z(ç9¥lˆ
ŠžÛ[Š™XÛÜ™Ê_JŠˆ‹ˆˆ‹ˆ
›YØXÞK—ØØ[™Y]WÜ›ÝÜÊ™XÛÜ™ÊKˆYØXÞKUU×ÑS‘ˆBˆ
Bˆ]H™\×Ü›ÛÝÈšÛ›ÝÛYÙKÛ]\˜]\™KÙYÙ\ÝÈˆÈˆžÙ]_K›Y‚ˆ™X[X›HHÐQ‘WÑQÑTÕÔ‘PSP“K™›Ü›X]
]OY]JBˆYˆ]™^\ÝÊ
N‚ˆ^H]œ™XYÝ^
[˜ÛÙ[™ÏH]‹NŠBˆYˆ^˜ÛÝ[
YØXÞKUU×ÔÕT•
HOHHÜˆ^˜ÛÝ[
YØXÞKUU×ÑS‘
HOHN‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆžÜ]NˆYÙ\ÝX\šÙ\œÈ]\ÝØØÝ\ˆ^XÝHÛ˜ÙHŠBˆ™Yš^™[XZ[™\ˆH^œÜ]
YØXÞKUU×ÔÕT•JBˆËÝY™š^H™[XZ[™\‹œÜ]
YØXÞKUU×ÑS‘JBˆYˆ™Yš^œœÝš\

K™[™ÝÚ]
˜ŠHÜˆ¸àdøàk¹¥éy«(xàà8à©8à®8à©øà®xàâ8àkˆˆ[ˆ™Yš^‚ˆ™Yš^H™X[X›Bˆ^HˆžÜ™Yš^^Ø]]ß^ÜÝY™š^H‚ˆ[ÙN‚ˆ^HˆžÜ™X[X›_^Ø]]ßW—ˆÈÈ9¥éy«(xàë8àäøàéxàï—¹§+8àë8àäøàéxàï8à ‚—ˆ‚ˆ]œ\™[›ZÙ\Š\™[ÏUYK^\ÝÛÚÏUYJBˆ]Üš]WÝ^
^[˜ÛÙ[™ÏH]‹NŠBˆ™]\›ˆ]‚‚™YˆÜ][ÝWÙ[žJXØÙ\Yˆ[[Z]ˆ[
HOˆXÝÜÝ‹[N‚ˆ™]\›ˆÈ˜XØÙ\YŽˆXØÙ\Y›[Z]Žˆ[Z]œ™[XZ[š[™ÈŽˆX^
[Z]HXØÙ\Y
_B‚‚™Yˆ\]WÙZ[WÜÝ]\Ê™\×Ü›ÛÝˆ]]NˆÝ‹ÛÛ™šYÎˆXÝÜÝ‹[žWJHOˆ]‚ˆ[Y^›Û™HHÝŠÛÛ™šYË™Ù]
[Y^›Û™H‹\ÚXKÕÚÞ[ÈŠJBˆ™XÛÜ™ÈHYØXÞK˜Ø[™Y]WÜ™XÛÜ™Ê™\×Ü›ÛÝ
BˆØ[YWÙ^HHÜ™XÛÜ™›Üˆ™XÛÜ™[ˆ™XÛÜ™ÈYˆÜ™XÛÜ™Ù]J™XÛÜ™[Y^›Û™JHOH]WBˆÙ][™ÜÈHÛÛ™šYÖÈš[™Ù\Ý[Ûˆ—BˆÛÛXÝÜ—Û[Z]H[
Ù][™ÜÖÈ›X^ØØ[™Y]\×Ü\—ØÛÛXÝÜ—Ü\—Ù^H—JBˆÜX×Û[Z]H[
Ù][™ÜÖÈ›X^ØØ[™Y]\×Ü\—ÝÜX×Ü\—Ù^H—JBˆÛÛXÝÜœÈHÂˆÛÛXÝÜ—ÚYˆÜ][ÝWÙ[žJˆÝ[JÜ™XÛÜ™Ú\×ØÛÛXÝÜŠ™XÛÜ™ÛÛXÝÜ—ÚY
H›Üˆ™XÛÜ™[ˆØ[YWÙ^JKˆÛÛXÝÜ—Û[Z]ˆ
Bˆ›ÜˆÛÛXÝÜ—ÚY[ˆÛÛ™šYÖÈ˜ÛÛXÝÜœÈ—BˆBˆÛÛ™šYÝ\™YÝÜXÜÈHÛÜY
ˆÝÜXÈ›ÜˆÛÛXÝÜˆ[ˆÛÛ™šYÖÈ˜ÛÛXÝÜœÈ—K˜[Y\Ê
H›ÜˆÜXÈ[ˆÛÛXÝÜ‹™Ù]
ÜXÜÈ‹×J_Bˆ
BˆÜXÜÈHÂˆÜXÎˆÜ][ÝWÙ[žJÝ[JÜXÈ[ˆÜ™XÛÜ™ÝÜXÜÊ™XÛÜ™
H›Üˆ™XÛÜ™[ˆØ[YWÙ^JKÜX×Û[Z]
Bˆ›ÜˆÜXÈ[ˆÛÛ™šYÝ\™YÝÜXÜÂˆBˆ]H™\×Ü›ÛÝÈšÛ›ÝÛYÙKÛ]\˜]\™KÜÝ]\ÈˆÈˆžÙ]_KšœÛÛˆ‚ˆ^\Ý[™ÈHYØXÞKœ™XYÚœÛÛŠ]
HYˆ]™^\ÝÊ
H[ÙHßBˆÛÜ™HHÂˆ˜XØÙ\YØØ[™Y]\ÈŽˆ[ŠØ[YWÙ^JKˆ™Z[WÛ[Z]Žˆ[
Ù][™ÜÖÈ›X^ØØ[™Y]\×ÝÝ[Ü\—Ù^H—JKˆ›Ü[—ØØ[™Y]\ÈŽˆÝ[J™XÛÜ™™Ù]
œÝ]HŠHOHš[˜›Þˆ›Üˆ™XÛÜ™[ˆ™XÛÜ™ÊKˆ›Ü[—Û[Z]Žˆ[
Ù][™ÜÖÈ›X^ÛÜ[—ØØ[™Y]\È—JKˆ˜ÛÛXÝÜœÈŽˆÛÛXÝÜœËˆÜXÜÈŽˆÜXÜËˆBˆ™]š[Ý\×ØÛÜ™HH^\Ý[™Ë™Ù]
š[™Ù\Ý[ÛˆŠBˆÙ[™\˜]YØ]H^\Ý[™Ë™Ù]
™Ù[™\˜]YØ]ŠBˆYˆ™]š[Ý\×ØÛÜ™HOHÛÜ™HÜˆ›Ý\Ú[œÝ[˜ÙJÙ[™\˜]YØ]ÝŠN‚ˆÙ[™\˜]YØ]H]][YK››ÝÊ›Û™R[™›Ê[Y^›Û™JJKš\ÛÙ›Ü›X]
[Y\ÜXÏHœÙXÛÛ™ÈŠBˆÝ]\ÈHÂˆœØÚ[XWÝ™\œÚ[ÛˆŽˆKˆ™]HŽˆ]Kˆš[š]X[^™YØ]Žˆ^\Ý[™Ë™Ù]
š[š]X[^™YØ]ŠKˆ›\ÝØÝ\˜]YØ]Žˆ^\Ý[™Ë™Ù]
›\ÝØÝ\˜]YØ]ŠKˆ™Ù[™\˜]YØ]ŽˆÙ[™\˜]YØ]ˆš[™Ù\Ý[ÛˆŽˆÛÜ™KˆBˆ]œ\™[›ZÙ\Š\™[ÏUYK^\ÝÛÚÏUYJBˆYØXÞKÜš]WÚœÛÛŠ]Ý]\ÊBˆ™]\›ˆ]‚‚™Yˆ˜[Y]WÜ™\ÜÚ]ÜžJ™\×Ü›ÛÝˆ]
HOˆ\ÝÜÝ—N‚ˆ\œ›ÜœÈHYØXÞK˜[Y]WÜ™\ÜÚ]ÜžJ™\×Ü›ÛÝ
Bˆ[X\Ù\ÎˆXÝÜÝ‹Ý—HHßBˆ›Üˆ™XÛÜ™[ˆYØXÞK˜Ø[™Y]WÜ™XÛÜ™Ê™\×Ü›ÛÝ
N‚ˆ›Üˆ[X\È[ˆ\\—Ø[X\Ù\Ê™XÛÜ™Èœ\\ˆ—JN‚ˆ™]š[Ý\ÈH[X\Ù\Ë™Ù]
[X\ÊBˆYˆ™]š[Ý\È[™™]š[Ý\ÈOH™XÛÜ™ÈšY—N‚ˆ\œ›ÜœË˜\[™
ˆ™\XØ]H]\˜]\™H[X\ÈØ[X\ßNˆÜ™]š[Ý\ßKÜ™XÛÜ™ÉÚY	×_HŠBˆ[X\Ù\ÖØ[X\×HHÝŠ™XÛÜ™ÈšY—JBˆ›Üˆ][ˆÛÜY

™\×Ü›ÛÝÈšÛ›ÝÛYÙKÛ]\˜]\™KÙYÙ\ÝÈŠK™ÛØŠÏÏÏËOÏËOÏË›YŠJN‚ˆ^H]œ™XYÝ^
[˜ÛÙ[™ÏH]‹NŠBˆYˆ^˜ÛÝ[
YØXÞKUU×ÔÕT•
HOHHÜˆ^˜ÛÝ[
YØXÞKUU×ÑS‘
HOHN‚ˆ\œ›ÜœË˜\[™
ˆžÜ]NˆYÙ\ÝX\šÙ\œÈ]\ÝØØÝ\ˆ^XÝHÛ˜ÙHŠBˆ™]\›ˆ\œ›ÜœÂ‚‚™YˆZ[Ü\œÙ\Š
HOˆ\™Ü\œÙK\™Ý[Y[\œÙ\Ž‚ˆ\œÙ\ˆH\™Ü\œÙK\™Ý[Y[\œÙ\Š\ØÜš\[ÛW×ÙØ××ÊBˆÝXœ\œÙ\œÈH\œÙ\‹˜YÜÝXœ\œÙ\œÊ\ÝH˜ÛÛ[X[™‹™\]Z\™YUYJBˆ]WÜ\œÙ\ˆHÝXœ\œÙ\œË˜YÜ\œÙ\Š™]HŠBˆ]WÜ\œÙ\‹˜YØ\™Ý[Y[
š[œ]‹\OT]
Bˆ]WÜ\œÙ\‹˜YØ\™Ý[Y[
‹K\™\Ë\›ÛÝ‹\OT]Y˜][T]˜ÝÙ

JBˆ]WÜ\œÙ\‹˜YØ\™Ý[Y[
‹KY^XÝYXÛÛXÝÜˆŠBˆ[™Ù\ÝÜ\œÙ\ˆHÝXœ\œÙ\œË˜YÜ\œÙ\Šš[™Ù\ÝŠBˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
š[œ]‹˜\™ÜÏHŠÈ‹\OT]
Bˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
‹K\™\Ë\›ÛÝ‹\OT]Y˜][T]˜ÝÙ

JBˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
‹KYY\\™Y‹\™Yš^ŠBˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
‹KY^XÝYXÛÛXÝÜˆŠBˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
‹K]\]KYYÙ\Ý‹XÝ[ÛHœÝÜ™WÝYHŠBˆ[™Ù\ÝÜ\œÙ\‹˜YØ\™Ý[Y[
‹K]\]K\Ý]\È‹XÝ[ÛHœÝÜ™WÝYHŠBˆ˜[Y]WÜ\œÙ\ˆHÝXœ\œÙ\œË˜YÜ\œÙ\Š˜[Y]HŠBˆ˜[Y]WÜ\œÙ\‹˜YØ\™Ý[Y[
‹K\™\Ë\›ÛÝ‹\OT]Y˜][T]˜ÝÙ

JBˆ™]\›ˆ\œÙ\‚‚‚™YˆXZ[Š
HOˆ[‚ˆ\™ÜÈHZ[Ü\œÙ\Š
Kœ\œÙWØ\™ÜÊ
Bˆ™\×Ü›ÛÝH\™ÜËœ™\×Ü›ÛÝœ™\ÛÛ™J
BˆžN‚ˆÛÛ™šYÈHYØXÞK›ØYØÛÛ™šYÊ™\×Ü›ÛÝ
BˆYˆ\™ÜË˜ÛÛ[X[™OH™]HŽ‚ˆ^[ØYHYØXÞKœ™XYÚœÛÛŠ\™ÜËš[œ]
Bˆ˜[Y]WÚ\™[™YØØ[™Y]J^[ØY™\×Ü›ÛÝÛÛ™šYË\™ÜË™^XÝYØÛÛXÝÜŠBˆš[
YØXÞK›ØØ[Ù]J^[ØYÛÛ™šYÊJBˆ™]\›ˆˆYˆ\™ÜË˜ÛÛ[X[™OH˜[Y]HŽ‚ˆ\œ›ÜœÈH˜[Y]WÜ™\ÜÚ]ÜžJ™\×Ü›ÛÝ
Bˆ›Üˆ\œ›Üˆ[ˆ\œ›ÜœÎ‚ˆš[
ˆ‘T”“ÔŽˆÙ\œ›ÜŸHŠBˆš[
ˆžÛ[Š\œ›ÜœÊ_H]\˜]\™H˜Y\ˆ\œ›ÜŠÊKˆŠBˆ™]\›ˆHYˆ\œ›ÜœÈ[ÙH‚ˆX^[][HH[
ÛÛ™šYÖÈš[™Ù\Ý[Ûˆ—VÈ›X^ØØ[™Y]\×Ü\—ÚÝ\›WÜ[ˆ—JBˆYˆ[Š\™ÜËš[œ]
HˆX^[][N‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠˆˆšÝ\›H[ˆÝ\YYÛ[Š\™ÜËš[œ]
_HØ[™Y]\ÎÈX^[][H\ÈÛX^[][_H‚ˆ
Bˆ^[ØYÎˆ\ÝÝ\VÔ]XÝÜÝ‹[žWWWHH×Bˆ]\ÎˆÙ]ÜÝ—HHÙ]

Bˆ›Üˆ[œ]Ü][ˆ\™ÜËš[œ]‚ˆ^[ØYHYØXÞKœ™XYÚœÛÛŠ[œ]Ü]
Bˆ˜[Y]WÚ\™[™YØØ[™Y]J^[ØY™\×Ü›ÛÝÛÛ™šYË\™ÜË™^XÝYØÛÛXÝÜŠBˆ^[ØYË˜\[™

[œ]Ü]^[ØY
JBˆ]\Ë˜Y
YØXÞK›ØØ[Ù]J^[ØYÛÛ™šYÊJBˆYˆ[Š]\ÊHOHN‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŽˆ›Û™H[™Ù\Ý[›ØØ][Ûˆ™\]Z\™\ÈÛ™H”Õ]NˆÜÛÜY
]\Ê_HŠBˆ[™Ù\ÝÙ]HH™^
]\Š]\ÊJBˆ^\›˜[Ø[X\Ù\ÈH[X\Ù\×Ú[—ÙÚ]Ü™YœÊˆ™\×Ü›ÛÝˆ\™ÜË™Y\Ü™Y—Ü™Yš^ˆ[™Ù\ÝÙ]Kˆ[
ÛÛ™šYÖÈš[™Ù\Ý[Ûˆ—VÈ™Y\Øœ˜[˜ÚÝÚ[™Ý×Ù^\È—JKˆ
Bˆ›Üˆ[œ]Ü]È[ˆ^[ØYÎ‚ˆ™\Ý[H[™Ù\ÝÛÛ™Jˆ[œ]Ü]ˆ™\×Ü›ÛÝˆÛÛ™šYËˆ^\›˜[Ø[X\Ù\Ëˆ\™ÜË™^XÝYØÛÛXÝÜ‹ˆ
Bˆš[
ˆžÜ™\Ý[˜XÝ[ÛŸNˆÜ™\Ý[œ\\—ÚYH8 %Ü™\Ý[›Y\ÜØYÙ_HŠBˆYˆ\™ÜË\]WÙYÙ\Ý‚ˆš[
ˆ\]YYÙ\ÝˆÝ\]WÙZ[WÙYÙ\Ý
™\×Ü›ÛÝ[™Ù\ÝÙ]KÛÛ™šYÊ_HŠBˆYˆ\™ÜË\]WÜÝ]\Î‚ˆš[
ˆ\]YÝ]\ÎˆÝ\]WÙZ[WÜÝ]\Ê™\×Ü›ÛÝ[™Ù\ÝÙ]KÛÛ™šYÊ_HŠBˆ\œ›ÜœÈH˜[Y]WÜ™\ÜÚ]ÜžJ™\×Ü›ÛÝ
BˆYˆ\œ›ÜœÎ‚ˆ˜Z\ÙHYØXÞKØ[™Y]Q\œ›ÜŠŽÈ‹š›Ú[Š\œ›ÜœÊJBˆ™]\›ˆˆ^Ù\
YØXÞKØ[™Y]Q\œ›Ü‹ÝXœ›ØÙ\ÜËØ[Y›ØÙ\ÜÑ\œ›ÜŠH\È^Î‚ˆš[
ˆ‘T”“ÔŽˆÙ^ßHŠBˆ™]\›ˆB‚‚šYˆ×Û˜[YW×ÈOH—×ÛXZ[—×ÈŽ‚ˆ˜Z\ÙHÞ\Ý[Q^]
XZ[Š
JB