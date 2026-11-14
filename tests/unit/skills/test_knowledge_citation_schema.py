"""Small, network-free value and document contracts shared by citation consumers."""
from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/knowledge/citation_map"


@pytest.fixture(scope="module")
def api() -> ModuleType:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / ".agents/skills/knowledge-control/scripts"))
        return importlib.import_module("kg_citations")


@pytest.fixture(scope="module")
def identifiers(api: ModuleType) -> ModuleType:
    return importlib.import_module("kg_identifiers")


@pytest.fixture(scope="module")
def paper_data() -> dict[str, Any]:
    for path in (FIXTURE / "papers").glob("*.md"):
        data = yaml.safe_load(path.read_text().split("---", 2)[1])
        if data["title"] == "Fixture A":
            return data  # type: ignore[no-any-return]
    raise AssertionError("Fixture A is missing")


def markdown(data: dict[str, Any]) -> str:
    return "---\n" + yaml.safe_dump(data, allow_unicode=True) + "---\nAdditional reading note.\n"


@pytest.mark.parametrize(("namespace", "value", "expected"), [
    ("doi", " https://doi.org/10.5555/FiXtUrE%2DB ", "10.5555/fixture-b"),
    ("doi", "DOI:10.5555/FIXTURE-B", "10.5555/fixture-b"),
    ("arxiv", "https://arxiv.org/pdf/2401.00002v12.pdf", "2401.00002"),
    ("arxiv", "arXiv:hep-th/9901001v2", "hep-th/9901001"),
    ("openalex", "https://openalex.org/w1000002", "W1000002"),
    ("url", "HTTPS://EXAMPLE.ORG/a?version=2#section", "https://example.org/a?version=2"),
])
def test_identifier_aliases_have_explicit_normal_forms(identifiers: ModuleType, namespace: str, value: str, expected: str) -> None:
    assert identifiers.normalize_identifier(namespace, value) == expected


@pytest.mark.parametrize(("field_path", "value", "message"), [
    (("schema_version",), 2, "schema_version"),
    (("schema_version",), True, "schema_version"),
    (("type",), "paper", "type must"),
    (("extra",), "ignored?", "expected keys"),
    (("title",), " ", "title"),
    (("year",), True, "year"),
    (("authors",), "Author", "authors"),
    (("authors",), ["Author", "Author"], "duplicate"),
    (("external_ids",), {}, "external_ids"),
    (("external_ids",), {"isbn": "unknown"}, "invalid external ID"),
    (("external_ids",), {"doi": "10.5555/FIXTURE-A"}, "normalized"),
    (("identity_key",), "doi:10.5555/other", "identity_key"),
    (("source",), "javascript:alert(1)", "source"),
    (("summary",), "", "summary"),
    (("text_status",), "downloaded", "text_status"),
    (("references", 0, "roles"), ["baseline"], "roles"),
    (("references", 0, "roles"), ["comparison", "comparison"], "duplicate"),
    (("references", 0, "roles"), [], "requires roles and evidence"),
    (("references", 0, "evidence"), [], "requires roles and evidence"),
    (("references", 0, "reason"), "", "reason"),
    (("references", 0, "identification_status"), "unknown", "identification_status"),
    (("references", 0, "identification_status"), "unresolved", "disagree"),
    (("references", 0, "decision_status"), "guessed", "decision_status"),
    (("references", 0, "decision_status"), "unclassifiable", "empty roles"),
    (("references", 1, "ref_id"), "ref-1", "duplicate reference"),
])
def test_malformed_paper_is_rejected(api: ModuleType, paper_data: dict[str, Any], field_path: tuple[Any, ...], value: Any, message: str) -> None:
    data = deepcopy(paper_data)
    cursor = data
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = value
    with pytest.raises(ValueError, match=message):
        api.parse_document(markdown(data))


def test_yaml_duplicate_keys_and_aliases_are_rejected(api: ModuleType, paper_data: dict[str, Any]) -> None:
    raw = markdown(paper_data)
    with pytest.raises(ValueError, match="duplicate YAML key"):
        api.parse_document(raw.replace("type: citation_paper", "type: citation_paper\ntype: citation_paper"))
    with pytest.raises(ValueError, match="aliases"):
        api.parse_document(raw.replace("title: Fixture A", "title: &title Fixture A\nextra: *title"))


def test_fixture_has_independent_failure_states_and_all_roles(api: ModuleType, tmp_path: Path) -> None:
    library = api.load_citations(FIXTURE)
    assert api.validate_library(library) == []
    papers = {p.title[-1]: p for p in library.papers.values()}
    assert set(papers) == set("ABCDE")
    assert papers["D"].text_status == "not_requested" and papers["D"].summary == ""
    assert papers["E"].text_status == "failed" and "403" in papers["E"].text_reason
    a = papers["A"]
    assert a.references[3].identification_status == "unresolved"
    assert a.references[4].identification_status == "resolved"
    assert a.references[3].decision_status == a.references[4].decision_status == "unclassifiable"
    assert {r for ref in a.references for r in ref.roles} == {"foundation", "component", "comparison", "related", "benchmark"}
    comparisons: dict[str, set[str]] = {}
    for paper in library.papers.values():
        for ref in paper.references:
            if "comparison" in ref.roles and ref.target_id:
                comparisons.setdefault(ref.target_id, set()).add(paper.title[-1])
    assert comparisons == {papers["B"].id: {"A"}, papers["D"].id: {"B", "C"}, papers["E"].id: {"A"}}
    for suffix, expected in (("comparison-depth2", [set("A"), set("BE"), set("D")]), ("comparison-depth1", [set("A"), set("BE")]), ("mixed-depth2", [set("A"), set("BCE"), set("D")])):
        search = library.searches["search-fixture-" + suffix]
        labels = {p.id: p.title[-1] for p in papers.values()}
        assert [{labels[x] for x in g.paper_ids} for g in search.generations] == expected
    # Actual disk write/read boundary; independent content assertions guard against
    # a serializer/parser pair dropping fields together.
    a.body = "\nA separate reading note.\n"
    written = api.write_document(a, tmp_path)
    raw = written.read_text()
    data = yaml.safe_load(raw.split("---", 2)[1])
    assert data["references"][0]["roles"] == ["comparison", "foundation"]
    assert data["external_ids"] == {"doi": "10.5555/fixture-a"}
    assert data["references"][3]["target_id"] is None
    assert "A separate reading note." in raw
    assert api.load_citations(tmp_path).papers[a.id] == a


def test_alias_addition_keeps_id_and_ambiguous_union_is_rejected(api: ModuleType, identifiers: ModuleType) -> None:
    library = api.load_citations(FIXTURE)
    paper = next(p for p in library.papers.values() if p.title == "Fixture B")
    assert paper.id == "cite-15ff6ca63fce212e"
    paper.external_ids["url"] = "https://example.org/other-b"
    parsed = api.parse_document(api.serialize_document(paper))
    assert parsed.id == paper.id and parsed.identity_key == "doi:10.5555/fixture-b"
    index = identifiers.external_id_index((p.id, p.external_ids) for p in library.papers.values())
    assert identifiers.resolve_identity({"arxiv": "2401.00002v3"}, index) == paper.id
    with pytest.raises(ValueError, match="ambiguous literature aliases"):
        identifiers.resolve_identity({"arxiv": "2401.00002", "doi": "10.5555/fixture-c"}, index)


@pytest.mark.parametrize(("key", "value", "message"), [
    ("roles", ["baseline"], "roles"),
    ("roles", [], "nonempty"),
    ("max_depth", -1, "nonnegative integer"),
    ("max_depth", True, "nonnegative integer"),
    ("created_at", "2026-11-14T00:00:00", "timezone"),
    ("root_id", "../escape", "citation paper id"),
])
def test_search_schema_rejects_invalid_conditions(api: ModuleType, key: str, value: Any, message: str) -> None:
    path = FIXTURE / "searches/search-fixture-comparison-depth2.md"
    data = yaml.safe_load(path.read_text().split("---", 2)[1])
    data[key] = value
    with pytest.raises(ValueError, match=message):
        api.parse_document(markdown(data))


def test_full_text_failure_requires_a_reason_and_no_invented_references(api: ModuleType, paper_data: dict[str, Any]) -> None:
    data = deepcopy(paper_data)
    data.update(text_status="failed", text_reason="")
    with pytest.raises(ValueError, match="text_reason"):
        api.parse_document(markdown(data))
    data["text_reason"] = "HTTP 403"
    with pytest.raises(ValueError, match="cannot have references"):
        api.parse_document(markdown(data))


def test_papers_join_rejects_aliases_that_bridge_multiple_documents(api: ModuleType) -> None:
    node = importlib.import_module("kg_lib").Node
    library = api.load_citations(FIXTURE)
    registered = [node("paper-2024-one", "paper", {"external_ids": {"doi": "10.5555/fixture-b"}}, "", Path()),
                  node("paper-2024-two", "paper", {"external_ids": {"arxiv": "2401.00002"}}, "", Path())]
    assert any("ambiguous literature aliases" in e for e in api.validate_library(library, registered))
    registered = [node("paper-2024-one", "paper", {"external_ids": {"doi": "10.5555/fixture-c", "arxiv": "2401.00002"}}, "", Path())]
    assert any("duplicate literature through Papers" in e for e in api.validate_library(library, registered))


@pytest.mark.parametrize("mutation", ["target", "aliases", "generation", "expanded", "depth", "missing-generation", "issues", "unread-expanded", "root", "duplicate-id"])
def test_library_rejects_cross_document_inconsistency(api: ModuleType, mutation: str) -> None:
    library = api.load_citations(FIXTURE)
    by_name = {p.title[-1]: p for p in library.papers.values()}
    search = library.searches["search-fixture-comparison-depth2"]
    if mutation == "target":
        by_name["A"].references[0].target_id = "cite-0000000000000000"
    elif mutation == "aliases":
        by_name["C"].external_ids["arxiv"] = "2401.00002"
    elif mutation == "generation":
        search.generations[1].paper_ids.append(by_name["C"].id)
    elif mutation == "expanded":
        search.generations[2].expanded_ids.append(by_name["D"].id)
    elif mutation == "depth":
        search.generations[1].depth = 2
    elif mutation == "missing-generation":
        search.generations.pop()
    elif mutation == "issues":
        search.issues.pop()
    elif mutation == "unread-expanded":
        by_name["B"].text_status = "not_requested"
    elif mutation == "root":
        search.root_id = "cite-0000000000000000"
    else:
        by_name["C"].id = by_name["B"].id
    assert api.validate_library(library), mutation
