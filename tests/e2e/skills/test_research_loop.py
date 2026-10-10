"""Validate the research-loop skill package and its literature helper offline.

All HTTP goes through an injected fake fetcher; urlopen is patched to fail so a
missed injection cannot reach the network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / ".agents/skills/research-loop"
TEMPLATES = ROOT / ".github/ISSUE_TEMPLATE"
ARIS_SKILLS = {
    "ablation-planner",
    "analyze-results",
    "arxiv",
    "auto-review-loop",
    "experiment-audit",
    "experiment-bridge",
    "experiment-plan",
    "experiment-queue",
    "idea-creator",
    "idea-discovery",
    "monitor-experiment",
    "novelty-check",
    "openalex",
    "render-html",
    "research-lit",
    "research-pipeline",
    "research-refine",
    "research-refine-pipeline",
    "research-review",
    "result-to-claim",
    "run-experiment",
    "semantic-scholar",
    "system-profile",
    "training-check",
    "web-debug-search",
    "shared-references",
}


def frontmatter(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def ls() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lit_search", SKILL / "scripts/lit_search.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["lit_search"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)


class FakeHTTP:
    """Route URLs by prefix to queued responses (bytes or exceptions)."""

    def __init__(self, routes: dict[str, list[bytes | Exception]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> bytes:
        self.calls.append((url, headers))
        for prefix, responses in self.routes.items():
            if url.startswith(prefix):
                item = responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(f"unexpected URL {url}")


def http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(b""))  # type: ignore[arg-type]


ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2008.04524v2</id>
    <published>2020-08-11T00:00:00Z</published>
    <title>Vid2Player: Controllable Video
      Sprites</title>
    <summary>  Tennis   sprites. </summary>
    <author><name>Haotian Zhang</name></author>
    <author><name>Kayvon Fatahalian</name></author>
    <arxiv:doi>10.1145/3446015</arxiv:doi>
    <arxiv:journal_ref>ACM TOG 2021</arxiv:journal_ref>
  </entry>
</feed>"""

S2_JSON = json.dumps(
    {
        "total": 1,
        "data": [
            {
                "paperId": "abc123",
                "title": "Ball tracking",
                "abstract": None,
                "year": 2022,
                "venue": "CVPR",
                "url": "https://www.semanticscholar.org/paper/abc123",
                "openAccessPdf": {"url": "https://example.org/a.pdf"},
                "authors": [{"authorId": "1", "name": "A. Author"}],
                "externalIds": {"DOI": "10.1/x", "ArXiv": "2201.00001"},
                "citationCount": 7,
            }
        ],
    }
).encode()

OPENALEX_JSON = json.dumps(
    {
        "results": [
            {
                "id": "https://openalex.org/W42",
                "display_name": "Court | detection",
                "publication_year": 2019,
                "doi": "https://doi.org/10.2/y",
                "cited_by_count": 3,
                "authorships": [{"author": {"display_name": "B. Author"}}],
                "primary_location": {"source": None},
                "open_access": {"oa_url": None},
                "ids": {"arxiv": "https://arxiv.org/abs/1901.00002"},
                "abstract_inverted_index": {"world": [1], "hello": [0]},
            }
        ]
    }
).encode()


# --- package structure -------------------------------------------------------


def test_skill_frontmatter_is_short_and_body_stays_small() -> None:
    meta = frontmatter(SKILL / "SKILL.md")
    assert meta["name"] == SKILL.name == "research-loop"
    assert (
        isinstance(meta["description"], str) and 40 <= len(meta["description"]) <= 200
    )
    assert len((SKILL / "SKILL.md").read_text(encoding="utf-8").splitlines()) <= 60


def test_local_links_resolve() -> None:
    documents = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
    documents += [
        TEMPLATES / "04_research_theme.md",
        TEMPLATES / "05_research_cycle.md",
    ]
    checked = 0
    for doc in documents:
        for link in re.findall(
            r"\[[^]]+\]\(([^)]+)\)", doc.read_text(encoding="utf-8")
        ):
            if "://" in link:
                continue
            assert (doc.parent / link.split("#")[0]).exists(), (doc, link)
            checked += 1
    assert checked >= 10


def test_every_reference_is_reachable_from_skill() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    for ref in (SKILL / "references").glob("*.md"):
        assert f"references/{ref.name}" in text, ref


def test_no_aris_skill_enters_the_default_skill_directory() -> None:
    installed = {p.name for p in (ROOT / ".agents/skills").iterdir() if p.is_dir()}
    assert "research-loop" in installed
    assert not installed & ARIS_SKILLS
    assert not (ROOT / ".agents/aris").exists()


def test_aris_derived_parts_carry_mit_notice_and_source_revision() -> None:
    notice = (SKILL / "LICENSE.ARIS.txt").read_text(encoding="utf-8")
    assert "MIT License" in notice and "Copyright (c) 2026 wanshuiyin" in notice
    assert (
        "3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d" in notice and "57ae07631" in notice
    )
    for derived in (
        "scripts/lit_search.py",
        "references/evidence-checklist.md",
        "references/literature.md",
    ):
        assert derived in notice
        assert "ARIS" in (SKILL / derived).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "headings"),
    [
        (
            "04_research_theme.md",
            [
                "問題提起",
                "目標指標",
                "評価条件",
                "停止条件",
                "候補手法",
                "予算の目安",
                "合意",
            ],
        ),
        (
            "05_research_cycle.md",
            [
                "調査",
                "選定理由",
                "実験条件",
                "結果（knowledgeノード）",
                "考察",
                "停止条件の判定",
                "次サイクル提案",
                "skill改善メモ",
            ],
        ),
    ],
)
def test_issue_templates_have_required_sections(name: str, headings: list[str]) -> None:
    path = TEMPLATES / name
    meta = frontmatter(path)
    assert meta["name"] and meta["about"] and meta["title"].startswith("[研究")
    assert meta["labels"] == ["research"]
    found = re.findall(r"^## (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert found == headings


# --- literature helper -------------------------------------------------------


def test_arxiv_parse_and_query_construction(ls: ModuleType) -> None:
    fake = FakeHTTP({ls.ARXIV_API: [ARXIV_FEED]})
    [paper] = ls.search(
        "ball trajectory", ["arxiv"], 5, ls.YearRange.parse("2020-2021"), fake
    )
    assert paper.id == paper.arxiv_id == "2008.04524"
    assert paper.title == "Vid2Player: Controllable Video Sprites"
    assert paper.abstract == "Tennis sprites."
    assert paper.authors == ["Haotian Zhang", "Kayvon Fatahalian"]
    assert (paper.year, paper.doi, paper.venue) == (
        2020,
        "10.1145/3446015",
        "ACM TOG 2021",
    )
    url, headers = fake.calls[0]
    assert "tennis-lab-research-loop" in headers["User-Agent"]
    query = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
    assert "all%3Aball+AND+all%3Atrajectory" in query["search_query"]
    assert "submittedDate%3A%5B202001010000+TO+202112312359%5D" in query["search_query"]
    assert (
        ls.arxiv_query('ti:"structure from motion"', None)
        == 'ti:"structure from motion"'
    )
    assert (
        ls.arxiv_query('"structure from motion" tennis', None)
        == 'all:"structure from motion" AND all:tennis'
    )


def test_arxiv_id_lookup_and_api_error(ls: ModuleType) -> None:
    fake = FakeHTTP({ls.ARXIV_API: [ARXIV_FEED]})
    ls.search("id:2008.04524v2", ["arxiv"], 3, None, fake)
    assert "id_list=2008.04524v2" in fake.calls[0][0]
    with pytest.raises(ValueError, match="arXiv ID"):
        ls.search("2008.04524", ["arxiv"], 3, ls.YearRange.parse("2020"), fake)
    error_feed = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#x</id>'
        b"<title>Error</title><summary>malformed query</summary></entry></feed>"
    )
    with pytest.raises(ls.LitSearchError, match="malformed query"):
        ls.search("x", ["arxiv"], 3, None, FakeHTTP({ls.ARXIV_API: [error_feed]}))
    with pytest.raises(ls.LitSearchError, match="rate limited"):
        ls.search(
            "x", ["arxiv"], 3, None, FakeHTTP({ls.ARXIV_API: [b"Rate exceeded.\n"]})
        )


def test_semantic_scholar_parse_and_api_key(
    ls: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret")
    fake = FakeHTTP({ls.S2_API: [S2_JSON]})
    [paper] = ls.search("ball", ["s2"], 2, ls.YearRange.parse("2020-"), fake)
    assert (paper.source, paper.id, paper.doi, paper.arxiv_id, paper.citations) == (
        "s2",
        "abc123",
        "10.1/x",
        "2201.00001",
        7,
    )
    assert paper.pdf_url == "https://example.org/a.pdf" and paper.abstract is None
    url, headers = fake.calls[0]
    assert headers["x-api-key"] == "secret" and "year=2020-" in url and "limit=2" in url


def test_openalex_parse_filters_and_markdown(
    ls: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_EMAIL", "lab@example.org")
    fake = FakeHTTP({ls.OPENALEX_API: [OPENALEX_JSON]})
    papers = ls.search("court", ["openalex"], 1, ls.YearRange.parse("-2019"), fake)
    [paper] = papers
    assert (paper.id, paper.doi, paper.arxiv_id, paper.venue) == (
        "W42",
        "10.2/y",
        "1901.00002",
        None,
    )
    assert paper.abstract == "hello world"
    url = fake.calls[0][0]
    assert (
        "to_publication_date%3A2019-12-31" in url and "from_publication_date" not in url
    )
    assert "mailto=lab%40example.org" in url
    table = ls.to_markdown(papers)
    assert "Court \\| detection" in table and table.count("\n") == 2


def test_retry_is_bounded_and_logged(
    ls: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    sleeps: list[float] = []
    fake = FakeHTTP({ls.S2_API: [http_error(ls.S2_API, 429), S2_JSON]})
    body = ls.get(ls.S2_API, source="s2", fetch=fake, sleep=sleeps.append)
    assert body == S2_JSON and sleeps == [3.0]
    assert "HTTP 429; retry 1/2" in capsys.readouterr().err
    always = FakeHTTP({ls.S2_API: [http_error(ls.S2_API, 503)] * 3})
    with pytest.raises(ls.LitSearchError, match="HTTP 503"):
        ls.get(ls.S2_API, source="s2", fetch=always, sleep=sleeps.append)
    assert len(always.calls) == 3
    not_retried = FakeHTTP({ls.S2_API: [http_error(ls.S2_API, 400)]})
    with pytest.raises(ls.LitSearchError, match="HTTP 400"):
        ls.get(ls.S2_API, source="s2", fetch=not_retried, sleep=sleeps.append)
    offline = FakeHTTP({ls.S2_API: [urllib.error.URLError("no route")]})
    with pytest.raises(ls.LitSearchError, match="network error"):
        ls.get(ls.S2_API, source="s2", fetch=offline, sleep=sleeps.append)
    assert len(offline.calls) == 1


def test_failing_source_aborts_without_partial_output(
    ls: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeHTTP(
        {ls.ARXIV_API: [ARXIV_FEED], ls.OPENALEX_API: [b"<html>maintenance</html>"]}
    )
    code = ls.main(
        ["search", "tennis", "--source", "arxiv", "--source", "openalex"], fetch=fake
    )
    out = capsys.readouterr()
    assert code == 1 and out.out == "" and "openalex: response is not JSON" in out.err
    for payload, message in [
        (b'{"error": "x"}', "no 'data'"),
        (b'{"data": [{"title": "t"}]}', "without paperId"),
    ]:
        with pytest.raises(ls.LitSearchError, match=message):
            ls.search("q", ["s2"], 1, None, FakeHTTP({ls.S2_API: [payload]}))


def test_cli_json_output_and_argument_validation(
    ls: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeHTTP({ls.ARXIV_API: [ARXIV_FEED], ls.S2_API: [S2_JSON]})
    assert (
        ls.main(
            ["search", "tennis", "--source", "arxiv", "--source", "s2", "--max", "3"],
            fetch=fake,
        )
        == 0
    )
    records = json.loads(capsys.readouterr().out)
    assert [r["source"] for r in records] == ["arxiv", "s2"]
    assert set(records[0]) == {
        "source",
        "id",
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "url",
        "pdf_url",
        "doi",
        "arxiv_id",
        "citations",
    }
    with pytest.raises(SystemExit):
        ls.main(
            ["search", "tennis"], fetch=fake
        )  # --source is required: no implicit default set
    with pytest.raises(SystemExit):
        ls.main(["search", "tennis", "--source", "scholar"], fetch=fake)
    assert ls.main(["search", " ", "--source", "arxiv"], fetch=fake) == 1
    assert ls.main(["search", "x", "--source", "arxiv", "--max", "0"], fetch=fake) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2020", (2020, 2020)),
        ("2020-", (2020, None)),
        ("-2021", (None, 2021)),
        ("2019-2021", (2019, 2021)),
    ],
)
def test_year_range_parse(
    ls: ModuleType, text: str, expected: tuple[int | None, int | None]
) -> None:
    years = ls.YearRange.parse(text)
    assert (years.start, years.end) == expected


@pytest.mark.parametrize("text", ["", "-", "20", "2021-2019", "2020--2021"])
def test_year_range_rejects_invalid(ls: ModuleType, text: str) -> None:
    with pytest.raises(ValueError):
        ls.YearRange.parse(text)


def test_arxiv_pdf_download_validates_content(ls: ModuleType, tmp_path: Path) -> None:
    pdf = b"%PDF-1.5\n" + b"0" * ls.MIN_PDF_BYTES
    out = tmp_path / "papers/vid2player.pdf"
    fake = FakeHTTP({"https://arxiv.org/pdf/2008.04524": [pdf]})
    assert ls.download_arxiv_pdf("https://arxiv.org/abs/2008.04524v2", out, fake) == out
    assert out.read_bytes() == pdf
    with pytest.raises(FileExistsError):
        ls.download_arxiv_pdf("2008.04524", out, fake)
    html = FakeHTTP({"https://arxiv.org/pdf/": [b"<html>" + b"x" * ls.MIN_PDF_BYTES]})
    with pytest.raises(ls.LitSearchError, match="did not return a PDF"):
        ls.download_arxiv_pdf("2008.04525", tmp_path / "bad.pdf", html)
    assert not (tmp_path / "bad.pdf").exists()
    with pytest.raises(ValueError, match="not an arXiv ID"):
        ls.download_arxiv_pdf("../etc/passwd", tmp_path / "x.pdf", html)
