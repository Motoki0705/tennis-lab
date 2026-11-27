from decimal import Decimal
from pathlib import Path

import pytest

from tools.wiki_assets.chart_spec import Scatter, pareto_frontier
from tools.wiki_assets.pages import Asset, asset_data, project
from tools.wiki_assets.sources import DEFAULT_REVISION, ROOT, Sources, fraction
from tools.wiki_assets.table_renderer import typst_source
from tools.wiki_assets.table_spec import Column, Row, Table, markdown_table


def test_table_keeps_zero_missing_and_exact_pre_rounding_rank() -> None:
    table = Table(
        "quality",
        "精度表",
        "固定入力の例",
        (
            Column("方式", 75),
            Column("Recall", 57, 2, "%", 100, "max"),
            Column("状態", 75),
        ),
        (
            Row("near", ("Near", Decimal("0.923441"), "計測"), "方式 A"),
            Row("best", ("Best", Decimal("0.923449"), "計測"), "方式 A"),
            Row("tie", ("Tie", Decimal("0.923449"), "計測"), "方式 B"),
            Row("zero", ("Zero", Decimal(0), "計測"), "方式 B"),
            Row("missing", ("Missing", None, "打ち切り"), exclusion="打ち切り"),
            Row(
                "diagnostic",
                ("Batch 2", Decimal("0.98"), "変換異常"),
                exclusion="変換異常",
            ),
        ),
    )
    markdown = markdown_table(table)
    assert "| Near | 92.34 | 計測 |" in markdown
    assert "| Best | **92.34** | 計測 |" in markdown
    assert "| Tie | **92.34** | 計測 |" in markdown
    assert "| Zero | 0.00 | 計測 |" in markdown
    assert "| Missing | — | 打ち切り |" in markdown
    assert "| Batch 2 | 98.00 | 変換異常 |" in markdown
    assert "| **方式 B** | | |" in markdown
    typst = typst_source(table)
    assert typst.count('strong(numeric("92.34"))') == 2
    assert 'numeric("0.00")' in typst
    assert 'numeric("—")' in typst
    assert 'table.cell(colspan: 3, fill: band, align: left, strong("方式 B"))' in typst


@pytest.mark.parametrize("value", [True, float("nan"), "", "Infinity", "-1", "1.01"])
def test_fraction_rejects_unavailable_or_untyped_values(value: object) -> None:
    with pytest.raises(ValueError):
        fraction(value)


def test_pareto_excludes_superior_diagnostics_and_uses_vram_when_requested() -> None:
    def row(
        key: str, latency: str, accuracy: str, memory: int, exclusion: str | None = None
    ) -> Row:
        return Row(
            key,
            (key, Decimal(latency), Decimal(accuracy), Decimal(memory)),
            exclusion=exclusion,
        )

    data = Table(
        "pareto",
        "Test",
        "One input cohort",
        (
            Column("Model", 52),
            Column("P50", 54, 2, "ms"),
            Column("OKS", 54, 3),
            Column("VRAM", 54, 0, "MiB"),
        ),
        (
            row("fast", "1.99", "0.701", 50),
            row("slower", "2.11", "0.7009", 30),
            row("accurate", "3.50", "0.7152", 80),
            row("dominated", "3.60", "0.71519", 81),
            row("diagnostic", "0.3", "0.99", 5, "conversion fault"),
            Row("missing", ("missing", None, None, None), exclusion="unmeasured"),
        ),
    )
    assert [r.key for r in pareto_frontier(Scatter(data, 1, 2))] == ["fast", "accurate"]
    assert [r.key for r in pareto_frontier(Scatter(data, 1, 2, memory=3))] == [
        "fast",
        "slower",
        "accurate",
    ]


def test_fixture_csv_and_json_validate_rectangular_and_typed_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "cases.csv").write_text(
        "model,value,status\nA,0,measured\nB,,load only\n"
    )
    (tmp_path / "method.json").write_text('{"target": 0.8}')
    sources = Sources("a" * 40, lambda p: (tmp_path / Path(p).name).read_bytes())
    cells = sources.csv("cases.csv")
    assert [(r["value"], r["status"]) for r in cells] == [
        ("0", "measured"),
        ("", "load only"),
    ]
    assert sources.json("method.json")["target"] == Decimal("0.8")
    (tmp_path / "cases.csv").write_text("model,value,status\nA,0\n")
    with pytest.raises(ValueError, match="source changed"):
        sources.csv("cases.csv")
    with pytest.raises(ValueError, match="wrong number"):
        Sources("a" * 40, lambda p: (tmp_path / Path(p).name).read_bytes()).csv(
            "cases.csv"
        )


@pytest.fixture(scope="module")
def page_assets() -> dict[str, Asset]:
    assets = {}
    for slug in (
        "person-model-comparison-methods",
        "person-detector-and-ecpose-comparison",
        "body-recovery-fasthmr-vs-gvhmr",
        "top-down-pose-model-comparison",
    ):
        for asset in project(slug, Sources.from_git(ROOT, DEFAULT_REVISION)):
            assets[asset_data(asset).slug] = asset
    return assets


def test_public_detector_projection_preserves_noncomparable_conditions(
    page_assets: dict[str, Asset],
) -> None:
    chart = page_assets["detection-b1-pareto"]
    assert isinstance(chart, Scatter)
    assert len(chart.plotted_rows) == 14
    assert sum(row.exclusion is not None for row in chart.plotted_rows) == 4
    assert {row.key for row in pareto_frontier(chart)} == {
        "rfdetr-l/pt16",
        "rfdetr-l/trt16",
        "ecdet-l-o365/pt16",
        "ecdet-x-o365/pt16",
    }
    rows = {r.key: r for r in asset_data(page_assets["detection-b1-speed"]).rows}
    assert rows["rfdetr-2xl/trt16"].cells[1:] == (
        None,
        None,
        None,
        None,
        "ライセンス除外",
    )
    assert rows["rtdetrv4-l/trt16"].cells[1] is None
    assert rows["rtdetrv4-l/trt16-fp32attn"].exclusion == "変換異常・実行不能"
    b2 = asset_data(page_assets["joint-b2-speed"])
    assert all(row.exclusion == "速度のみ・精度未計測" for row in b2.rows)


def test_public_pose_and_body_projection_preserves_batch_and_scope(
    page_assets: dict[str, Asset],
) -> None:
    for batch, crops in [(1, 300), (4, 384), (8, 384)]:
        chart = page_assets[f"pose-b{batch}-pareto"]
        assert isinstance(chart, Scatter)
        assert f"{crops} crop" in chart.data.note
        assert len(pareto_frontier(chart)) == 9
        assert all(row.key.endswith(f"/{batch}") for row in chart.plotted_rows)
    rows = {r.key: r.cells for r in asset_data(page_assets["sapiens2-status"]).rows}
    assert rows["sapiens2-5b/PT32"][2:] == (None, None, "重みロードのみ")
    assert rows["sapiens2-1b/TRT16"][2:] == (None, None, "変換停止・計測取消")
    body = asset_data(page_assets["hmr-b1-wall-vram"])
    assert [c.heading for c in body.columns] == [
        "経路 / Backend",
        "P50 (ms/人・フレーム)",
        "NVML (MiB)",
        "状態",
    ]
    assert body.rows[0].cells[1] == Decimal("30.7678055")
    assert body.rows[-1].exclusion == "参考・経路差"
    assert "目視評価は未実施" in body.note
