import { test, expect } from "@playwright/test";
import matter from "gray-matter";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const nodeRoot = fileURLToPath(new URL("../../../nodes/", import.meta.url));
const pageSize = 30;
const nodeFiles = (task: string) =>
  readdirSync(join(nodeRoot, task))
    .filter((name) => /^\d{6}-(run|group)-.+\.md$/.test(name))
    .map((name) => join(nodeRoot, task, name));
const allNodeFiles = readdirSync(nodeRoot, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .flatMap((entry) => nodeFiles(entry.name));
const totalNodes = allNodeFiles.length;

test("browse, compare, follow research, and open the primary paper source", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /^plcs/ }).click();
  await page.getByRole("textbox", { name: "検索", exact: true }).fill("accad");
  // The live research library grows. Select the intended matching experiments
  // by identity instead of fixing the number or order of ACCAD search results.
  for (const id of [
    "run-plcs-accad-gvhmr-meiji-1000-v1",
    "run-plcs-accad-gvhmr-meiji-1000-v1-train",
  ]) {
    await page.getByLabel(`${id} を比較`, { exact: true }).check();
  }
  await page.getByRole("button", { name: "実験比較 (2)" }).click();
  await expect(page.locator(".comparison-table")).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "CSVを書き出す" }).click();
  expect((await download).suggestedFilename()).toBe(
    "experiment-comparison.csv",
  );
  await page.goto("/?node=run-plcs-accad-gvhmr-meiji-1000-v1-train");
  await expect(
    page.getByRole("complementary", { name: "実験の詳細" }),
  ).toContainText("0.382341");
  await page
    .locator(".panel")
    .getByRole("button", { name: /World-Grounded/ })
    .click();
  await expect(page.locator(".paper-detail")).toBeVisible();
  const primarySource = page.locator(".paper-detail").getByRole("link", { name: "一次資料を開く" });
  await expect(primarySource).toHaveAttribute("href", "https://arxiv.org/abs/2409.06662v1");
  await expect(primarySource).toHaveAttribute("target", "_blank");
  await expect(page.locator(".paper-detail")).toContainText("Gravity-View");
  await expect(page.locator(".paper-detail")).toContainText("採録理由");
  await expect(page.locator(".paper-detail")).toContainText("固定split");
  await expect(page.locator(".paper-detail a[href^='/api/papers/']")).toHaveCount(0);
  expect((await request.get("/api/papers/paper-2024-gvhmr")).status()).toBe(404);
  await page.screenshot({ path: "/tmp/knowledge-papers-document-only.png", fullPage: true });
  await page.getByRole("button", { name: "研究サマリー", exact: true }).click();
  await page
    .locator(".summary-document a[href^='/?node=']")
    .first()
    .click();
  await expect(
    page.getByRole("complementary", { name: "実験の詳細" }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("complementary", { name: "実験の詳細" }),
  ).toHaveCount(0);
  await page.screenshot({ path: "/tmp/knowledge-library-desktop.png" });
  expect(errors).toEqual([]);
});

test("filters, pagination, graph and mobile layout", async ({ page }) => {
  const syntheticNodes = nodeFiles("synthetic_data_generation").length;
  expect(totalNodes).toBeGreaterThan(pageSize);
  await page.goto("/");
  await expect(page.locator(".experiment-row")).toHaveCount(pageSize);
  await page.getByRole("button", { name: "次へ", exact: true }).click();
  await expect(page.locator(".pagination")).toContainText(
    `2 / ${Math.ceil(totalNodes / pageSize)}`,
  );
  await page
    .getByRole("button", { name: /^synthetic data generation/ })
    .click();
  await expect(page.locator(".experiment-row")).toHaveCount(
    Math.min(syntheticNodes, pageSize),
  );
  await expect(page.locator(".pagination")).toContainText(
    `1 / ${Math.max(1, Math.ceil(syntheticNodes / pageSize))}`,
  );
  await page.getByRole("button", { name: "知識グラフ", exact: true }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
  await page.getByLabel("比較・確認などの関連線を表示").check();
  await page.getByRole("button", { name: "実験一覧", exact: true }).click();
  await page
    .getByRole("textbox", { name: "検索", exact: true })
    .fill("no-such-experiment");
  await expect(page.getByText("該当する実験がありません")).toBeVisible();
  await page.getByRole("button", { name: "リセット", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".experiment-row").first()).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
});

test("selected paper and experiment paper filter survive reload independently", async ({
  page,
}) => {
  const filterPaper = "paper-2024-gvhmr";
  const filterPaperTitle =
    "World-Grounded Human Motion Recovery via Gravity-View Coordinates";
  const otherPaper = "paper-2023-wasb";
  const otherPaperTitle =
    "Widely Applicable Strong Baseline for Sports Ball Detection and Tracking";
  const matchingIds = allNodeFiles
    .map((file) => matter(readFileSync(file, "utf8")).data)
    .filter(({ papers }) => Array.isArray(papers) && papers.includes(filterPaper))
    .map(({ id }) => String(id));
  expect(matchingIds.length).toBeGreaterThan(0);
  expect(matchingIds.length).toBeLessThan(totalNodes);

  const timeline = page.locator(".timeline");
  const rowIds = timeline.locator(".experiment-row .row-id");
  const paperHeading = page.locator(".paper-detail > h2");

  async function expectFilteredExperiments() {
    await expect(
      timeline.getByText(`${matchingIds.length} 件のノード`, { exact: true }),
    ).toBeVisible();
    await expect(rowIds).toHaveCount(Math.min(matchingIds.length, pageSize));
    const visibleIds = await rowIds.allTextContents();
    expect(new Set(visibleIds).size).toBe(visibleIds.length);
    for (const id of visibleIds) expect(matchingIds).toContain(id);
  }

  await page.goto(`/?paper=${filterPaper}`);
  await expect(paperHeading).toHaveText(filterPaperTitle);
  await page.reload();
  await expect(paperHeading).toHaveText(filterPaperTitle);
  await page.getByRole("button", { name: "実験一覧", exact: true }).click();
  await expect(
    timeline.getByText(`${totalNodes} 件のノード`, { exact: true }),
  ).toBeVisible();
  await expect(rowIds).toHaveCount(Math.min(totalNodes, pageSize));
  await expect(page).toHaveURL((url) => !url.searchParams.has("paperFilter"));

  await page.goto(`/?view=timeline&paperFilter=${filterPaper}`);
  await expectFilteredExperiments();
  await page.getByRole("button", { name: "Papers", exact: true }).click();
  await page
    .locator(".paper-card")
    .filter({
      has: page.getByRole("heading", { name: otherPaperTitle, exact: true }),
    })
    .click();
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get("paper") === otherPaper &&
      url.searchParams.get("paperFilter") === filterPaper,
  );
  await page.reload();
  await expect(paperHeading).toHaveText(otherPaperTitle);
  await page.getByRole("button", { name: "実験一覧", exact: true }).click();
  await expectFilteredExperiments();
  await expect(page).toHaveURL(
    (url) => url.searchParams.get("paperFilter") === filterPaper,
  );
});

test("nested and boolean measurements render in list, details, comparison and graph", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?query=run-court-b00-alignment-v14");
  const booleanRow = page.locator(".experiment-row").filter({
    has: page.getByLabel("run-court-b00-alignment-v14 を比較", { exact: true }),
  });
  await expect(booleanRow.locator(".row-metric strong").first()).toHaveText("true");
  const id = "run-court-b00-canonical-sfm";
  await page.goto(`/?query=${id}&node=${id}`);
  const panel = page.getByRole("complementary", { name: "実験の詳細" });
  await expect(panel).toContainText('{"test":238,"train":1718,"validation":220}');
  await page.keyboard.press("Escape");
  await page.getByLabel(`${id} を比較`, { exact: true }).check();
  await page.getByRole("button", { name: "実験比較 (1)" }).click();
  await expect(page.locator(".comparison-table")).toContainText('{"test":238,"train":1718,"validation":220}');
  await page.getByRole("button", { name: "知識グラフ", exact: true }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
  expect(errors).toEqual([]);
});
