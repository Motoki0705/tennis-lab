import { test, expect } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { citationFixtureRoot, fixtureIds } from "../citation-fixture";

const mixed = "search-fixture-mixed-depth2";
const comparison = "search-fixture-comparison-depth2";
const detail = (page: import("@playwright/test").Page) =>
  page.getByRole("complementary", { name: "引用論文の詳細" });

test("Papers opens a converging map, filters roles and links ranking counts to all evidence and registered Papers", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/?view=papers");
  // Citation-only papers never appear as registered Papers cards.
  await expect(page.locator(".paper-card")).toHaveCount(1);
  await page
    .locator(".citation-search-card")
    .filter({ hasText: mixed })
    .click();
  await expect(page.locator(".citation-canvas .react-flow")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "起点: Fixture A", exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(`.react-flow__node[data-id='${fixtureIds.D}']`),
  ).toHaveCount(1);
  await page.screenshot({ path: "assets/citation-map.png" });
  await page
    .getByLabel("引用の役割", { exact: true })
    .selectOption("comparison");
  await page.locator(".citation-relations summary").click();
  await expect(page.locator(".citation-relations article")).toHaveCount(4);
  const convergence = page
    .locator(".citation-relations article")
    .filter({
      has: page.getByRole("button", { name: "Fixture D", exact: true }),
    });
  await expect(convergence).toHaveCount(2);
  await expect(convergence.filter({ hasText: "Fixture B" })).toHaveCount(1);
  await expect(convergence.filter({ hasText: "Fixture C" })).toHaveCount(1);
  await page.screenshot({ path: "assets/citation-filter.png" });
  await page
    .getByRole("button", { name: "役割別ランキング", exact: true })
    .click();
  const ranking = page.getByRole("region", { name: "比較対象ランキング" });
  await expect(
    ranking.getByRole("button", { name: "Fixture D の比較対象 2件の根拠" }),
  ).toBeVisible();
  await expect(
    ranking.getByRole("button", { name: "Fixture B の比較対象 1件の根拠" }),
  ).toBeVisible();
  await page.screenshot({ path: "assets/citation-ranking.png" });
  await ranking
    .getByRole("button", { name: "Fixture D の比較対象 2件の根拠" })
    .click();
  await expect(detail(page)).toContainText("引用元 2 論文");
  await expect(detail(page)).toContainText("Fixture B uses D for comparison.");
  await expect(detail(page)).toContainText(
    "Fixture C uses D for comparison, foundation.",
  );
  await expect(detail(page)).toContainText("Section 3, paragraph 1; Table 1");
  await expect(detail(page)).toContainText("doi10.5555/fixture-d");
  await page.setViewportSize({ width: 1440, height: 1800 });
  await page.screenshot({ path: "assets/citation-evidence.png" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page).toHaveURL(new RegExp(`citationPaper=${fixtureIds.D}`));
  await page.reload();
  await expect(page.getByLabel("保存済み探索", { exact: true })).toHaveValue(
    mixed,
  );
  await expect(page.getByLabel("引用の役割", { exact: true })).toHaveValue(
    "comparison",
  );
  await expect(detail(page)).toContainText("引用元 2 論文");
  await detail(page).getByRole("button", { name: "Papersの詳細へ" }).click();
  await expect(page.locator(".paper-detail")).toContainText(
    "Registered Fixture D",
  );
  await expect(page).toHaveURL(/paper=paper-2024-fixture-d/);
  expect(errors).toEqual([]);
});

test("scope and issues stay explicit across search changes, keyboard, mobile and history", async ({
  page,
}) => {
  await page.goto(`/?citationSearch=${comparison}&citationRole=foundation`);
  // C is visible as A's foundation reference; its cached outgoing edge is out of scope.
  await page.locator(".citation-relations summary").click();
  await expect(page.locator(".citation-relations article")).toHaveCount(2);
  await page
    .getByRole("button", { name: "役割別ランキング", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: "Fixture D の基盤アーキテクチャ・モデル",
      exact: false,
    }),
  ).toHaveCount(0);
  const issues = page.getByRole("region", { name: "探索の問題箇所" });
  await expect(issues).toContainText("取得失敗");
  await expect(issues).toContainText("未同定");
  await expect(issues).toContainText("判定不能");
  await expect(issues).toContainText("HTTP 403");
  await page.getByLabel("保存済み探索", { exact: true }).selectOption(mixed);
  await page
    .getByLabel("引用の役割", { exact: true })
    .selectOption("comparison");
  const d = page.getByRole("button", {
    name: "Fixture D の引用詳細",
    exact: true,
  });
  await d.focus();
  await page.keyboard.press("Enter");
  await expect(
    detail(page).getByRole("heading", { name: "Fixture D", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(detail(page)).toHaveCount(0);
  await expect(d).toBeFocused();
  await page.setViewportSize({ width: 390, height: 844 });
  await d.click();
  await expect(detail(page)).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "/tmp/citation-map-webui/mobile.png",
    fullPage: true,
  });
  const saved = page.url();
  await page.evaluate(() => {
    history.pushState(null, "", "?view=papers");
    dispatchEvent(new PopStateEvent("popstate"));
  });
  await expect(page.locator(".paper-card")).toHaveCount(1);
  await page.goBack();
  await expect(page).toHaveURL(saved);
  await expect(detail(page)).toContainText("引用元 2 論文");
});

test("missing or empty storage displays no searches; malformed and conflicting documents display errors", async ({
  page,
}) => {
  const citationDir = path.join(citationFixtureRoot, "citation_map");
  const savedDir = path.join(citationFixtureRoot, "citation_map-saved");
  await fs.rename(citationDir, savedDir);
  try {
    await page.goto("/?view=papers");
    await expect(
      page.getByText("探索なし — 保存された引用探索はありません。"),
    ).toBeVisible();
    await fs.mkdir(citationDir);
    await page.reload();
    await expect(
      page.getByText("探索なし — 保存された引用探索はありません。"),
    ).toBeVisible();
    await fs.mkdir(path.join(citationDir, "papers"));
    await fs.writeFile(
      path.join(citationDir, "papers", "bad.md"),
      "---\nschema_version: 999\n---\n",
    );
    await page.reload();
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toContainText("schema_version");
    await expect(page.getByText("探索なし", { exact: false })).toHaveCount(0);
  } finally {
    await fs.rm(citationDir, { recursive: true, force: true });
    await fs.rename(savedDir, citationDir);
  }
  const paperPath = path.join(
    citationFixtureRoot,
    "Papers",
    "paper-2024-fixture-d",
    "paper.md",
  );
  const raw = await fs.readFile(paperPath, "utf8");
  try {
    // A matching DOI with a conflicting second namespace is an error, even for one candidate.
    await fs.writeFile(
      paperPath,
      raw.replace(
        "doi: 10.5555/fixture-d",
        "doi: 10.5555/fixture-b\n  arxiv: '2401.99999'",
      ),
    );
    await page.reload();
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toContainText("conflicting external IDs");
  } finally {
    await fs.writeFile(paperPath, raw);
  }
  await page.goto("/?citationSearch=search-missing");
  await expect(
    page.getByRole("alert", { name: "引用探索のエラー" }),
  ).toContainText("指定された探索が見つかりません");
});
