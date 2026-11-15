import { test, expect } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { citationFixtureRoot, fixtureIds } from "../citation-fixture";

const mixed = "search-fixture-mixed-depth2";
const comparison = "search-fixture-comparison-depth2";
const detail = (page: import("@playwright/test").Page) =>
  page.getByRole("complementary", { name: "引用論文の詳細" });

test("complete and interrupted searches coexist and unfinished identification remains visible outside counts", async ({
  page,
}) => {
  await page.goto("/?view=papers");
  await expect(page.locator(".citation-search-card")).toHaveCount(5);
  await page
    .locator(".citation-search-card")
    .filter({ hasText: "search-fixture-pending-identification" })
    .click();
  const state = page.getByRole("region", { name: "探索の状態" });
  await expect(state).toContainText("探索中断");
  await expect(state).toContainText("Paper processing incomplete");
  await expect(page.locator(".citation-header")).toContainText("展開 0 論文");
  const waiting = page.getByRole("region", { name: "照会未完了の論文" });
  await expect(waiting).toContainText("照会待ち");
  await expect(waiting).toContainText("一意性未確認");
  await page
    .getByLabel("引用の役割", { exact: true })
    .selectOption("benchmark");
  await expect(waiting).toContainText("照会待ち");
  await waiting.getByRole("button", { name: "Fixture P", exact: true }).click();
  await detail(page)
    .getByText("照会に使う情報", { exact: true })
    .first()
    .click();
  await expect(
    detail(page).getByText("A Pending Target", { exact: true }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await page
    .getByRole("button", { name: "役割別ランキング", exact: true })
    .click();
  await expect(page.locator(".citation-rankings tbody tr")).toHaveCount(0);
  await page
    .getByLabel("保存済み探索", { exact: true })
    .selectOption("search-fixture-interrupted-depth2");
  await expect(state).toContainText("agy:cli_error");
  await page
    .getByLabel("引用の役割", { exact: true })
    .selectOption("comparison");
  await page.locator(".citation-relations summary").click();
  await expect(page.locator(".citation-relations article")).toHaveCount(3);
  await expect(
    page.getByRole("region", { name: "探索の問題箇所" }),
  ).toContainText("問題箇所 (2)");
  await page.getByLabel("保存済み探索", { exact: true }).selectOption(mixed);
  await expect(state).toContainText("探索完了");
  await expect(
    page.getByRole("alert", { name: "引用探索のエラー" }),
  ).toHaveCount(0);
});

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
  const convergence = page.locator(".citation-relations article").filter({
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
      path.join(citationDir, "papers", "cite-0000000000000000.md"),
      "---\nschema_version: 999\n---\n",
    );
    await page.reload();
    await expect(
      page.getByRole("alert", { name: "引用文書のエラー" }),
    ).toContainText("schema_version");
    await expect(
      page.getByRole("alert", { name: "引用文書のエラー" }),
    ).toContainText("cite-0000000000000000.md");
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
      page.getByRole("alert", { name: "引用文書のエラー" }),
    ).toContainText("conflicting external IDs");
    await expect(
      page.getByRole("alert", { name: "引用文書のエラー" }),
    ).toContainText(`citation_map/papers/${fixtureIds.B}.md`);
    await expect(page.locator(".citation-search-card")).toHaveCount(1);
    await page.locator(".citation-search-card").click();
    await expect(
      page.getByRole("region", { name: "探索の状態" }),
    ).toContainText("探索中断");
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toHaveCount(0);
  } finally {
    await fs.writeFile(paperPath, raw);
  }
  const searchPath = path.join(citationDir, "searches", `${mixed}.md`);
  const sourcePath = path.join(citationDir, "papers", `${fixtureIds.A}.md`);
  const searchRaw = await fs.readFile(searchPath, "utf8");
  const sourceRaw = await fs.readFile(sourcePath, "utf8");
  try {
    await fs.writeFile(
      searchPath,
      searchRaw.replace(
        /created_at:[^\r\n]*/,
        "created_at: '2026-11-14T22:00:00+0900'",
      ),
    );
    await page.goto(`/?citationSearch=${mixed}`);
    await expect(page.getByLabel("保存済み探索", { exact: true })).toHaveValue(
      mixed,
    );
    await expect(page.locator(".citation-header")).toContainText(
      "2026-11-14T22:00:00+0900",
    );
    await fs.writeFile(
      sourcePath,
      sourceRaw.replace(
        "source: https://example.org/fixture-a",
        "source: https:example.org/paper",
      ),
    );
    await page.reload();
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toContainText("absolute HTTP(S)");
  } finally {
    await fs.writeFile(searchPath, searchRaw);
    await fs.writeFile(sourcePath, sourceRaw);
  }
  await page.goto("/?citationSearch=search-missing");
  await expect(
    page.getByRole("alert", { name: "引用探索のエラー" }),
  ).toContainText("指定された探索が見つかりません");
});

test("healthy direct URLs preserve maps, ranking counts and Papers links beside an invalid search", async ({
  page,
}) => {
  const badId = "search-invalid-status";
  const file = path.join(
    citationFixtureRoot,
    "citation_map",
    "searches",
    `${badId}.md`,
  );
  const source = await fs.readFile(
    path.join(citationFixtureRoot, "citation_map", "searches", `${mixed}.md`),
    "utf8",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await fs.writeFile(
    file,
    source
      .replaceAll(mixed, badId)
      .replace("schema_version: 1", "status: invalid\nschema_version: 1"),
  );
  try {
    await page.goto(`/?citationSearch=${mixed}&citationRole=comparison`);
    const warning = page.getByRole("alert", { name: "引用文書のエラー" });
    await expect(warning).toContainText(`${badId}.md`);
    await expect(warning).toContainText("search status");
    await expect(
      page.locator(`.react-flow__node[data-id='${fixtureIds.D}']`),
    ).toHaveCount(1);
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "役割別ランキング", exact: true })
      .click();
    const ranking = page.getByRole("region", { name: "比較対象ランキング" });
    await expect(ranking.locator("tbody tr")).toHaveCount(3);
    await page.setViewportSize({ width: 1600, height: 1400 });
    await warning.scrollIntoViewIfNeeded();
    await page.screenshot({ path: "assets/citation-errors.png" });
    await ranking
      .getByRole("button", { name: "Fixture D の比較対象 2件の根拠" })
      .click();
    await expect(detail(page)).toContainText("引用元 2 論文");
    await expect(detail(page)).toContainText("Fixture B");
    await expect(detail(page)).toContainText("Fixture C");
    await detail(page).getByRole("button", { name: "Papersの詳細へ" }).click();
    await expect(page.locator(".paper-detail")).toContainText(
      "Registered Fixture D",
    );
    await page.goto(`/?citationSearch=${badId}&citationView=ranking`);
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toContainText(`${badId}.md`);
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toContainText("search status");
    await expect(page.locator(".citation-search-card")).toHaveCount(5);
    await page
      .locator(".citation-search-card")
      .filter({ hasText: mixed })
      .click();
    await expect(page.locator(".citation-canvas .react-flow")).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    await fs.rm(file, { force: true });
  }
});

test("WASB remains viewable with unchanged counts while another search is invalid", async ({
  page,
}) => {
  const real = fileURLToPath(
    new URL("../../../citation_map/", import.meta.url),
  );
  const target = path.join(citationFixtureRoot, "citation_map");
  const added: string[] = [];
  const badId = "search-invalid-wasb-neighbor";
  try {
    // Mutate only the isolated browser fixture, never the real knowledge store.
    for (const kind of ["papers", "searches"])
      for (const name of await fs.readdir(path.join(real, kind))) {
        const file = path.join(target, kind, name);
        await fs.copyFile(
          path.join(real, kind, name),
          file,
          fs.constants.COPYFILE_EXCL,
        );
        added.push(file);
      }
    const file = path.join(target, "searches", `${badId}.md`);
    added.push(file);
    const raw = await fs.readFile(
      path.join(real, "searches", "search-wasb-comparison-depth2.md"),
      "utf8",
    );
    await fs.writeFile(
      file,
      raw
        .replaceAll("search-wasb-comparison-depth2", badId)
        .replace("schema_version: 1", "status: invalid\nschema_version: 1"),
    );
    await page.goto(
      "/?citationSearch=search-wasb-comparison-depth2&citationRole=comparison",
    );
    await expect(
      page.getByRole("alert", { name: "引用文書のエラー" }),
    ).toContainText(`${badId}.md`);
    await expect(
      page.locator(".react-flow__node[data-id='cite-baef928f0aefd5a6']"),
    ).toHaveCount(1);
    await expect(
      page.getByRole("alert", { name: "引用探索のエラー" }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "役割別ランキング", exact: true })
      .click();
    const ranking = page.getByRole("region", { name: "比較対象ランキング" });
    await expect(ranking.locator("tbody tr")).toHaveCount(10);
    await expect(
      ranking.locator(".citation-count").filter({ hasText: /^2 件/ }),
    ).toHaveCount(3);
    await page.reload();
    await expect(ranking.locator("tbody tr")).toHaveCount(10);
    await ranking.locator(".citation-count").first().click();
    await expect(detail(page)).toContainText("判断理由");
    await expect(detail(page)).toContainText("根拠の位置");
  } finally {
    await Promise.all(added.map((file) => fs.rm(file, { force: true })));
  }
});
