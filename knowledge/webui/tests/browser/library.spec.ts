import { test, expect } from "@playwright/test";

test("browse, compare, follow research, and read PDF", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /^plcs/ }).click();
  await page.getByRole("textbox", { name: "検索", exact: true }).fill("accad");
  await expect(page.locator(".experiment-row")).toHaveCount(3);
  await page.locator(".compare-check input").first().check();
  await page.locator(".compare-check input").nth(1).check();
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
  const pdf = await request.get("/api/papers/paper-2024-gvhmr");
  expect(pdf.status()).toBe(200);
  expect(pdf.headers()["content-type"]).toBe("application/pdf");
  expect((await pdf.body()).subarray(0, 5).toString()).toBe("%PDF-");
  expect((await request.get("/api/papers/paper-2024-missing")).status()).toBe(
    404,
  );
  await page.getByRole("button", { name: "研究サマリー", exact: true }).click();
  await page
    .locator(".summary-document")
    .getByRole("link", {
      name: "run-plcs-accad-gvhmr-meiji-1000-v1-train",
      exact: true,
    })
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
  await page.goto("/");
  await expect(page.locator(".experiment-row")).toHaveCount(30);
  await page.getByRole("button", { name: "次へ", exact: true }).click();
  await expect(page.locator(".pagination")).toContainText("2 / 7");
  await page
    .getByRole("button", { name: /^synthetic data generation/ })
    .click();
  await expect(page.locator(".experiment-row")).toHaveCount(11);
  await expect(page.locator(".pagination")).toContainText("1 / 1");
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
