import { test, expect } from "@playwright/test";

test("saved WASB comparison depth2 opens a map and ranking", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1600, height: 1800 });
  await page.goto(
    "/?citationSearch=search-wasb-comparison-depth2&citationRole=comparison",
  );
  await expect(page.getByLabel("保存済み探索", { exact: true })).toHaveValue(
    "search-wasb-comparison-depth2",
  );
  await expect(page.getByRole("region", { name: "探索の状態" })).toContainText(
    "探索完了",
  );
  await expect(
    page.locator(".react-flow__node[data-id='cite-baef928f0aefd5a6']"),
  ).toHaveCount(1);
  await expect(
    page.getByRole("alert", { name: "引用探索のエラー" }),
  ).toHaveCount(0);
  await expect
    .poll(() =>
      page.locator(".citation-canvas").evaluate((canvas) => {
        const bounds = canvas.getBoundingClientRect();
        return [...canvas.querySelectorAll(".react-flow__node")].every(
          (node) => {
            const rect = node.getBoundingClientRect();
            return (
              rect.left >= bounds.left &&
              rect.right <= bounds.right &&
              rect.top >= bounds.top &&
              rect.bottom <= bounds.bottom
            );
          },
        );
      }),
    )
    .toBeTruthy();
  await page.screenshot({ path: "assets/wasb-citation-map.png" });
  await page
    .getByRole("button", { name: "役割別ランキング", exact: true })
    .click();
  const ranking = page.getByRole("region", { name: "比較対象ランキング" });
  await expect(ranking.locator("tbody tr").first()).toBeVisible();
  await expect(ranking.locator("tbody tr")).toHaveCount(10);
  await expect(
    ranking.locator(".citation-count").filter({ hasText: /^2 件/ }),
  ).toHaveCount(3);
  await page.setViewportSize({ width: 1600, height: 1400 });
  await page.screenshot({ path: "assets/wasb-citation-ranking.png" });
  await ranking.locator(".citation-count").first().click();
  await expect(
    page.getByRole("complementary", { name: "引用論文の詳細" }),
  ).toContainText("判断理由");
  await expect(
    page.getByRole("complementary", { name: "引用論文の詳細" }),
  ).toContainText("根拠の位置");
  expect(errors).toEqual([]);
});
