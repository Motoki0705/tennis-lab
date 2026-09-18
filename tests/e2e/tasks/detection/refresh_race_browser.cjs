// Deterministic regression for the catalog-refresh / scene-list race.
// 旧datasetの一覧応答が、更新後catalogの選択sceneとして残らないことを検証する。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname, "../../../..");
const staticRoot = path.join(
  root,
  "src/tasks/base/visualization/detection/static",
);
// 1x1 PNG。画像内容ではなく選択状態だけを検証するため、外部データに依存しない。
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);
const CATALOGS = {
  old: {
    task: "ball_detection",
    title: "Ball Detection",
    mode: "review",
    cuda_available: false,
    datasets: [
      {
        id: "old",
        label: "tennis / old",
        path: "/data/tennis/old",
        available: true,
        count: 1,
      },
    ],
    checkpoints: [],
    warnings: [],
  },
  new: {
    task: "ball_detection",
    title: "Ball Detection",
    mode: "review",
    cuda_available: false,
    datasets: [
      {
        id: "new",
        label: "tennis / new",
        path: "/data/tennis/new",
        available: true,
        count: 1,
      },
    ],
    checkpoints: [],
    warnings: [],
  },
};
const SCENES = {
  old: [{ id: "old::one", label: "old / one", frames: 3 }],
  new: [{ id: "new::one", label: "new / one", frames: 3 }],
};
function deferred() {
  let resolve;
  const promise = new Promise((r) => (resolve = r));
  return { promise, resolve };
}
async function waitFor(predicate, description) {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > 5000)
      throw new Error(`timeout waiting for ${description}`);
    await new Promise((r) => setTimeout(r, 10));
  }
}
// releaseOrder: 旧一覧応答を catalog 反映の "during" 中か "after" 後に解放する。
async function runScenario(browser, releaseOrder) {
  const sameDataset = releaseOrder === "interaction";
  const updatedCatalog = sameDataset ? CATALOGS.old : CATALOGS.new;
  const updatedScenes = sameDataset
    ? [{ id: "old::two", label: "old / two", frames: 3 }]
    : SCENES.new;
  let catalogReleased = false;
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const requests = [];
  page.on("request", (request) => requests.push(request.url()));
  const catalogGate = deferred();
  const staleSceneGate = deferred();
  let catalogCalls = 0;
  // 初回の旧dataset一覧だけを保留し、refresh中の応答として後から解放する。
  let holdStaleScenes = true;
  await page.route("http://detection.test/**", async (route) => {
    const url = new URL(route.request().url());
    const p = url.pathname;
    const json = (body) => route.fulfill({ json: body });
    if (p === "/api/catalog") {
      catalogCalls += 1;
      const body = catalogCalls === 1 ? CATALOGS.old : updatedCatalog;
      if (catalogCalls > 1) await catalogGate.promise;
      if (catalogCalls > 1) catalogReleased = true;
      return json(body);
    }
    if (p === "/api/scenes") {
      const dataset = url.searchParams.get("dataset");
      if (dataset === "old" && holdStaleScenes) {
        holdStaleScenes = false;
        await staleSceneGate.promise;
      }
      const items = catalogReleased ? updatedScenes : SCENES[dataset] || [];
      return json({ items, total: items.length });
    }
    if (p === "/api/preview") {
      const index = Number(url.searchParams.get("start"));
      return json({
        scene: url.searchParams.get("scene"),
        label: "Clip",
        frames: 3,
        start: index,
        width: 1,
        height: 1,
        items: [
          {
            index,
            name: `frame_${index}.jpg`,
            gt: { points: [{ x: 0, y: 0, label: "b001" }], rasters: [] },
          },
        ],
        warnings: [],
      });
    }
    if (p === "/api/image")
      return route.fulfill({ contentType: "image/png", body: PNG });
    const file = p === "/" ? "index.html" : p.replace("/static/", "");
    if (
      ![
        "index.html",
        "app.js",
        "viewer.mjs",
        "icons.mjs",
        "style.css",
      ].includes(file)
    )
      return route.fulfill({ status: 404, body: "" });
    return route.fulfill({
      contentType: file.endsWith(".html")
        ? "text/html"
        : file.endsWith(".css")
          ? "text/css"
          : "text/javascript",
      body: fs.readFileSync(path.join(staticRoot, file)),
    });
  });
  await page.goto("http://detection.test/");
  const catalogCount = () =>
    requests.filter((u) => u.endsWith("/api/catalog")).length;
  const sceneRequests = () =>
    requests.filter((u) => u.includes("/api/scenes") && u.includes("dataset="));
  await waitFor(
    () => catalogCount() === 1 && sceneRequests().length === 1,
    "initial catalog and held scene list",
  );
  await page.click("#refresh");
  await waitFor(() => catalogCount() === 2, "refresh catalog request");
  if (sameDataset) {
    await page.locator(".dataset").click();
    await page.waitForFunction(
      () => document.getElementById("scene-title").textContent === "old / one",
    );
    staleSceneGate.resolve();
    catalogGate.resolve();
  } else if (releaseOrder === "during") {
    // 旧一覧応答を、更新後catalogの反映前に解放する。
    staleSceneGate.resolve();
    await page.waitForTimeout(50);
    catalogGate.resolve();
  } else {
    // 旧一覧応答を、更新後catalogの反映後に解放する。
    catalogGate.resolve();
    await waitFor(() => sceneRequests().length >= 2, "refreshed scene list");
    await page.waitForTimeout(30);
    staleSceneGate.resolve();
  }
  await page.waitForFunction(
    () => document.getElementById("status").textContent === "カタログ更新完了",
  );
  return { page, errors };
}
async function assertRefreshedSelection(page, releaseOrder) {
  const where = `${releaseOrder}: `;
  const sameDataset = releaseOrder === "interaction";
  assert.equal(
    await page.locator("#dataset-label").textContent(),
    sameDataset ? "tennis / old" : "tennis / new",
    `${where}dataset label`,
  );
  assert.equal(await page.locator(".dataset").count(), 1, `${where}datasets`);
  assert.equal(
    await page.locator(".dataset").first().getAttribute("class"),
    "dataset selected",
    `${where}selected dataset`,
  );
  assert.equal(await page.locator(".scene").count(), 1, `${where}scene list`);
  assert.equal(
    await page.locator(".scene").first().getAttribute("data-scene"),
    sameDataset ? "old::two" : "new::one",
    `${where}rendered scene`,
  );
  assert.equal(
    await page.locator(".scene").first().getAttribute("class"),
    "scene selected",
    `${where}selected scene`,
  );
  assert.equal(
    await page.locator("#scene-title").textContent(),
    sameDataset ? "old / two" : "new / one",
    `${where}scene title`,
  );
}
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH,
    args: ["--no-sandbox"],
  });
  try {
    for (const releaseOrder of ["during", "after", "interaction"]) {
      const { page, errors } = await runScenario(browser, releaseOrder);
      try {
        await assertRefreshedSelection(page, releaseOrder);
        assert.deepEqual(errors, []);
      } finally {
        await page.close();
      }
    }
    console.log(
      "PASS: stale scene-list response cannot become the selected scene of a refreshed catalog",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
