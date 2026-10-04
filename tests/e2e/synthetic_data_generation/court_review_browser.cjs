/* Read-only browser regression against published B00–B03. */
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH
      ? { executablePath: process.env.CHROMIUM_PATH }
      : {}),
    args: ["--no-sandbox"],
  });
  try {
    const baseURL = process.env.COURT_REVIEW_URL || "http://127.0.0.1:8778";
    const page = await browser.newPage({
        viewport: { width: 1440, height: 1000 },
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(baseURL);
    await page.waitForSelector(".trajectory", { timeout: 90000 });
    const scenes = await page
      .locator("#scene option")
      .evaluateAll((options) => options.map((o) => o.value));
    assert(scenes.includes("B00") && scenes.includes("B03"));
    const modeBox = await page.locator("#image-mode").boundingBox();
    assert(modeBox && modeBox.y + modeBox.height < 1000);
    assert.equal(await page.locator("#label-scope").inputValue(), "target");
    await page.waitForSelector("#catalog-table tbody button");
    assert.match(
      await page.locator("#catalog-total").textContent(),
      /8,415採用 · 601 reject/,
    );
    for (const scene of ["B00", "B01", "B02", "B03"]) {
      await page.selectOption("#scene", scene);
      await page.waitForFunction(
        () =>
          document.querySelectorAll(".trajectory").length > 0 &&
          document.querySelector("#status").hidden,
        {},
        { timeout: 90000 },
      );
      await page.waitForFunction(
        () => {
          const img = document.querySelector(".thumbnail img");
          return img?.complete && img.naturalWidth > 0;
        },
        {},
        { timeout: 30000 },
      );
      const dimensions = await page.locator("#gallery").evaluate((el) => ({
        height: el.clientHeight,
        tile: el.firstElementChild.getBoundingClientRect().height,
        count: el.children.length,
      }));
      assert(
        Math.abs((dimensions.height + 8) / (dimensions.tile + 8) - 5.5) < 0.05,
      );
      await page.locator(".trajectory").nth(1).click();
      assert.equal(
        await page.locator(".trajectory[aria-current=true]").count(),
        1,
      );
      const name = await page
        .locator(".trajectory[aria-current=true] .name span")
        .first()
        .textContent();
      assert.equal(await page.locator("#selected-title").textContent(), name);
      const all = await page.locator(".thumbnail").count();
      assert(all > 5);
      await page
        .locator("#gallery")
        .evaluate((el) => (el.scrollTop = el.scrollHeight));
      await page.waitForFunction(
        () => {
          const img = document.querySelector(".thumbnail:last-child img");
          return img?.complete && img.naturalWidth > 0;
        },
        {},
        { timeout: 30000 },
      );
      await page.locator(".thumbnail").last().click();
      assert(await page.locator("#lightbox").isVisible());
      await page.waitForFunction(() => {
        const i = document.querySelector("#large-image");
        return i.complete && i.naturalWidth > 0;
      });
      assert.equal(await page.locator("#next").isDisabled(), true);
      await page.keyboard.press("ArrowLeft");
      assert(
        (await page.locator("#position").textContent()).startsWith(
          `${all - 1} /`,
        ),
      );
      await page.keyboard.press("Escape");
      assert.equal(await page.locator("#lightbox").isVisible(), false);
      await page.selectOption("#image-mode", "compare");
      await page.locator(".thumbnail").first().click();
      await page.waitForFunction(() => {
        const raw = document.querySelector("#raw-image"),
          overlay = document.querySelector("#large-image");
        return (
          raw.complete &&
          raw.naturalWidth > 0 &&
          overlay.complete &&
          overlay.naturalWidth > 0 &&
          document.querySelector("#point-table table")
        );
      });
      assert(await page.locator("#raw-pane").isVisible());
      assert.equal(await page.locator("#point-table table").count(), 1);
      assert.equal(await page.locator("#point-table tr").count(), 14);
      assert.match(
        await page.locator("#large-caption").textContent(),
        /target court-/,
      );
      await page.selectOption("#large-scope", "all");
      await page.waitForFunction(
        () =>
          document
            .querySelector("#large-caption")
            .textContent.includes("全court参考") &&
          document.querySelectorAll("#point-table table").length > 0,
      );
      assert.match(
        await page.locator("#large-sample-info").textContent(),
        /全court参考/,
      );
      await page.selectOption("#large-scope", "target");
      await page.waitForFunction(
        () => document.querySelectorAll("#point-table tr").length === 14,
      );
      assert.match(
        await page.locator("#large-sample-info").textContent(),
        /target court-/,
      );
      await page.keyboard.press("Escape");
      await page.selectOption("#image-mode", "overlay");
      await page.selectOption("#split-filter", "test");
      const splitCount = await page.locator(".trajectory").count();
      assert(splitCount > 0);
      const sceneList = await (
        await page.request.get(new URL("/api/scenes", baseURL).href)
      ).json();
      const response = await page.request.get(
        new URL(
          `/api/scenes/${scene}?revision=${sceneList.find((s) => s.id === scene).revision}`,
          baseURL,
        ).href,
      );
      const summary = await response.json();
      if (summary.courts.length > 1) {
        const s = summary.groups
          .flatMap((g) => g.samples)
          .find(
            (s) =>
              s.target_counts && s.counts.in_frame > s.target_counts.in_frame,
          );
        assert(
          s,
          "a sample with additional in-frame courts is required for scope comparison",
        );
        const endpoint = new URL(
          `/api/scenes/${scene}/images/${s.id}?revision=${summary.revision}&width=480`,
          baseURL,
        ).href;
        const targetImage = await (
          await page.request.get(endpoint + "&label_scope=target")
        ).body();
        const allImage = await (
          await page.request.get(endpoint + "&label_scope=all")
        ).body();
        assert(
          !targetImage.equals(allImage),
          "target and all-court image caches must remain distinct",
        );
      }
      assert.equal(
        splitCount,
        summary.groups.filter((g) => g.split === "test").length,
      );
      if (scene === "B01") {
        assert.deepEqual(summary.split_targets.test, {
          "court-000": 0,
          "court-001": 198,
          "court-002": 0,
        });
        assert.equal(await page.locator(".zero-count").count(), 2);
      }
      await page.selectOption("#split-filter", "all");
      await page.locator("#rejection-details summary").click();
      await page.locator("#rejected-table tbody button").first().click();
      await page.waitForFunction(
        () =>
          document.querySelector("#reject-dialog").open &&
          document.querySelector("#reject-info details"),
      );
      assert.equal(await page.locator("#reject-dialog img").count(), 0);
      const rejectedSample = summary.rejections.samples[0];
      if (rejectedSample.projection_recorded) {
        assert(
          (
            await page.locator("#reject-projection > h3").first().textContent()
          ).startsWith(rejectedSample.target_court),
        );
        assert.equal(
          await page.locator("#reject-projection > table tr").count(),
          14,
        );
        assert.equal(
          await page
            .locator("#reject-projection .reference-courts[open]")
            .count(),
          0,
        );
      }
      assert.equal(
        await page.locator("#large-image").getAttribute("src"),
        null,
      );
      assert.match(
        await page.locator("#reject-dialog").textContent(),
        /画像未保存/,
      );
      assert.match(
        await page.locator("#reject-info").textContent(),
        /scene位置\(m\)/,
      );
      await page.locator("#reject-close").click();
      await page.locator("#rejection-details summary").click();
      await page.evaluate(() => scrollTo(0, 0));
      console.log(
        `${scene}: ${await page.locator(".trajectory").count()} trajectories; gallery, full-screen, compare, split and reject isolation passed`,
      );
    }
    // Clicking a pickable trajectory directly in the 3D canvas updates the list/gallery.
    await page.selectOption("#scene", "B01");
    await page.waitForFunction(
      () =>
        document.querySelectorAll(".trajectory").length === 26 &&
        document.querySelector("#status").hidden,
      {},
      { timeout: 90000 },
    );
    const hit = await page.evaluate(async () => {
      const { project, distanceToSegment } = await import("/static/scene.mjs");
      const scenes = await (await fetch("/api/scenes")).json(),
        scene = scenes.find((s) => s.id === "B01"),
        data = await (
          await fetch(`/api/scenes/B01?revision=${scene.revision}`)
        ).json(),
        rect = document.querySelector("#view").getBoundingClientRect();
      const pts = [
          ...data.groups.flatMap((g) => g.points),
          ...data.courts.flatMap((c) => c.points),
        ],
        low = [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i]))),
        high = [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i]))),
        extent = Math.max(...high.map((x, i) => x - low[i]), 20),
        v = {
          center: low.map((x, i) => (x + high[i]) / 2),
          yaw: -0.55,
          pitch: 0.85,
          distance: extent * 4,
          scale: (Math.min(rect.width, rect.height) / extent) * 0.75,
          pan: [0, 0],
          width: rect.width,
          height: rect.height,
        };
      const paths = data.groups.map((g) => ({
        id: g.id,
        points: g.points.map((p) => project(p, v)),
      }));
      for (const g of paths.slice(1)) {
        for (const point of g.points) {
          if (
            point[0] < 20 ||
            point[0] > rect.width - 20 ||
            point[1] < 120 ||
            point[1] > rect.height - 120
          )
            continue;
          let nearest = null,
            best = 9;
          for (const path of paths)
            for (let i = 1; i < path.points.length; i++) {
              const d = distanceToSegment(
                point,
                path.points[i - 1],
                path.points[i],
              );
              if (d < best) {
                best = d;
                nearest = path.id;
              }
            }
          if (nearest === g.id)
            return {
              x: point[0] + rect.left,
              y: point[1] + rect.top,
              id: g.id,
            };
        }
      }
      throw Error("No pickable path");
    });
    await page.mouse.click(hit.x, hit.y);
    await page.waitForFunction(
      (id) =>
        document.querySelector(".trajectory[aria-current=true]").dataset
          .group === id,
      hit.id,
    );
    const canvas = page.locator("#view"),
      box = await canvas.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + box.width / 2 + 90,
      box.y + box.height / 2 + 30,
      { steps: 12 },
    );
    await page.mouse.up();
    await page.mouse.wheel(0, -120);
    await page.getByRole("link", { name: "統計を見る ↓" }).click();
    await page.waitForFunction(() => scrollY > 100);
    assert(await page.locator("#statistics").isVisible());
    await page.evaluate(() => scrollTo(0, 0));
    await page.click("#reset");
    await page.screenshot({
      path:
        process.env.COURT_REVIEW_SCREENSHOT || "/tmp/court-review-tested.png",
    });
    await page.setViewportSize({ width: 1100, height: 760 });
    await page.waitForTimeout(100);
    const ratio = await page
      .locator("#gallery")
      .evaluate(
        (el) =>
          (el.clientHeight + 8) /
          (el.firstElementChild.getBoundingClientRect().height + 8),
      );
    assert(Math.abs(ratio - 5.5) < 0.05);
    assert.deepEqual(errors, []);
    console.log(
      "3D picking, camera manipulation, statistics scrolling, responsive layout: passed",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
