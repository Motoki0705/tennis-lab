import { defineConfig } from "@playwright/test";
import { citationFixtureRoot } from "./tests/citation-fixture";
export default defineConfig({
  testDir: "./tests/browser",
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: "http://127.0.0.1:3137",
    browserName: "chromium",
    viewport: { width: 1440, height: 1000 },
  },
  projects: [
    { name: "library", testMatch: "library.spec.ts" },
    {
      name: "citations",
      testMatch: "citations.spec.ts",
      use: { baseURL: "http://127.0.0.1:3138" },
    },
  ],
  webServer: [
    {
      command:
        "KNOWLEDGE_NEXT_DIST=.next-e2e npm run dev -- --hostname 127.0.0.1 --port 3137",
      url: "http://127.0.0.1:3137",
      timeout: 120000,
      reuseExistingServer: false,
    },
    {
      command:
        "npx tsx tests/citation-fixture.ts && npm run dev -- --hostname 127.0.0.1 --port 3138",
      env: {
        KNOWLEDGE_DIR: citationFixtureRoot,
        KNOWLEDGE_NEXT_DIST: ".next-e2e-citations",
      },
      url: "http://127.0.0.1:3138",
      timeout: 120000,
      reuseExistingServer: false,
    },
  ],
});
