import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/browser",
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: "http://127.0.0.1:3137",
    browserName: "chromium",
    viewport: { width: 1440, height: 1000 },
  },
  webServer: {
    command:
      "KNOWLEDGE_NEXT_DIST=.next-e2e npm run dev -- --hostname 127.0.0.1 --port 3137",
    url: "http://127.0.0.1:3137",
    timeout: 120000,
    reuseExistingServer: false,
  },
});
