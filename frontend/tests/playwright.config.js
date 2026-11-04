import { defineConfig } from "@playwright/test";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifacts = path.resolve(frontend, "../.cache/ui-01");
mkdirSync(path.join(artifacts, "tmp"), { recursive: true });
process.env.TMPDIR = path.join(artifacts, "tmp");
const port = Number(process.env.UI_TEST_PORT || 4173);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error("Invalid UI_TEST_PORT");
const baseURL = `http://127.0.0.1:${port}`;
const installedChrome = existsSync("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.spec.js",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  outputDir: path.join(artifacts, "results"),
  reporter: [["list"], ["json", { outputFile: path.join(artifacts, "browser-results.json") }]],
  use: {
    baseURL,
    channel: process.env.PLAYWRIGHT_CHANNEL || (installedChrome ? "chrome" : undefined),
    headless: true,
    trace: "off",
    screenshot: "only-on-failure",
    video: "off",
  },
  webServer: {
    command: "node tests/mock-server.mjs",
    cwd: frontend,
    url: `${baseURL}/__tests/health`,
    reuseExistingServer: false,
    timeout: 30_000,
    stdout: "pipe",
    stderr: "pipe",
    env: { UI_TEST_PORT: String(port) },
  },
});
