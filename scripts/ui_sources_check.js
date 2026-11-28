// Browser check for the sources toggle (Phase 1 of docs/ROADMAP_v4.md).
//
//   PYTHON=python node scripts/ui_sources_check.js [outDir]
//
// Starts `streamlit run scripts/ui_fake_app.py` (fake Gemini, fake chain),
// asks three questions in Chromium through Playwright, and checks what a
// visitor sees:
//   * only the latest answer shows "Sources:" in full;
//   * each earlier answer has a closed "Sources" expander with its own file;
//   * opening one shows that answer's citation and chunk, and no other's.
// Screenshots go to outDir (default .ui-check/). Not part of the test suite.

const { spawn } = require("child_process");
const fs = require("fs");
const net = require("net");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.dirname(__dirname);
const OUT = process.argv[2] || path.join(ROOT, ".ui-check");
const PYTHON = process.env.PYTHON || "python";

function freePort() {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitHealthy(url, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url + "/_stcore/health");
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("streamlit did not become healthy");
}

// Polls check() until it returns true; anything else is reported on timeout.
async function waitFor(check, what, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    last = await check();
    if (last === true) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`timed out waiting for: ${what} (last: ${last})`);
}

function assert(cond, msg) {
  if (!cond) throw new Error("check failed: " + msg);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const port = await freePort();
  const url = `http://127.0.0.1:${port}`;
  const server = spawn(PYTHON, ["-m", "streamlit", "run", "scripts/ui_fake_app.py",
    "--server.headless", "true", "--server.port", String(port),
    "--browser.gatherUsageStats", "false"], { cwd: ROOT, stdio: "ignore" });
  let browser;
  try {
    await waitHealthy(url);
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1280, height: 1600 },
                                         deviceScaleFactor: 2 });
    await page.goto(url);
    const chat = page.getByTestId("stChatInputTextArea");
    await chat.waitFor({ timeout: 120000 });

    const answers = page.getByText("An answer about");
    async function ask(question, expected) {
      await chat.fill(question);
      await chat.press("Enter");
      await waitFor(async () => {
        const n = await answers.count();
        return n === expected || n;
      }, `${expected} answers`);
      await page.waitForTimeout(800);
    }
    await ask("What about alpha?", 1);
    await ask("What about beta?", 2);
    await ask("What about gamma?", 3);

    const full = page.locator("[data-testid=stMarkdownContainer]", { hasText: "Sources:" });
    const earlier = page.locator("[data-testid=stExpander]")
      .filter({ has: page.getByText("Sources", { exact: true }) });
    await waitFor(async () => {
      const n = await earlier.count();
      return n === 2 || n;
    }, "two earlier Sources expanders");

    assert((await full.count()) === 1, "exactly one full Sources block");
    assert((await full.innerText()).includes("gamma.txt"), "full block is the latest answer's");
    for (let i = 0; i < 2; i++) {
      const open = await earlier.nth(i).locator("details").getAttribute("open");
      assert(open === null, `earlier expander ${i} starts closed`);
    }
    await page.screenshot({ path: path.join(OUT, "1-collapsed.png"), fullPage: true });

    await earlier.nth(0).getByText("Sources", { exact: true }).click();
    await waitFor(async () => (await earlier.nth(0).innerText()).includes("alpha.txt"),
                  "first expander shows alpha.txt");
    const firstText = await earlier.nth(0).innerText();
    assert(firstText.includes("Everything the documents say about alpha"), "chunk preview shown");
    assert(!firstText.includes("beta.txt") && !firstText.includes("gamma.txt"),
           "first expander holds only its own answer's sources");
    const secondOpen = await earlier.nth(1).locator("details").getAttribute("open");
    assert(secondOpen === null, "opening one expander leaves the other closed");
    await page.waitForTimeout(1000);                // let the expander finish opening
    await page.screenshot({ path: path.join(OUT, "2-earlier-opened.png"), fullPage: true });

    console.log(`OK: sources toggle behaves as specified; screenshots in ${OUT}`);
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch((err) => { console.error(err.message); process.exit(1); });
