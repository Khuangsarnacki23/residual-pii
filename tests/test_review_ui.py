"""Drives the review page in a headless Chromium-family browser through Puppeteer.

The page's own script is exercised end to end against the synthetic batch: select text, press a type
key, undo, mark a message as empty, compare with the detectors. Skipped unless node, a `puppeteer`
or `puppeteer-core` module (global, or inside a globally installed package) and a browser are found.
"""
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pandas as pd
import pytest

from residpii import label, review
from test_label import labeled  # noqa: F401  (fixture)

BROWSERS = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            *filter(None, map(shutil.which, ("chromium", "chromium-browser", "google-chrome")))]

RUNNER = """
const puppeteer = require(process.argv[2]);
const stop = setTimeout(() => { console.log("TIMEOUT"); process.exit(2); }, 60000);
(async () => {
  const browser = await puppeteer.launch({ headless: true, executablePath: process.argv[3], args: ["--no-first-run"] });
  const page = await browser.newPage();
  await page.goto(process.argv[4]);
  await page.waitForSelector("#testout", { timeout: 45000 });
  console.log(await page.$eval("#testout", (el) => el.textContent));
  if (process.env.REVIEW_UI_SCREENSHOT) {          // for a look at the layout while changing the page
    await page.setViewport({ width: 1280, height: 900 });
    await page.evaluate(() => { document.getElementById("testout").remove(); goTo(L.items.findIndex((x) => x.spans.length)); reveal(L.items[L.i]); window.scrollTo(0, 0); });
    await page.screenshot({ path: process.env.REVIEW_UI_SCREENSHOT, fullPage: true });
  }
  await browser.close(); clearTimeout(stop);
})().catch((e) => { console.log("ERROR " + e); process.exit(1); });
"""


def _puppeteer() -> str | None:
    npm = shutil.which("npm")
    if not shutil.which("node") or not npm:
        return None
    root = Path(subprocess.run([npm, "root", "-g"], capture_output=True, text=True).stdout.strip())
    hits = [*root.glob("puppeteer*"), *root.glob("*/node_modules/puppeteer*"), *root.glob("@*/*/node_modules/puppeteer*")]
    return str(sorted(hits)[0]) if hits else None


DRIVER = """
(async function drive() {
  const out = {}, wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const press = (key) => document.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
  const select = (needle) => {
    const walker = document.createTreeWalker($("label-text"), NodeFilter.SHOW_TEXT);
    for (let node; (node = walker.nextNode());) {
      const at = node.data.indexOf(needle);
      if (at === -1) continue;
      const r = document.createRange(); r.setStart(node, at); r.setEnd(node, at + needle.length);
      getSelection().removeAllRanges(); getSelection().addRange(r);
      return onSelect(L.items[L.i], $("label-text"));
    }
  };
  try {
    for (let k = 0; k < 100 && !L.items.length; k++) await wait(50);
    show("label");
    // A message the detectors flagged; the text of their first span is what gets selected.
    goTo(L.items.findIndex((x) => (L.suggestions[x.item_id] || []).length));
    const it = L.items[L.i], first = L.suggestions[it.item_id][0];
    const NEEDLE = Array.from(it.text).slice(first.start, first.end).join(""); out.needle = NEEDLE;
    select(NEEDLE); out.selected = !!L.selection;
    press("2"); out.about = L.about;
    press("n"); await wait(300);
    out.spans = it.spans.map((s) => [Array.from(it.text).slice(s.start, s.end).join(""), s.pii_type, s.about, s.after_reveal]);
    out.everywhere = it.spans.length === it.text.split(NEEDLE).length - 1;
    press("z"); await wait(300); out.afterUndo = it.spans.length;
    select(NEEDLE); press("n"); await wait(300);
    // An inferred attribute has no text to select: it goes in through the attribute panel.
    L.draft.evidence = "inferred"; L.draft.type = "education_level"; L.draft.about = "user";
    [...document.querySelectorAll("button")].find((b) => b.textContent === "Add attribute").click(); await wait(300);
    out.attrs = it.attrs.map((a) => [a.type, a.evidence, a.about, a.after_reveal]);
    press("r"); out.compareShown = !!L.revealed[it.item_id];
    out.suggested = L.suggestions[it.item_id].length; out.unmatched = unmatched(it).length;
    const before = L.i; press("Enter"); out.advanced = L.i !== before;
    const empty = L.items[L.i]; out.emptyId = empty.item_id;
    press("r"); out.blindFirst = !L.revealed[empty.item_id];
    press("n"); await wait(300); out.none = empty.none; out.movedOn = L.items[L.i].item_id !== empty.item_id;
    out.id = it.item_id;
  } catch (e) { out.error = String(e && e.stack || e); }
  document.body.append(h("pre", { id: "testout" }, JSON.stringify(out)));
})();
"""


def test_keys_label_undo_and_compare(labeled, tmp_path, monkeypatch):
    browser, module = next((b for b in BROWSERS if Path(b).exists()), None), _puppeteer()
    if browser is None or module is None:
        pytest.skip("needs node with puppeteer and a Chromium-family browser")
    pipe, path, _, _ = labeled
    page, runner = tmp_path / "ui.html", tmp_path / "run.js"
    html = review.UI.read_text(encoding="utf-8")
    page.write_text(html.replace("</script>", DRIVER + "</script>"), encoding="utf-8")
    runner.write_text(RUNNER)
    monkeypatch.setattr(review, "UI", page)
    server, url = review.make_server(review.Review(pipe))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        got = subprocess.run(["node", str(runner), module, browser, url], capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
        server.server_close()
    line = got.stdout.strip().splitlines()[-1] if got.stdout.strip() else ""
    assert line.startswith("{"), f"the page did not finish its script: {line[:200]}"
    out = json.loads(line)
    assert "error" not in out, out.get("error")
    assert out["selected"] and out["about"] == "third_party"
    assert out["spans"] and all(s == [out["needle"], "name", "third_party", False] for s in out["spans"])
    assert out["everywhere"] and out["afterUndo"] == 0
    assert out["compareShown"] and out["suggested"] >= 1 and out["unmatched"] < out["suggested"]
    assert out["advanced"] and out["blindFirst"] and out["none"] and out["movedOn"]
    # What the keys did is what is on disk.
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    mine = rows[rows["item_id"] == out["id"]]
    assert set(mine["span_text"]) == {out["needle"]} and set(mine["pii_type"]) == {"name"}
    assert set(mine[review.AFTER_REVEAL]) == {""}
    assert out["attrs"] == [["education_level", "inferred", "user", False]]
    attrs = label.read_attribute_labels(path)
    assert attrs[["item_id", "attribute_type", "evidence", "about"]].values.tolist() == [
        [out["id"], "education_level", "inferred", "user"]]
    assert rows.loc[rows["item_id"] == out["emptyId"], "pii_type"].tolist() == ["none"]
