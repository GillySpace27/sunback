# SB-8 hand-off to the Website register: live pulse, freshness badges, honest integration copy, deep links, Share

Producer: sunback SB-8 (`status.json`, written by the video Lambda beside
`manifest/index.json`). Consumer: Website `sun.html`, through WS-5 (card
contract, `assets/sun.js`) and WS-15 (stage). Written 2026-10-01 against
Website@c272b93.

## What the page reads

- `status.json` (new, public, `Cache-Control: public, max-age=60`):
  `{"generated": str, "products": [{"id": str, "updated": str, "age_s": int}], "worst_age_s": int | null}`.
- `manifest/<id>.json` fields already there: `updated` (ISO, UTC) and
  `integration` (`{"frames": int, "method": str}`), which replaces the
  hard-coded "median of the three most recent exposures" (`sun.html:118`;
  production integrates 5 frames, `deploy.py:15`).

## Behaviour

- Live dot plus polite live text in the eyebrow (`sun.html:81`), from the
  newest product age in `status.json`, refreshed every 2 minutes: green and
  pulsing when fresh, amber "Delayed" after 40 min, grey "Not updating right
  now" after 2 h, "Live status unavailable" when `status.json` cannot be read.
  The pulse stops under `prefers-reduced-motion: reduce`.
- Per-card badge from that card's `updated`: none when fresh, "delayed" (amber)
  after 40 min, "not updating" (grey) after 2 h.
- Thresholds are estimates to tune after a week of `status.json`.
  Decision (Gilly, 2026-10-02, decision sheet A5): two tiers. 40 min and 2 h
  is the internal alarm tier (the constants in the patch below, used by the
  page before WS-5 lands); 60 and 180 min is the public badge tier, WS-5's
  `SunData.freshness`. Once WS-5 has landed, call `SunData.freshness(updated)`
  and map `"ok" | "stale" | "old"` to the same classes instead of `freshClass`,
  so the public page shows 60 and 180 min.
  Which time the ages count: observation time is carried in the new S3 metadata key
  and fragment field `obs_end` (SB-9). `obstime`, and so `updated` and the age in
  `status.json`, stays the reducer's upload time until Gilly decides to switch
  (open question), so the thresholds above (40 min / 2 h alarm, 60 / 180 min badge)
  are measured on upload time until then.
- `sun.html#<id>` (lowercase ids from `PRODUCTS`) opens that card's lightbox
  and outlines the card; `hashchange` does the same. Same form as SU-17.
- Share button per card: `navigator.share({title, url})`, else copy the
  `#<id>` URL; the button reads "Link copied" or "Copy failed" for 2 s. If
  WS-16 has landed, call `SunData.share(...)` instead.

## Patch (today's page, before WS-5 moves the script to `assets/sun.js`)

Run from the Website repository root. It never touches the `PRODUCTS` block
(lines 141-148).

```python
"""SB-8 page half for Website: sun.html (Live dot, freshness badges, integration copy, deep links, Share).

Written against Website@c272b93 sun.html. Run from the Website repository root.
Patches by exact match; AssertionError when a match count is not 1. Never edits
the PRODUCTS block (lines 141-148), which HG-4 and SB-6 parse.
"""
import pathlib

p = pathlib.Path("sun.html")
text = p.read_text(encoding="utf-8")


def swap(old, new):
    global text
    n = text.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:70]!r}"
    text = text.replace(old, new)


# 1. styles: dot, badges, share feedback; motion only when the reader allows it
swap('''  @media (max-width: 560px){ .lb-nav { font-size: 24px; padding: 10px 12px; } }
</style>''', '''  @media (max-width: 560px){ .lb-nav { font-size: 24px; padding: 10px 12px; } }
  /* SB-8: live pulse and per-card freshness (thresholds are estimates; tune after a week) */
  .live-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
              margin-right: 7px; vertical-align: 1px; background: var(--muted); }
  .live-dot.ok { background: #2e9d4f; animation: sun-pulse 2.4s ease-in-out infinite; }
  .live-dot.stale { background: #d08a00; }
  .live-dot.old { background: var(--muted); }
  @keyframes sun-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
  @media (prefers-reduced-motion: reduce) { .live-dot.ok { animation: none; } }
  .fresh-badge { font: 600 11px/1 inherit; border-radius: 5px; padding: 3px 6px;
                 margin-left: 6px; vertical-align: 2px; }
  .fresh-badge.stale { background: #d08a00; color: #fff; }
  .fresh-badge.old { background: var(--muted); color: var(--card-bg); }
  .card--sun.is-hl { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>''')

# 2. eyebrow becomes the live line (dot plus polite live text)
swap('''    <p class="eyebrow">Live · updates every 20 minutes</p>''',
     '''    <p class="eyebrow"><span id="live-dot" class="live-dot" aria-hidden="true"></span><span id="live-text" aria-live="polite">Live · updates every 20 minutes</span></p>''')

# 3. integration copy filled from the manifest instead of the hard-coded "three"
swap('''each still is the <b>median of the three most recent exposures</b>''',
     '''each still is the <b id="integration-copy">median of the most recent exposures</b>''')

# 4. card state: keep the id with each lightbox item; card id for #<id> links
swap('''        ITEMS.push({ label, img: u(m.img1k, v), video: u(m.video, v) });''',
     '''        ITEMS.push({ id, label, img: u(m.img1k, v), video: u(m.video, v) });
        el.id = id;
        if (m.integration && m.integration.frames) setIntegrationCopy(m.integration);''')
swap('''            <h3>${label}</h3>''', '''            <h3>${label}${badgeHtml(m.updated)}</h3>''')
swap('''              ${id==="dem" ? `<button class="ghost" onclick="openTscan('${u(TSCAN_KEY,v)}')">&#9658; T-scan</button>` : ""}''',
     '''              ${id==="dem" ? `<button class="ghost" onclick="openTscan('${u(TSCAN_KEY,v)}')">&#9658; T-scan</button>` : ""}
              <button class="ghost" onclick="shareCard('${id}', this)" aria-label="Share ${label}">Share</button>''')

# 5. after the cards: deep link, status pulse
swap('''    loadTimes();
  })();''', '''    loadTimes();
    openFromHash();
    window.addEventListener("hashchange", openFromHash);
    loadStatus();
    setInterval(loadStatus, 120000);
  })();''')

# 6. helpers, defined before the card loop runs
swap('''  async function loadTimes(){''', '''  // --- SB-8 helpers -----------------------------------------------------------
  const FRESH_AMBER_S = 40 * 60;   // estimated: two missed 20-minute runs
  const FRESH_GREY_S = 2 * 3600;   // estimated: dispatcher and hourly fallback both missed
  function freshClass(ageS){
    if (ageS == null || !isFinite(ageS)) return "old";
    return ageS > FRESH_GREY_S ? "old" : (ageS > FRESH_AMBER_S ? "stale" : "ok");
  }
  function ageOf(iso){ const t = Date.parse(iso); return isNaN(t) ? null : Math.max(0, (Date.now() - t) / 1000); }
  function badgeHtml(updated){
    const c = freshClass(ageOf(updated));
    if (c === "ok") return "";
    return ` <span class="fresh-badge ${c}" title="newest image ${humanElapsed((ageOf(updated) || 0) * 1000)}">${c === "stale" ? "delayed" : "not updating"}</span>`;
  }
  const WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine","ten"];
  function setIntegrationCopy(integ){
    const n = integ.frames, how = integ.method || "median";
    const word = WORDS[n] || String(n);
    document.getElementById("integration-copy").textContent =
      `${how} of the ${word} most recent exposure${n === 1 ? "" : "s"}`;
  }
  async function loadStatus(){
    const dot = document.getElementById("live-dot"), txt = document.getElementById("live-text");
    try {
      const r = await fetch(u("status.json", Date.now()), { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const s = await r.json();
      const newest = Math.min(...s.products.map(p => ageOf(p.updated)).filter(a => a != null));
      const c = freshClass(isFinite(newest) ? newest : null);
      dot.className = "live-dot " + c;
      txt.textContent = c === "ok" ? `Live · newest image ${humanElapsed(newest * 1000)}`
        : (c === "stale" ? `Delayed · newest image ${humanElapsed(newest * 1000)}`
                         : "Not updating right now · showing the last images we have");
    } catch (e) {
      dot.className = "live-dot old";
      txt.textContent = "Live status unavailable · updates every 20 minutes";
    }
  }
  function openFromHash(){
    const id = decodeURIComponent((location.hash || "").slice(1));
    const i = ITEMS.findIndex(it => it.id === id);
    if (i < 0) return;
    document.querySelectorAll(".card--sun.is-hl").forEach(c => c.classList.remove("is-hl"));
    const card = document.getElementById(id);
    if (card) { card.classList.add("is-hl"); card.scrollIntoView({ block: "center" }); }
    openLB("img", i);
  }
  async function shareCard(id, btn){
    const it = ITEMS.find(x => x.id === id);
    const url = location.origin + location.pathname + "#" + encodeURIComponent(id);
    try {
      if (navigator.share) { await navigator.share({ title: it ? it.label : "The Sun, right now", url }); return; }
      await navigator.clipboard.writeText(url);
      btn.textContent = "Link copied";
    } catch (e) {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => { btn.textContent = "Share"; }, 2000);
  }

  async function loadTimes(){''')

p.write_text(text, encoding="utf-8")
print("patched", p)
```

Syntax check (computed 2026-10-01 with Node 22 on the patched page):
`python3 -c "import re;t=open('sun.html').read();open('/tmp/p.js','w').write(re.findall(r'<script>\n(.*?)</script>',t,re.S)[-1])" && node --check /tmp/p.js` exits 0.

## Phone check (the Website implementer)

1. iPhone Safari and an Android phone, light and dark: the dot and live text
   fit on one line under 375 px; badges do not wrap the card title.
2. With Reduce Motion on, the dot does not pulse.
3. VoiceOver reads the live text once on load and again only when it changes.
4. `https://gilly.space/sun.html#304` opens the AIA 304 lightbox; Share on a
   phone opens the share sheet; on a desktop it copies the link.
5. A staging check: point a local copy's `BUCKET` at
   `https://the-sun-now.s3.us-east-2.amazonaws.com/staging/` after SB-8 Task 10
   Step 7 edits `staging/manifest/171.json`; the 171 card shows "delayed".
