# SB-9 hand-off to the Website register: "How this was made" drawer with the observation window and Cite line

Producer: sunback SB-9. Consumer: Website `sun.html` through WS-17 (provenance
drawer). Written 2026-10-01 against Website@c272b93 with the SB-8 hand-off
applied.

## What the page reads

- `manifest/<id>.json`: new optional `obs_start`, `obs_end` (ISO, UTC);
  existing `updated`, `integration` (`{"frames", "method"}`), `through`,
  `frame_count`.
- `meta/rhef_<id>.json` (schema.org `ImageObject`, optional for the page):
  `additionalProperty` entries `observationStartUTC`, `observationDateUTC`,
  `obstimeSource`, `integrationFrames`, `integrationMethod`,
  `sunkitImageVersion`, `sunbackVersion`.

## Behaviour

- "Observed 11:52 to 12:00 UTC" from `obs_start`/`obs_end`. Only when they are
  absent does the drawer show `updated` with the flag "upload time; the image
  header carried no observation time" (honesty: say unknown, never guess).
- "Combined: median of 5 exposures" from `integration`.
- "Timelapse: 48 hours through <through>, <frame_count> distinct frames".
- RHEF note, Cite line and credit, wording for Gilly to confirm (Q18), the
  same strings WS-17 fixes as `SunData` constants:
  `Gilly and Cranmer 2025, Solar Physics, doi:10.1007/s11207-025-02578-x`;
  `Imagery courtesy of NASA/SDO and the AIA science team.`;
  `Processed with RHEF (Gilly and Cranmer 2025, Solar Physics); visualization, not a calibrated radiance.`
- If WS-17 has built `<dialog id="sun-info">` and `openInfo(id)`, do not run
  the patch: use `provenanceHtml(m)` from it as the dialog body.

## Patch (today's page with the SB-8 patch applied)

```python
"""SB-9 page half for Website: sun.html "How this was made" drawer with the observation window and Cite line.

Written against Website@c272b93 sun.html with the SB-8 handoff patch applied
(docs/handoffs/SB-8-website.md). Run from the Website repository root. Patches
by exact match; AssertionError when a match count is not 1. Never edits the
PRODUCTS block. If WS-17 has already built <dialog id="sun-info">, do not run
this: feed provenanceHtml(m) into WS-17's openInfo(id) instead.
"""
import pathlib

p = pathlib.Path("sun.html")
text = p.read_text(encoding="utf-8")


def swap(old, new):
    global text
    n = text.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:70]!r}"
    text = text.replace(old, new)


swap('''  .card--sun.is-hl { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>''', '''  .card--sun.is-hl { outline: 2px solid var(--accent); outline-offset: 3px; }
  /* SB-9: provenance drawer */
  #sun-info { max-width: min(560px, 92vw); border: 1px solid var(--card-border); border-radius: 10px;
              background: var(--card-bg); color: var(--text); padding: 18px 20px; }
  #sun-info::backdrop { background: rgba(0,0,0,.55); }
  #sun-info dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0 0 12px; }
  #sun-info dt { color: var(--muted); }
  #sun-info .flag { color: #d08a00; font-weight: 600; }
  #sun-info .cite { font-size: 13px; color: var(--muted); }
</style>''')

swap('''<div data-include="footer"></div>''', '''<dialog id="sun-info" aria-labelledby="sun-info-title">
  <h2 id="sun-info-title" style="margin-top:0">How this was made</h2>
  <div id="sun-info-body"></div>
  <form method="dialog"><button class="ghost">Close</button></form>
</dialog>

<div data-include="footer"></div>''')

swap('''        ITEMS.push({ id, label, img: u(m.img1k, v), video: u(m.video, v) });''',
     '''        ITEMS.push({ id, label, img: u(m.img1k, v), video: u(m.video, v), m });''')

swap('''              <button class="ghost" onclick="shareCard('${id}', this)" aria-label="Share ${label}">Share</button>''',
     '''              <button class="ghost" onclick="shareCard('${id}', this)" aria-label="Share ${label}">Share</button>
              <button class="ghost" onclick="openInfo('${id}')">How this was made</button>''')

swap('''  async function loadTimes(){''', '''  // --- SB-9: provenance drawer ---------------------------------------------------
  // Wording for Gilly to confirm (overview Q18); WS-17 uses the same constants.
  const CITATION = "Gilly and Cranmer 2025, Solar Physics, doi:10.1007/s11207-025-02578-x";
  const CREDIT = "Imagery courtesy of NASA/SDO and the AIA science team.";
  const RHEF_NOTE = "Processed with RHEF (Gilly and Cranmer 2025, Solar Physics); visualization, not a calibrated radiance.";
  function utcText(iso){
    const t = new Date(iso);
    return isNaN(t) ? "unknown" : t.toLocaleString("en-US", { timeZone: "UTC", dateStyle: "medium", timeStyle: "short" }) + " UTC";
  }
  function provenanceHtml(m){
    const hasWindow = m.obs_start && m.obs_end;
    const when = hasWindow
      ? `${utcText(m.obs_start)} to ${utcText(m.obs_end)}`
      : `${utcText(m.updated)} <span class="flag">(upload time; the image header carried no observation time)</span>`;
    const integ = m.integration && m.integration.frames
      ? `${m.integration.method || "median"} of ${m.integration.frames} exposures` : "unknown";
    const movie = m.through ? `48 hours through ${utcText(m.through)}, ${m.frame_count} distinct frames` : "unknown";
    return `<dl>
        <dt>Observed</dt><dd>${when}</dd>
        <dt>Combined</dt><dd>${integ}</dd>
        <dt>Timelapse</dt><dd>${movie}</dd>
        <dt>Instrument</dt><dd>NASA SDO / AIA, synoptic near-real-time data from JSOC</dd>
      </dl>
      <p>${RHEF_NOTE}</p>
      <p class="cite">Cite: ${CITATION}<br>${CREDIT}</p>`;
  }
  function openInfo(id){
    const it = ITEMS.find(x => x.id === id);
    if (!it) return;
    document.getElementById("sun-info-title").textContent = `How this was made: ${it.label}`;
    document.getElementById("sun-info-body").innerHTML = provenanceHtml(it.m);
    document.getElementById("sun-info").showModal();
  }

  async function loadTimes(){''')

p.write_text(text, encoding="utf-8")
print("patched", p)
```

Syntax check (computed 2026-10-01, Node 22, after the SB-8 and SB-9 patches):
`node --check` on the page's last inline script exits 0.

## Phone check (the Website implementer)

1. iPhone Safari and Android, light and dark: the dialog fits 375 px, the
   definition list does not overflow, Close works, Escape closes on desktop.
2. Before SB-9's Lambda deploy, the drawer shows the "upload time" flag; after
   it, every AIA card shows a window that ends at or before `updated`.
3. VoiceOver announces the dialog title "How this was made: <label>".
