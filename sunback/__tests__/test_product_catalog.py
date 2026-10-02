"""One product catalog: every copy of the product list agrees with manifest.PRODUCTS (SB-6).

``manifest.PRODUCTS`` (aws_lambda/video_builder/manifest.py) is canonical. Bound
here: ``serve_keys.SERVED_CHANNELS`` and its composite sources, the wave lists in
``NRTFitsFetcher.py`` (read with ``ast``, so the fetcher's network imports are not
needed), the ``PRODUCTS`` array in ``web/sun.html`` and the duplicated key
builders. Entries with a ``"kind"`` (SB-14 difference movies) are made in the
Lambda, not uploaded by the reducer, so the reducer and page comparisons skip them.

Set ``SUNBACK_SUN_HTML`` to another copy of the page (for example
``~/vscode/Website/sun.html``) to check that copy instead of ``web/sun.html``.
"""
import ast
import os
import re
from pathlib import Path

import pytest

from aws_lambda.video_builder import manifest
from sunback.putter import serve_keys

ROOT = Path(__file__).resolve().parents[2]
FETCHER = ROOT / "sunback" / "fetcher" / "NRTFitsFetcher.py"
SUN_HTML = Path(os.environ.get("SUNBACK_SUN_HTML") or ROOT / "web" / "sun.html").expanduser()


def fetcher_wave_lists(path=FETCHER):
    """``SERVE_WAVES`` and ``COMPOSITE_ONLY_WAVES`` exactly as written in the fetcher."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("SERVE_WAVES", "COMPOSITE_ONLY_WAVES"):
                found[name] = ast.literal_eval(node.value)
    return found["SERVE_WAVES"], found["COMPOSITE_ONLY_WAVES"]


def sun_html_ids(text):
    """Product ids from the ``const PRODUCTS = [ ... ];`` array of the page."""
    m = re.search(r"const PRODUCTS = \[(.*?)\];", text, re.S)
    assert m, "no `const PRODUCTS = [...]` array in the page"
    return re.findall(r'\[\s*"([^"]+)"\s*,', m.group(1))


def composite_ids():
    """The ids serve_keys gives the three composite sources (rainbow, UV, DEM)."""
    sources = (serve_keys.RAINBOW_SOURCE, serve_keys.UV_COMPOSITE_SOURCE, serve_keys.DEM_SOURCE)
    return {serve_keys.serve_id_for_local_png(f"{src}.png") for src in sources}


def catalog_problems(products, served_channels, serve_waves, composite_only, page_ids):
    """Every disagreement between the copies, as one sentence each ([] means none)."""
    ids = [p["id"] for p in products if "kind" not in p]
    served = set(served_channels.values()) | composite_ids()
    problems = []
    for pid in sorted(set(ids) - served):
        problems.append(f"PRODUCTS id {pid!r} is never uploaded by the reducer (serve_keys)")
    for pid in sorted(served - set(ids)):
        problems.append(f"the reducer uploads {pid!r} but manifest.PRODUCTS does not list it")
    for wave in list(serve_waves) + list(composite_only):
        if wave not in served_channels:
            problems.append(f"NRTFitsFetcher fetches wave {wave!r} but serve_keys.SERVED_CHANNELS does not map it")
    for wave in served_channels:
        if wave not in serve_waves and wave not in composite_only:
            problems.append(f"serve_keys.SERVED_CHANNELS maps wave {wave!r} but NRTFitsFetcher never fetches it")
    if list(page_ids) != ids:
        problems.append(f"page PRODUCTS ids {list(page_ids)} differ from manifest.PRODUCTS ids {ids}")
    return problems


def test_product_list_copies_agree():
    serve_waves, composite_only = fetcher_wave_lists()
    page_ids = sun_html_ids(SUN_HTML.read_text(encoding="utf-8"))
    assert catalog_problems(manifest.PRODUCTS, serve_keys.SERVED_CHANNELS,
                            serve_waves, composite_only, page_ids) == []


def test_aia_ids_are_the_served_single_channels():
    ids = [p["id"] for p in manifest.PRODUCTS]
    assert len(manifest.AIA_IDS) == len(set(manifest.AIA_IDS)) == 9
    assert set(manifest.AIA_IDS) == set(serve_keys.SERVED_CHANNELS.values())
    assert [i for i in ids if i in manifest.AIA_IDS] == list(manifest.AIA_IDS)


@pytest.mark.parametrize("pid", [p["id"] for p in manifest.PRODUCTS if "kind" not in p])
def test_duplicated_key_builders_agree(pid):
    assert serve_keys.s3_img_key(pid) == manifest.img1k_key(pid)
    assert serve_keys.s3_thumb_key(pid) == manifest.thumb_key(pid)


def test_drift_is_caught_when_a_served_channel_is_removed():
    serve_waves, composite_only = fetcher_wave_lists()
    channels = dict(serve_keys.SERVED_CHANNELS)
    del channels["0131"]
    problems = catalog_problems(manifest.PRODUCTS, channels, serve_waves, composite_only,
                                [p["id"] for p in manifest.PRODUCTS])
    assert problems == [
        "PRODUCTS id '131' is never uploaded by the reducer (serve_keys)",
        "NRTFitsFetcher fetches wave '0131' but serve_keys.SERVED_CHANNELS does not map it",
    ]


def test_drift_is_caught_in_the_page_copy():
    serve_waves, composite_only = fetcher_wave_lists()
    page_ids = [p["id"] for p in manifest.PRODUCTS if p["id"] != "dem"]
    problems = catalog_problems(manifest.PRODUCTS, serve_keys.SERVED_CHANNELS,
                                serve_waves, composite_only, page_ids)
    assert len(problems) == 1 and problems[0].startswith("page PRODUCTS ids")
