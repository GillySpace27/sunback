"""RHEF conformance (RH-6): sunkit_image.radial.rhef against the fastRHEF golden bundle.

Every case in the bundle (fastRHEF golden/, bundle 1.0.0 or later) is rebuilt as a sunpy Map
from its input.f64 and header.json and filtered by sunkit_image.radial.rhef with the keywords
sunback's RHEFProcessor passes (method="scipy", vignette=None) and the case's own edges,
Upsilon, application radius and fill. The result is compared with expected_sunkit-0.7.f64.
One RHEF-CONFORMANCE line per case and a summary line.

    RHEF_GOLDEN_DIR=<fastRHEF>/golden python -m unittest sunback.processor.tests.test_rhef_conformance
    RHEF_CONFORMANCE_MODE=enforce RHEF_GOLDEN_DIR=... python -m unittest sunback.processor.tests.test_rhef_conformance

Where the bundle comes from: RHEF_GOLDEN_DIR, else sunback/processor/tests/golden. That folder is
not committed: fastRHEF is private and copying its bundle into this public repository
(fastRHEF tools/sync_golden.sh) waits on Gilly's yes. Until then CI skips this test.

Report mode (the default) never fails on a number; enforce mode fails on any FAIL line. Both
fail when the bundle does not match its manifest. Without a bundle, report mode skips and
enforce mode fails. A case's result: PASS within the case's tol_f64; otherwise REPORT in report
mode or when the case's sensitive_to names one of DECLARED; otherwise FAIL.
RHEF output is a visualization, not a calibrated radiance.
"""
import hashlib
import importlib.util
import inspect
import json
import math
import os
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.environ.get("RHEF_GOLDEN_DIR") or os.path.join(HERE, "golden")
MODE = os.environ.get("RHEF_CONFORMANCE_MODE", "report")
IMPL = "sunback-processor"
CONVENTION = "sunkit-0.7"
# The deviations of the sunback-processor row in fastRHEF conventions/implementations.json.
DECLARED = frozenset()


def _read_golden():
    spec = importlib.util.spec_from_file_location("read_golden", os.path.join(GOLDEN, "read_golden.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_for(diff, tol, sensitive, mode=None):
    mode = mode or MODE
    if not math.isnan(diff) and diff <= tol:
        return "PASS"
    return "REPORT" if mode == "report" or sensitive & DECLARED else "FAIL"


class TestResultRule(unittest.TestCase):
    """The PASS / REPORT / FAIL rule, without a bundle (runs everywhere)."""

    def test_rule(self):
        none = frozenset()
        self.assertEqual(result_for(0.0, 0.0, none, "enforce"), "PASS")
        self.assertEqual(result_for(1e-3, 0.0, none, "report"), "REPORT")
        self.assertEqual(result_for(1e-3, 0.0, none, "enforce"), "FAIL")
        self.assertEqual(result_for(math.nan, 1.0, none, "enforce"), "FAIL")


class TestRhefConformance(unittest.TestCase):

    def test_golden_bundle(self):
        self.assertIn(MODE, ("report", "enforce"), "RHEF_CONFORMANCE_MODE must be report or enforce")
        manifest_path = os.path.join(GOLDEN, "manifest.json")
        if not os.path.isfile(manifest_path):
            if MODE == "enforce":
                self.fail(f"no RHEF golden bundle at {GOLDEN}")
            self.skipTest(f"no RHEF golden bundle at {GOLDEN}; set RHEF_GOLDEN_DIR or copy it with "
                          "fastRHEF tools/sync_golden.sh (Gilly's yes first)")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        for rel, digest in manifest["files"].items():
            with open(os.path.join(GOLDEN, rel), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), digest, f"{rel} does not match the manifest")

        import numpy as np
        import astropy.units as u
        import sunpy.map
        from sunkit_image.radial import rhef
        takes_fill = "fill" in inspect.signature(rhef).parameters

        counts = {"PASS": 0, "REPORT": 0, "FAIL": 0}
        for case_id, case in _read_golden().iter_cases(GOLDEN):
            p, a = case["props"], case["arrays"]
            ny, nx = (int(x) for x in p["shape"].split(","))
            smap = sunpy.map.Map(np.array(a["input.f64"]).reshape(ny, nx), case["header"])
            edges = np.array(a["edges.f64"]).reshape(2, int(p["nbins"])) * u.R_sun
            if p["upsilon"] == "none":
                upsilon = None
            else:
                lo, hi = (float(x) for x in p["upsilon"].split(","))
                upsilon = lo if lo == hi else (lo, hi)
            kwargs = dict(radial_bin_edges=edges, application_radius=float(p["application_radius"]) * u.R_sun,
                          upsilon=upsilon, method="scipy", vignette=None)
            if takes_fill:
                kwargs["fill"] = math.nan if p["fill"] == "nan" else float(p["fill"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = np.asarray(rhef(smap, **kwargs).data, dtype=np.float64).ravel()
            want = np.array(a[f"expected_{CONVENTION}.f64"])
            nan_got, nan_want = np.isnan(got), np.isnan(want)
            if (nan_got != nan_want).any():
                diff = math.nan
            else:
                finite = ~nan_got
                diff = float(np.max(np.abs(got[finite] - want[finite]))) if finite.any() else 0.0
            sensitive = frozenset(p["sensitive_to"].split(","))
            result = result_for(diff, float(p["tol_f64"]), sensitive)
            counts[result] += 1
            shown = "nan" if math.isnan(diff) else f"{diff:.3e}"
            print(f"RHEF-CONFORMANCE impl={IMPL} bundle={manifest['bundle_version']} case={case_id} "
                  f"convention={CONVENTION} max_abs_diff={shown} result={result}")
        print(f"RHEF-CONFORMANCE impl={IMPL} summary pass={counts['PASS']} report={counts['REPORT']} "
              f"fail={counts['FAIL']} mode={MODE}")
        if not takes_fill:
            print("NOTE: this sunkit_image.radial.rhef takes no fill keyword; unbinned pixels follow its own default")
        self.assertGreater(sum(counts.values()), 0, f"no cases in {GOLDEN}")
        self.assertEqual(counts["FAIL"], 0, f"{counts['FAIL']} case(s) FAIL in enforce mode")


if __name__ == "__main__":
    unittest.main()
