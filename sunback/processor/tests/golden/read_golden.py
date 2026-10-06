"""Read one fastRHEF golden RHEF case with the Python standard library only (RH-5).

    import read_golden
    case = read_golden.load_case("golden/ties_zero_fill_64")
    case["props"]["shape"]            # "64,64" (ny,nx), from case.properties
    case["arrays"]["input.f64"]       # array.array("d"), C order, ny*nx values
    case["header"]                    # dict of FITS cards from header.json

Arrays are raw little-endian, C order, no header: .f64 -> "d", .f32 -> "f",
.i32 -> "i" (int32; -1 in bin_index.i32 means the pixel is in no bin). They are
byteswapped on a big-endian host. This file travels with the bundle, so a
consumer imports it from its own copy of golden/.
"""
import array
import json
import os
import sys

_TYPES = {".f64": "d", ".f32": "f", ".i32": "i"}
assert array.array("i").itemsize == 4, "this platform's C int is not 32 bits"


def _properties(path):
    """The key=value lines of a java.util.Properties file (no escapes are used)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#!":
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def load_case(case_dir):
    """{"props": dict[str, str], "arrays": dict[str, array.array], "header": dict | None}"""
    props = _properties(os.path.join(case_dir, "case.properties"))
    arrays = {}
    for name in sorted(os.listdir(case_dir)):
        code = _TYPES.get(os.path.splitext(name)[1])
        if code is None:
            continue
        a = array.array(code)
        with open(os.path.join(case_dir, name), "rb") as fh:
            a.frombytes(fh.read())
        if sys.byteorder == "big":
            a.byteswap()
        arrays[name] = a
    header = None
    header_path = os.path.join(case_dir, "header.json")
    if os.path.isfile(header_path):
        with open(header_path, encoding="utf-8") as fh:
            header = json.load(fh)
    return {"props": props, "arrays": arrays, "header": header}


def iter_cases(bundle_dir):
    """(case_id, load_case(...)) for every folder holding a case.properties, sorted by name."""
    for name in sorted(os.listdir(bundle_dir)):
        case_dir = os.path.join(bundle_dir, name)
        if os.path.isfile(os.path.join(case_dir, "case.properties")):
            yield name, load_case(case_dir)
