"""Loader for the hand-maintained catalogue under ``datasets/actuators/data/``.

One row per part, in data/actuators/{qdd,midgear,harmonic}.yaml (33 + 33 + 48).
``data/derived/`` is script-written; do not hand-edit it. This module only
reads, validates and normalises units.
"""

import datetime
import glob
import math
import os

import yaml

from draft.paths import repo_root as _repo_root
from draft.robots import ROBOTS_DIR as _ROBOTS_DIR
from draft.trends import ACTUATOR_TRENDS_JSON as _A_T
_REPO = _repo_root()
_TRENDS_DIR = _A_T.parent

from draft.paths import repo_root as _repo_root  # noqa: E402

ROOT = os.path.join(str(_repo_root()), "datasets", "actuators")
DATA = os.path.join(ROOT, "data")
ACTUATORS = os.path.join(DATA, "actuators")
DERIVED = os.path.join(DATA, "derived")

RPM = 2 * math.pi / 60

#: file stem -> the ``category`` every row in it carries.
CLASS_FILES = {
    "qdd": "QDD", "midgear": "MidGear", "harmonic": "Harmonic",
}

#: accepted rotor-inertia units -> kg.m^2 factor.
INERTIA_UNITS = {"g.cm2": 1e-7, "kg.cm2": 1e-4, "g.mm2": 1e-9, "kg.m2": 1.0}

REQUIRED = ("name", "vendor", "tau_peak_Nm", "mass_kg")


def derived(*parts):
    """Path inside ``data/derived/``, creating it on first use."""
    os.makedirs(DERIVED, exist_ok=True)
    return os.path.join(DERIVED, *parts)


#: The package's trend files (src/draft/trends/data/); these stages write them.
TRENDS_DATA = str(_TRENDS_DIR)


def trends(name):
    """Path to a trend file the package ships."""
    os.makedirs(TRENDS_DATA, exist_ok=True)
    return os.path.join(TRENDS_DATA, name)


def _read(path, key):
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    rows = doc.get(key)
    if rows is None:
        raise ValueError(f"{path}: expected a top-level '{key}:' list")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: '{key}' must be a list, found {type(rows).__name__}")
    return rows


def _omega(row, where):
    """rad/s from ``no_load_rpm`` or ``omega_NL_rad_s``; exactly one must be set."""
    has_rpm = row.get("no_load_rpm") is not None
    has_rad = row.get("omega_NL_rad_s") is not None
    if has_rpm and has_rad:
        raise ValueError(f"{where}: set no_load_rpm OR omega_NL_rad_s, not both")
    if has_rpm:
        return float(row["no_load_rpm"]) * RPM
    if has_rad:
        return float(row["omega_NL_rad_s"])
    raise ValueError(f"{where}: needs no_load_rpm or omega_NL_rad_s")


def checked_on(row, where=""):
    """'YYYY-MM-DD' a person last verified this row against ``source``, or None.

    Accepts a YAML date or a string. Clear it whenever another field changes.
    """
    v = row.get("checked")
    if v in (None, "", False):
        return None
    if isinstance(v, datetime.date):
        return v.isoformat()
    s = str(v).strip()
    try:
        datetime.date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{where}: checked must be a YYYY-MM-DD date, got {s!r}")
    return s


def rotor_inertia(row, gear=None, where=""):
    """(J_rotor kg.m^2, kind, source) from a row's ``rotor_inertia:`` block.

    ``kind: output`` values are output-side and divided by N^2 to land motor-side;
    without a gear ratio they are dropped (None).
    """
    e = row.get("rotor_inertia")
    if not e:
        return None, None, None
    for k in ("value", "unit", "kind"):
        if k not in e:
            raise ValueError(f"{where}: rotor_inertia needs a '{k}'")
    if e["unit"] not in INERTIA_UNITS:
        raise ValueError(f"{where}: rotor_inertia unit {e['unit']!r} is not one of "
                         f"{sorted(INERTIA_UNITS)}")
    if e["kind"] not in ("rotor", "output"):
        raise ValueError(f"{where}: rotor_inertia kind must be 'rotor' or 'output'")
    j = float(e["value"]) * INERTIA_UNITS[e["unit"]]
    if e["kind"] == "output":
        n = gear if gear is not None else row.get("gear")
        if not n:
            return None, None, None
        j /= float(n) ** 2
    return j, e["kind"], e.get("source")


def actuators(categories=None):
    """Every catalogued actuator, as dicts, in file order.

    `categories` filters by the ``category`` field (e.g. ``{"QDD", "MidGear"}``).
    """
    out = []
    for stem, cat in CLASS_FILES.items():
        if categories is not None and cat not in categories:
            continue
        path = os.path.join(ACTUATORS, f"{stem}.yaml")
        for i, row in enumerate(_read(path, "actuators")):
            where = f"{os.path.basename(path)}[{i}] {row.get('name', '?')}"
            for k in REQUIRED:
                if row.get(k) in (None, ""):
                    raise ValueError(f"{where}: missing required field {k!r}")
            r = dict(row)
            r["category"] = cat
            r["source_file"] = f"actuators/{stem}.yaml"
            r["tau_peak_Nm"] = float(row["tau_peak_Nm"])
            r["mass_kg"] = float(row["mass_kg"])
            r["omega_NL_rad_s"] = _omega(row, where)
            r["no_load_rpm_output"] = (float(row["no_load_rpm"])
                                       if row.get("no_load_rpm") is not None else None)
            r["P_peak_published_W"] = (float(row["P_peak_W"])
                                       if row.get("P_peak_W") is not None else None)
            r["gear"] = float(row["gear"]) if row.get("gear") is not None else None
            r["notes"] = row.get("notes") or ""
            r["rotor_is_geared_inrunner"] = bool(row.get("geared_inrunner"))
            r["checked"] = checked_on(row, where)
            j, kind, src = rotor_inertia(row, r["gear"], where)
            r["rotor_inertia_kgm2"] = j
            r["rotor_inertia_kind"] = kind
            r["rotor_inertia_source"] = src
            out.append(r)
    return out


def dims(rows):
    """{name: (OD_mm, L_mm)} as floats, for rows with a published envelope."""
    return {r["name"]: (float(r["OD_mm"]), float(r["L_mm"])) for r in rows
            if r.get("OD_mm") is not None and r.get("L_mm") is not None}


def files():
    """Every hand-maintained source file, for provenance printouts."""
    return sorted(glob.glob(os.path.join(ACTUATORS, "*.yaml")))


# ── writing ──────────────────────────────────────────────────────────────────
# Regenerates only the row block; the file's header comments are kept verbatim.

#: emission order. Unlisted keys are dropped on save.
FIELD_ORDER = ("name", "vendor", "source", "tau_peak_Nm", "mass_kg", "no_load_rpm",
               "omega_NL_rad_s", "P_peak_W", "gear", "OD_mm", "L_mm",
               "rotor_inertia", "geared_inrunner", "inner_motor", "notes", "checked")

#: keys omitted when falsy rather than written as `null`.
SKIP_IF_FALSY = ("geared_inrunner", "checked")
ROTOR_ORDER = ("value", "unit", "kind", "source")
MOTOR_ORDER = ("name", "kind", "stall_mNm", "rotor_inertia_gcm2", "mass_g")

#: comments re-attached to null fields on save.
HINTS = {
    "P_peak_W": "vendor-published MECHANICAL peak; null -> tau*w/4",
    "source": "TODO the page these values came from",
    "OD_mm": "TODO envelope not in the text; check the drawing",
    "L_mm": "TODO envelope not in the text; check the drawing",
    "rotor_inertia": "TODO not in the text of the vendor page",
}


def _scalar(v):
    """YAML scalar that reads back as `v` (quoted if needed; YAML 1.1 reads ``8:1`` as 481)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)
    if isinstance(v, int):
        return repr(v)
    s = str(v)
    if s != "":
        try:
            if yaml.safe_load("k: " + s)["k"] == s:
                return s
        except Exception:
            pass
    return "'" + s.replace("'", "''") + "'"


def _row_yaml(row):
    out = []
    for k in FIELD_ORDER:
        if k not in row:
            continue
        v = row[k]
        if k in SKIP_IF_FALSY and not v:
            continue
        if k in ("rotor_inertia", "inner_motor"):
            if not v:
                if k == "rotor_inertia":
                    line = "    rotor_inertia: null"
                    out.append(line + " " * max(1, 30 - len(line))
                               + f"# {HINTS['rotor_inertia']}")
                continue
            out.append(f"    {k}:")
            for kk in (ROTOR_ORDER if k == "rotor_inertia" else MOTOR_ORDER):
                if kk in v and v[kk] is not None:
                    out.append(f"      {kk}: {_scalar(v[kk])}")
            continue
        line = f"    {k}: {_scalar(v)}"
        if k in HINTS and v is None:
            line += " " * max(1, 26 - len(line)) + f"# {HINTS[k]}"
        out.append(line)
    out[0] = "  - " + out[0].lstrip()[2:] if False else "  - " + out[0].strip()
    return out


def save_actuators(stem, rows):
    """Rewrite one class file from `rows`, keeping its header comments.

    Writes nothing unless the new text reads back as `rows`.
    """
    path = os.path.join(ACTUATORS, f"{stem}.yaml")
    with open(path, encoding="utf-8") as fh:
        old = fh.read()
    head = old[:old.index("actuators:")] if "actuators:" in old else ""

    body = ["actuators:"]
    for r in rows:
        body += _row_yaml(r) + [""]
    text = head + "\n".join(body).rstrip() + "\n"

    check = yaml.safe_load(text)
    got = check.get("actuators") or []
    if len(got) != len(rows):
        raise ValueError(f"{stem}: would write {len(got)} rows, expected {len(rows)}")
    for want, back in zip(rows, got):
        for k in FIELD_ORDER:
            if k not in want or (k in SKIP_IF_FALSY and not want[k]):
                continue
            if want[k] != back.get(k):
                raise ValueError(f"{stem}: {want.get('name')}.{k} would read back as "
                                 f"{back.get(k)!r}, not {want[k]!r}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)          # atomic: a crash mid-write cannot truncate the file
    return path


def raw_actuators(stem):
    """Rows of one class file exactly as written, with no normalisation."""
    return _read(os.path.join(ACTUATORS, f"{stem}.yaml"), "actuators")


def _help_and_exit() -> None:
    """`--help` on a stage prints what it does and stops, rather than running it.

    The stages take no arguments, so without this `01_fetch_descriptions.py
    --help` would clone the whole survey and rewrite the committed manifest.
    """
    import sys
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        main = sys.modules.get("__main__")
        print((getattr(main, "__doc__", None) or "").strip() or "takes no arguments")
        raise SystemExit(0)


_help_and_exit()
