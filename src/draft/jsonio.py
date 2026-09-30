"""Writing committed JSON without committing floating-point noise.

Every stage that refits a trend or re-measures a twin writes a JSON file that is
also committed, and several are read back as inputs: `link_trends.json` is one
of the files whose sha256 pins the paper's machines. A refit on another machine
lands on the same numbers to the last bit or two — a different BLAS sums in a
different order — and rewriting the file for that would dirty the checkout and
break the pin over nothing. So a file is rewritten only when its content has
really changed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Relative difference below which two floats are the same measurement. Refits
#: across BLAS builds differ at ~1e-14; anything a trend or a twin could mean
#: moves far more than this.
RTOL = 1e-12
#: Absolute floor, for values that are zero up to rounding: a joint on the
#: axis comes out 0.0 on one machine and 7e-18 on another, which no relative
#: tolerance can call equal. Every quantity here is in kg, m or s.
ATOL = 1e-15


def same(a: Any, b: Any, rtol: float = RTOL, ignore: frozenset[str] = frozenset(),
         atol: float = ATOL) -> bool:
    """Equal apart from float noise and the keys in ``ignore`` (at any depth)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - ignore
        return all(k in a and k in b and same(a[k], b[k], rtol, ignore, atol) for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rtol, ignore, atol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
    return a == b


def write_json(path: str | Path, obj: Any, *, indent: int | None = None,
               trailing_newline: bool = False,
               ignore: frozenset[str] = frozenset()) -> bool:
    """Write ``obj`` to ``path`` unless the file already holds the same content.

    Returns whether the file was written. ``ignore`` names keys, such as a
    measurement date, whose change alone is not a reason to rewrite.
    """
    path = Path(path)
    if path.exists():
        try:
            if same(json.loads(path.read_text()), json.loads(json.dumps(obj)),
                    ignore=ignore):
                return False
        except ValueError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent) + ("\n" if trailing_newline else ""))
    return True
