"""Where this dataset's inputs and outputs live. Imported by every stage."""
from __future__ import annotations

import os
from pathlib import Path

from draft.paths import repo_root
from draft.trends import ACTUATOR_TRENDS_JSON

#: This dataset's directory.
ROOT = repo_root() / "datasets" / "robot_descriptions"

#: Artefacts the stages write and read (manifest, segments, decomposition, ...).
DATA = ROOT / "data"

#: Fitted coefficients, inside the package (src/draft/trends/data/); written by stage 5.
TRENDS_DATA = ACTUATOR_TRENDS_JSON.parent

#: Clones of the upstream descriptions (the `robot_descriptions` package's cache).
CACHE = Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE",
                            "~/.cache/robot_descriptions")).expanduser()

#: Vendors the catalogue does not carry, cloned straight from source.
EXTRA_CACHE = ROOT / "cache_extra"

#: Where each cache sits differs by machine, so committed data records a
#: description's path as `<cache>:<path inside it>` rather than an absolute one.
_CACHES = {"cache": CACHE, "cache_extra": EXTRA_CACHE}


def portable(path: str | Path) -> str:
    """`path` in the machine-independent form the committed data stores."""
    p = Path(path)
    for tag, root in _CACHES.items():
        if p.is_relative_to(root):
            return f"{tag}:{p.relative_to(root).as_posix()}"
    return str(path)


def local(path: str) -> str:
    """The absolute path on this machine for a path stored by `portable`."""
    tag, sep, rel = path.partition(":")
    return str(_CACHES[tag] / rel) if sep and tag in _CACHES else path


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
