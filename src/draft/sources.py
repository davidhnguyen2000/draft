"""Addresses of the vendor robot descriptions, and the code to fetch them.

No robot model is distributed with Draft; run ``scripts/setup_data.py``.
Fetching is best effort: a missing or moved source is reported, never raised.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The `robot_descriptions` package's default clone cache, reused if populated.
CACHE = Path("~/.cache/robot_descriptions").expanduser()


@dataclass
class Source:
    """One upstream description repository."""

    key: str
    #: "catalog" — a `robot_descriptions` entry, which clones on import.
    #: "clone"   — a repository the catalogue does not carry, cloned directly.
    kind: str
    #: One-line purpose, shown by `--list`.
    provides: str
    #: catalog: the module names to import. clone: the URL.
    modules: tuple[str, ...] = ()
    url: str = ""
    #: clone only: sparse-checkout patterns (only matched blobs are downloaded).
    #: A twin target must include its meshes: `twins/measure.py` fits envelopes
    #: to mesh geometry, so a URDF-only clone yields a different twin.
    patterns: tuple[str, ...] = ("*.urdf", "*.xml", "*.xacro")
    #: Path that exists after a successful fetch (relative to CACHE or extra_dir()).
    marker: str = ""
    #: clone only: the upstream commit the published numbers were measured at.
    #: Falls back to HEAD, with a warning, if upstream no longer has it.
    commit: str = ""


def extra_dir() -> Path:
    """Where directly-cloned repositories land. Gitignored."""
    from draft.paths import repo_root
    return repo_root() / "datasets" / "robot_descriptions" / "cache_extra"


#: The four platforms the twins rebuild (paper §IV-F), from three repositories.
TWIN_SOURCES: tuple[Source, ...] = (
    Source("unitree_ros", "catalog",
           "Unitree G1, Go2 and B2 URDFs — the twins' measurement source",
           modules=("g1_description", "go2_description", "b2_description"),
           marker="unitree_ros"),
    Source("mujoco_menagerie", "catalog",
           "MJCFs for the G1 and Go2 renders",
           modules=("g1_mj_description", "go2_mj_description"),
           marker="mujoco_menagerie"),
    Source("unitree_ros_all", "clone",
           "Unitree H2 — not in the catalogue, so cloned straight from the vendor",
           url="https://github.com/unitreerobotics/unitree_ros",
           # H2 only, meshes included (see `patterns`).
           patterns=("robots/h2_description/*",),
           marker="unitree_ros_all/robots/h2_description",
           commit="daadf41ee9afce8f90fdc09a98506012691fa122"),
)


@dataclass
class Result:
    key: str
    ok: bool
    detail: str
    path: Path | None = None
    already: bool = False


def present(src: Source) -> Path | None:
    """The fetched path, if this source is already on disk."""
    root = CACHE if src.kind == "catalog" else extra_dir()
    p = root / src.marker
    return p if p.exists() else None


def fetch(src: Source, *, timeout: int = 900) -> Result:
    """Fetch one source. Never raises — a dead upstream is a Result, not an error."""
    if (p := present(src)) is not None:
        dest = extra_dir() / src.key
        if src.kind == "clone" and (dest / ".git").exists():
            # The survey stage may have checked this repository out first, with
            # its XML only and at another commit; a twin measured without its
            # meshes is a different twin. Widen the checkout and move it to the pin.
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo",
                   "GCM_INTERACTIVE": "never"}
            try:
                subprocess.run(["git", "-C", str(dest), "sparse-checkout", "add",
                                *src.patterns],
                               check=True, capture_output=True, timeout=timeout, env=env)
                _checkout_pin(dest, src, env, timeout)
            except Exception as exn:
                return Result(src.key, False, _explain(exn, src))
        return Result(src.key, True, "already present", p, already=True)

    if src.kind == "catalog":
        got = []
        errs = []
        for mod in src.modules:
            try:
                __import__(f"robot_descriptions.{mod}")
                got.append(mod)
            except Exception as exn:                     # network, moved repo, bad pin
                errs.append(f"{mod}: {type(exn).__name__}: {exn}")
        p = present(src)
        if p is not None:
            note = "fetched" if not errs else f"fetched ({len(errs)} of {len(src.modules)} failed)"
            return Result(src.key, True, note, p)
        return Result(src.key, False, errs[0] if errs else "imported, but nothing landed on disk")

    dest = extra_dir() / src.key
    # `git clone` refuses a non-empty destination; clear any earlier failed attempt.
    shutil.rmtree(dest, ignore_errors=True)
    # Never prompt for credentials: a deleted repo 404s and git would hang asking.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo",
           "GCM_INTERACTIVE": "never"}
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             src.url, str(dest)],
            check=True, capture_output=True, timeout=timeout, env=env)
        subprocess.run(
            ["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *src.patterns],
            check=True, capture_output=True, timeout=timeout, env=env)
        pinned = _checkout_pin(dest, src, env, timeout)
    except Exception as exn:
        shutil.rmtree(dest, ignore_errors=True)
        return Result(src.key, False, _explain(exn, src))

    p = present(src)
    if p is None:
        # Upstream moved the file; clear the clone so a later run can retry.
        shutil.rmtree(dest, ignore_errors=True)
        return Result(src.key, False,
                      f"cloned, but {src.marker} is not in it any more — "
                      f"the vendor moved or renamed it")
    return Result(src.key, True, pinned, p)


def _checkout_pin(dest: Path, src: Source, env: dict, timeout: int) -> str:
    """Check out the pinned commit (fetched by name, since the clone is shallow).

    Stays at HEAD if upstream no longer has it. Returns the status to report.
    """
    if not src.commit:
        return "cloned at HEAD (unpinned)"
    try:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1",
                        "origin", src.commit],
                       check=True, capture_output=True, timeout=timeout, env=env)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", src.commit],
                       check=True, capture_output=True, timeout=timeout, env=env)
        return f"cloned at {src.commit[:10]}"
    except Exception:
        return (f"cloned at HEAD — the pinned commit {src.commit[:10]} is gone "
                f"from upstream, so measurements may differ from the published ones")


def _explain(exn: Exception, src: Source) -> str:
    """A one-line reason a person can act on, rather than a git stack trace."""
    if isinstance(exn, subprocess.TimeoutExpired):
        return f"timed out; retry with a longer --timeout, or the vendor is slow"
    if isinstance(exn, subprocess.CalledProcessError):
        err = (exn.stderr or b"").decode(errors="replace").strip()
        low = err.lower()
        # GitHub returns 404 for deleted and private repos alike.
        if any(s in low for s in ("not found", "could not read username",
                                  "repository not found", "authentication failed",
                                  "terminal prompts disabled")):
            return f"not found at {src.url} — deleted, renamed, or made private"
        if "could not resolve host" in low or "network" in low:
            return "no network"
        return err.splitlines()[-1] if err else f"git exited {exn.returncode}"
    return f"{type(exn).__name__}: {exn}"


def fetch_many(sources, *, timeout: int = 900) -> list[Result]:
    """Fetch each in turn. One failure never stops the rest."""
    return [fetch(s, timeout=timeout) for s in sources]
