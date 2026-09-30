"""Structural invariants of the repository: reachability, imports, paths,
and no vendor models committed."""

import ast
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from draft.paths import repo_root

REPO = repo_root()
SRC = REPO / "src" / "draft"

#: Directories whose imports seed reachability; other src/draft modules must
#: be reached from these, tasks.yaml, or a console entry point.
ENTRY_DIRS = ("scripts", "tests", "datasets")

#: Modules loaded at runtime by a mechanism an import walk cannot see.
BY_CONTRACT = {
    # Base class for a robot's optional mesh.py (loaded by the generator).
    "draft.generation.root_mesh",
    # Re-exported lazily by name so `speed_limits` imports without mjlab.
    "draft.tasks.quadruped.env",
}


def _py_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _module_name(p: Path) -> str:
    rel = p.relative_to(REPO / "src")
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(p: Path, top_level_only: bool = False) -> set[str]:
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    out = set()
    # Relative imports resolve against the package: the module itself for an
    # __init__.py, else its parent.
    if p.is_relative_to(REPO / "src"):
        name = _module_name(p)
        pkg = name if p.name == "__init__.py" else name.rsplit(".", 1)[0]
    else:
        pkg = ""
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    for n in nodes:
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level and pkg:
                base = pkg.split(".")
                up = base[:len(base) - (n.level - 1)]
                mod = ".".join(up + ([n.module] if n.module else []))
                out.add(mod)
                out.update(f"{mod}.{a.name}" for a in n.names)
            elif n.module:
                out.add(n.module)
                out.update(f"{n.module}.{a.name}" for a in n.names)
    out |= _subprocess_modules(tree)
    return out


def _subprocess_modules(tree: ast.AST) -> set[str]:
    """Modules launched as `python -m draft.x.y` in an argv list (e.g. by
    `draft.lineup.sweep`)."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, (ast.List, ast.Tuple)):
            continue
        parts = [e.value for e in n.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        for i, v in enumerate(parts[:-1]):
            if v == "-m" and parts[i + 1].startswith("draft."):
                out.add(parts[i + 1])
    return out


def _console_script_modules() -> list[str]:
    """The module behind each `[project.scripts]` entry in pyproject.toml."""
    import re
    text = (REPO / "pyproject.toml").read_text()
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)",
                      text, re.S | re.M)
    if not block:
        return []
    return re.findall(r'=\s*"([\w.]+):', block.group(1))


def _reachable() -> set[str]:
    by_name = {_module_name(p): p for p in _py_files(SRC)}
    seen, stack = set(), []

    # Seeds: scripts, tests, dataset stages, the task registry, console scripts.
    for d in ENTRY_DIRS:
        for p in _py_files(REPO / d):
            stack.extend(m for m in _imports(p) if m.startswith("draft"))
    import yaml
    reg = yaml.safe_load((SRC / "tasks" / "tasks.yaml").read_text())["tasks"]
    stack += [f"draft.tasks.{v['package'].replace('/', '.')}" for v in reg.values()]
    stack += _console_script_modules()

    while stack:
        m = stack.pop()
        # longest known prefix, so `draft.trends.feasibility.MotorTrends` resolves
        while m and m not in by_name:
            if "." not in m:
                m = None
                break
            m = m.rsplit(".", 1)[0]
        if not m or m in seen:
            continue
        seen.add(m)
        stack.extend(x for x in _imports(by_name[m]) if x.startswith("draft"))
        # a package's __init__ keeps its children reachable
        seen.add(m.rsplit(".", 1)[0])
    return seen


def test_no_orphaned_modules():
    """Every module under src/draft is reachable from something that runs it."""
    reachable = _reachable()
    orphans = sorted(
        _module_name(p) for p in _py_files(SRC)
        if _module_name(p) not in reachable
        and _module_name(p) not in BY_CONTRACT
        and not _module_name(p).endswith(".__init__")
        and _module_name(p) != "draft")
    assert not orphans, (
        "unreachable from any script, task or entry point — delete it or wire "
        "it up:\n  " + "\n  ".join(orphans))


def test_no_import_cycles():
    # Module-level imports only; function-level imports are a deliberate way
    # to break a cycle.
    edges = defaultdict(set)
    by_name = {_module_name(p): p for p in _py_files(SRC)}
    for name, p in by_name.items():
        for m in _imports(p, top_level_only=True):
            while m and m not in by_name:
                m = m.rsplit(".", 1)[0] if "." in m else None
            if m and m != name:
                edges[name].add(m)
    colour = {}

    def visit(n, path):
        if colour.get(n) == "done":
            return
        if colour.get(n) == "open":
            pytest.fail("import cycle: " + " -> ".join(path + [n]))
        colour[n] = "open"
        for m in sorted(edges[n]):
            visit(m, path + [n])
        colour[n] = "done"

    for n in sorted(by_name):
        visit(n, [])


def test_trends_do_not_import_the_engine():
    """draft.trends never imports draft.generation (dependency is one-way)."""
    for p in _py_files(SRC / "trends"):
        for m in _imports(p):
            assert not m.startswith("draft.generation"), \
                f"{p.name} imports {m}; trends must not depend on the engine"


def test_nothing_patches_sys_path_to_find_the_package():
    """The package is installed; nothing adds src/ to sys.path."""
    offenders = []
    for d in ("scripts", "tests", "datasets", "src"):
        for p in _py_files(REPO / d):
            if p.resolve() == Path(__file__).resolve():
                continue          # this file names the pattern it forbids
            text = p.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if "sys.path" in line and ('"src"' in line or "'src'" in line or "/src" in line):
                    offenders.append(f"{p.relative_to(REPO)}: {line.strip()}")
    assert not offenders, "\n  ".join(["these still reach for src/ by path:"] + offenders)


def test_no_module_recomputes_the_repo_root():
    """Only draft.paths resolves the repo root (a wrong root fails silently)."""
    offenders = []
    for p in _py_files(SRC):
        if p.name == "paths.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "Path(__file__)" in line and ("parents[" in line or line.count(".parent") >= 2):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "use draft.paths.repo_root() instead of counting directories:\n  "
        + "\n  ".join(offenders))


def test_the_package_is_source_only():
    """No file over 2 MB inside src/."""
    big = [p for p in SRC.rglob("*")
           if p.is_file() and p.stat().st_size > 2_000_000]
    assert not big, "large files in the source tree: " + ", ".join(str(p) for p in big)


def test_every_shipped_robot_has_the_three_files_a_robot_is():
    for d in (SRC / "robots").iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        assert (d / "parameters.yaml").exists(), f"{d.name} has no parameter table"
        assert (d / "tree.yaml").exists() or (d / "tree.py").exists(), \
            f"{d.name} has no kinematic tree"


# ── no third-party robot models are distributed ───────────────────────────────

#: Robot-model file extensions; vendor models are fetched, never committed.
MODEL_SUFFIXES = {".urdf", ".xacro", ".stl", ".dae", ".obj", ".mtl", ".skp", ".sdf"}


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                          text=True).stdout.split()


def test_no_vendor_robot_model_is_tracked():
    """No tracked file has a robot-model extension."""
    offenders = [f for f in _tracked() if Path(f).suffix.lower() in MODEL_SUFFIXES]
    assert not offenders, (
        "vendor robot models must not be committed — they belong to their "
        "vendors and are fetched with `scripts/setup_data.py`:\n  "
        + "\n  ".join(offenders))


def test_the_description_caches_are_ignored():
    """Fetched descriptions land outside version control, by construction."""
    ignored = (REPO / ".gitignore").read_text()
    for path in ("datasets/robot_descriptions/cache_extra/",):
        assert path in ignored, f"{path} must be gitignored — it holds vendor models"
    tracked = _tracked()
    assert not [f for f in tracked if f.startswith("datasets/robot_descriptions/cache_extra")]


def test_every_twin_source_has_an_address():
    """Every twin source states its purpose, a marker, and a pinned origin."""
    from draft.sources import TWIN_SOURCES
    assert TWIN_SOURCES
    for s in TWIN_SOURCES:
        assert s.provides, f"{s.key} does not say what it is for"
        assert s.marker, f"{s.key} has no way to tell whether it arrived"
        if s.kind == "clone":
            assert s.url.startswith("https://"), f"{s.key} has no URL"
            # Pinned so vendor pushes cannot move published results.
            assert len(s.commit) == 40 and all(c in "0123456789abcdef" for c in s.commit), \
                (f"{s.key} is not pinned to a commit, so a vendor push would "
                 f"change the numbers this repository publishes")
        else:
            # The catalogue pins a commit per repository itself.
            assert s.modules, f"{s.key} names no robot_descriptions module"


def test_a_dead_source_is_skipped_rather_than_raised():
    """Fetching a nonexistent repository (real URL, no mock) fails cleanly and
    leaves nothing behind."""
    from draft.sources import Source, extra_dir, fetch

    dead = Source("__test_dead__", "clone", "a repository that does not exist",
                  url="https://github.com/draft-test-org-does-not-exist/nope",
                  marker="__test_dead__/robots")
    r = fetch(dead, timeout=90)          # must not raise
    assert r.ok is False
    assert "not found" in r.detail or "network" in r.detail, r.detail
    assert not (extra_dir() / "__test_dead__").exists(), \
        "a failed clone must leave nothing behind, or the next attempt cannot retry"
