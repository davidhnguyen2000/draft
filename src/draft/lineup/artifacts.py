#!/usr/bin/env python3
"""Pack, fetch and verify the released outputs of the sweep.

Four archives, each fetchable alone; paths inside are relative to ``logs/``:

  checkpoints  final ``model_*.pt`` + ``run_meta.json`` of the 120 runs
  eval         the 360 evaluation JSONs in ``lineup_eval/``
  walk         the 30 flat-walk recordings in ``lineup_walk/``
  traces       the sprint traces behind the actuator-limit statistics

The manifest (``Study.manifest``) records each archive's size, sha256 and url.

  python scripts/train_all.py fetch pack --out /somewhere/release
  python scripts/train_all.py fetch --only eval walk
  python scripts/train_all.py fetch verify --dir /somewhere/release
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from ..paths import repo_root
from . import study
from .sweep import ARMS, DESIGNS, EVAL_CELLS, SEEDS, TASKS, Sweep

REPO = repo_root()
LOGS = REPO / "logs"
CACHE = LOGS / ".artifacts"
GROUP_NAMES = ("checkpoints", "eval", "walk", "traces")

R = study.STUDY


def groups() -> dict:
    return {
        "checkpoints": ("draft_lineup_checkpoints.tar",
                        "The final model_*.pt and run_meta.json of each of the 120 runs "
                        "(<robot>_<task>_s<seed>), and nothing else. Optimizer state is "
                        "removed: the policies evaluate, render and warm-start, but cannot "
                        "--resume. Extracts to logs/<run>/<timestamp>_<run>/."),
        "eval": ("draft_lineup_eval.tar",
                 f"The 360 evaluation JSONs (3 designs x 10 seeds x 6 cells x 2 arms). "
                 f"Extracts to logs/{R.eval_dir}/."),
        "walk": ("draft_lineup_walk.tar",
                 "The 30 flat-walk recordings of the flat policies. "
                 f"Extracts to logs/{R.walk_dir}/."),
        "traces": ("draft_lineup_traces.tar",
                   "Sprint-peak joint traces the actuator-limit statistics in "
                   f"experiments/results.json are reduced from "
                   f"(logs/{R.sprint_traces('*', ['*'])[0]})."),
    }


# ── what each archive should hold ────────────────────────────────────────────
def _expected(group: str) -> tuple[list[str], list[str]]:
    """(files present, relative to logs/; descriptions of what is missing)."""
    have, missing = [], []

    def want(rel: str):
        (have if (LOGS / rel).is_file() else missing).append(rel)

    if group == "checkpoints":
        for r in DESIGNS:
            for t in TASKS:
                for s in SEEDS:
                    run = R.run(r, t, s)
                    got = Sweep.newest(run)
                    if got is None:
                        missing.append(f"{run}/*/model_*.pt")
                        continue
                    _run_dir_files(got[0], have, missing)
    elif group == "eval":
        for arm in ARMS:
            for r in DESIGNS:
                for s in SEEDS:
                    for _, _, label, extra in EVAL_CELLS:
                        want(f"{R.eval_dir}/{arm}split_{r}_s{s}_{label}.json" if extra
                             else f"{R.eval_dir}/{arm}_{r}_s{s}_{label}.json")
    elif group == "walk":
        for r in DESIGNS:
            for s in SEEDS:
                want(f"{R.walk_dir}/{r}_s{s}.npz")
    elif group == "traces":
        for r in DESIGNS:
            for trace in R.sprint_traces(r, SEEDS):
                want(trace)
    return sorted(have), missing


def _run_dir_files(model: Path, have: list, missing: list) -> None:
    """The checkpoint, plus the run's metadata and any params dump beside it."""
    have.append(str(model.relative_to(LOGS)))
    meta = model.parent / "run_meta.json"
    if meta.is_file():
        have.append(str(meta.relative_to(LOGS)))
    else:
        missing.append(str(meta.relative_to(LOGS)))
    for p in sorted((model.parent / "params").glob("*")):
        if p.is_file():
            have.append(str(p.relative_to(LOGS)))


# ── helpers ──────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_only(path: Path) -> bytes:
    """A checkpoint without optimizer state: enough to evaluate or warm-start, not to --resume."""
    try:
        import torch
    except ModuleNotFoundError:
        raise SystemExit("packing checkpoints needs torch (pip install torch)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt.pop("optimizer_state_dict", None)
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    return buf.getvalue()


def _tar(path: Path, members: list[str]) -> None:
    """A reproducible tar: sorted, no owner. mtimes stay, since newest-checkpoint lookup reads them."""
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        for rel in members:
            info = tar.gettarinfo(LOGS / rel, arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            if rel.endswith(".pt"):
                data = _policy_only(LOGS / rel)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                continue
            with (LOGS / rel).open("rb") as f:
                tar.addfile(info, f)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}: the archives have not been published yet. "
            f"Depth 1 (`scripts/train_all.py numbers`) needs none; see docs/reproducing.md.")
    return json.loads(path.read_text())


def _entries(manifest: dict, only) -> list[dict]:
    return [a for a in manifest["archives"] if not only or a["name"] in only]


def skeleton() -> dict:
    return {
        "status": "unpublished: no archive has been uploaded yet, so every url is null",
        "about": "Released results of the §V study. Fetch with `python scripts/train_all.py fetch "
                 f"fetch`; written by `artifacts.py pack`. Paths inside each "
                 "archive are relative to logs/.",
        "archives": [{"name": g, "file": f, "size": None, "sha256": None, "n_files": None,
                      "url": None, "contents": d} for g, (f, d) in groups().items()],
    }


# ── subcommands ──────────────────────────────────────────────────────────────
def pack(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve() if args.manifest else R.manifest
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else skeleton()
    old_by_name = {a["name"]: a for a in old["archives"]}
    all_missing = {}
    table = groups()
    for group in args.only or GROUP_NAMES:
        members, missing = _expected(group)
        file, desc = table[group]
        if missing:
            all_missing[group] = missing
        if not members:
            print(f"{group:12s} nothing to pack ({len(missing)} missing)")
            continue
        _tar(out / file, members)
        digest, size = sha256(out / file), (out / file).stat().st_size
        prev = old_by_name.get(group, {})
        old_by_name[group] = {
            "name": group, "file": file, "size": size, "sha256": digest,
            "n_files": len(members),
            # an unchanged archive keeps its upload; a changed one needs a new one
            "url": prev.get("url") if prev.get("sha256") == digest else None,
            "contents": desc}
        print(f"{group:12s} {file:28s} {_human(size):>10s}  {len(members):4d} files"
              f"  {len(missing):3d} missing  sha256 {digest[:12]}")

    for group, missing in all_missing.items():
        print(f"\n{group}: {len(missing)} missing")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    manifest = {**old, "archives": [old_by_name[g] for g in GROUP_NAMES if g in old_by_name]}
    manifest["status"] = ("unpublished: fill in each url after uploading"
                          if any(a["url"] is None for a in manifest["archives"]) else "published")
    if all_missing and not args.manifest:
        # An incomplete pack must not overwrite the manifest users fetch from.
        manifest_path = out / R.manifest.name
        print(f"\npack is INCOMPLETE; manifest written to {manifest_path}, "
              f"not {R.manifest.relative_to(REPO)}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {manifest_path}")
    return 1 if all_missing else 0


def fetch(args) -> int:
    manifest = _load(Path(args.manifest) if args.manifest else R.manifest)
    cache, dest = Path(args.cache), Path(args.dest)
    entries = _entries(manifest, args.only)
    unpublished = [a["name"] for a in entries if not a.get("url") or not a.get("sha256")]
    if unpublished:
        raise SystemExit(f"not published yet (no url or sha256 in the manifest): "
                         f"{', '.join(unpublished)}. Ask the maintainers, or rebuild with "
                         f"scripts/train_all.py.")
    cache.mkdir(parents=True, exist_ok=True)
    for a in entries:
        path = cache / a["file"]
        if path.exists() and sha256(path) == a["sha256"]:
            print(f"{a['name']:12s} already downloaded: {path}")
        else:
            print(f"{a['name']:12s} downloading {a['url']} ({_human(a['size'])})", flush=True)
            part = path.with_suffix(".part")
            with urllib.request.urlopen(a["url"]) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256(part) != a["sha256"]:
                part.unlink()
                raise SystemExit(f"{a['file']}: sha256 does not match the manifest; "
                                 f"download discarded")
            part.replace(path)
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tar:
            n = len(tar.getmembers())
            tar.extractall(dest, filter="data")
        print(f"{a['name']:12s} verified, extracted {n} files into {dest}")
    return 0


def verify(args) -> int:
    manifest = _load(Path(args.manifest) if args.manifest else R.manifest)
    bad = 0
    for a in _entries(manifest, args.only):
        path = Path(args.dir) / a["file"]
        if not a.get("sha256"):
            print(f"{a['name']:12s} no sha256 in the manifest (unpublished)")
            bad += 1
        elif not path.exists():
            print(f"{a['name']:12s} MISSING {path}")
            bad += 1
        elif sha256(path) != a["sha256"]:
            print(f"{a['name']:12s} MISMATCH {path}")
            bad += 1
        else:
            print(f"{a['name']:12s} ok {path}")
    return 1 if bad else 0


def main() -> int:
    global R
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    only = dict(nargs="+", choices=GROUP_NAMES, default=None)

    p = sub.add_parser("pack", help="build the archives from local logs/")
    p.add_argument("--out", required=True, help="directory for the .tar files")
    p.add_argument("--only", **only)
    p.add_argument("--manifest", default=None,
                   help="manifest to write (default experiments/artifacts.json, or <out>/<its name> "
                        "when anything is missing)")

    p = sub.add_parser("fetch", help="download, verify and extract into logs/")
    p.add_argument("--only", **only)
    p.add_argument("--manifest", default=None)
    p.add_argument("--cache", default=str(CACHE), help="where downloads are kept")
    p.add_argument("--dest", default=str(LOGS), help="extract here (default logs/)")

    p = sub.add_parser("verify", help="check archives against the manifest's sha256")
    p.add_argument("--dir", default=str(CACHE))
    p.add_argument("--only", **only)
    p.add_argument("--manifest", default=None)

    args = ap.parse_args()
    return {"pack": pack, "fetch": fetch, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
