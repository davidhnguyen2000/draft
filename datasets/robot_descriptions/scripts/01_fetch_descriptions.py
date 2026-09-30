#!/usr/bin/env python3
"""Stage 1 — fetch official robot descriptions (URDF/MJCF) to a local cache.

Sources: the `robot_descriptions` catalog (commit-pinned), restricted to the
`humanoid` / `biped` / `quadruped` tags, plus `EXTRA_REPOS` cloned directly.
Quadrupeds are fitted as their own population, not used as a holdout.

Output: `data/manifest.json`, one record per description (path, repo, commit,
license). Stage 2 does the parsing.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from paths import CACHE, DATA, EXTRA_CACHE, ROOT, TRENDS_DATA, portable  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

# Not real hardware; fetched but kept out of the fit set.
SYNTHETIC = {
    "mujoco_humanoid_mj_description",  # the MuJoCo benchmark humanoid, not a robot
    "simple_humanoid_description",     # pedagogical model, box links
    "rhea_description",                # hobby build, no vendor inertials
    "spryped_description",             # hobby build
}

# Excluded quadrupeds, skipped before fetching (importing them clones large repos).
QUADRUPED_SKIP = {
    "minitaur_description": "8 linkage legs driven through a five-bar; motor mass "
                            "sits in the body, not at the joint",
    "wl_p311d_description": "wheeled quadruped — the leg carries a drive wheel",
    "wl_p311e_description": "wheeled quadruped — the leg carries a drive wheel",
}

# Platforms the catalog does not carry. Sparse, blobless clones: XML only.
EXTRA_REPOS = [
    # (key, url, [sparse patterns])
    ("openloong_azureloong", "https://github.com/loongOpen/openloong-dyn-control", ["*.urdf", "*.xml"]),
    ("engineai_pm01", "https://github.com/engineai-robotics/engineai_legged_gym", ["*.urdf", "*.xml"]),
    ("hector", "https://github.com/DRCL-USC/Hector_Simulation", ["*.urdf", "*.xml"]),
    ("bruce", "https://github.com/Westwood-Robotics/BRUCE-OP", ["*.urdf", "*.xml"]),
    ("berkeley_humanoid_lite", "https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets", ["*.urdf", "*.xml"]),
    ("nao", "https://github.com/ros-naoqi/nao_robot", ["*.urdf", "*.xacro", "*.xml"]),
    ("unitree_ros_all", "https://github.com/unitreerobotics/unitree_ros", ["*.urdf", "*.xml"]),
    ("menagerie", "https://github.com/google-deepmind/mujoco_menagerie", ["*.xml"]),
    ("deep_robotics", "https://github.com/DeepRoboticsLab/deep_robotics_model", ["*.urdf", "*.xml"]),
    ("limx_humanoid", "https://github.com/limxdynamics/humanoid-description", ["*.urdf", "*.xml"]),
    ("limx_tron2", "https://github.com/limxdynamics/tron2-robot-description", ["*.urdf", "*.xml"]),
    ("roboterax", "https://github.com/roboterax/models", ["*.urdf", "*.xml"]),
    ("robotis_ai_sapiens", "https://github.com/ROBOTIS-GIT/ai_sapiens", ["*.urdf", "*.xml"]),
    ("tienkung", "https://github.com/Open-X-Humanoid/TienKung_URDF", ["*.urdf", "*.xml"]),
    ("agibot_x1", "https://github.com/AgibotTech/agibot_x1_train", ["*.urdf", "*.xml"]),
    ("agibot_x2", "https://github.com/AgibotTech/agibot_x2_urdf", ["*.urdf", "*.xml"]),
    ("noetix_n2", "https://github.com/Noetix-Robotics/noetix_n2_gym", ["*.urdf", "*.xml"]),
    ("noetix_e1", "https://github.com/Noetix-Robotics/noetix_e1_lab", ["*.urdf", "*.xml"]),
    # fiveages-sim is a third-party aggregator, not a vendor (weaker provenance).
    ("fiveages", "https://github.com/fiveages-sim/robot_descriptions", ["*.urdf", "*.xml"]),
    ("magiclab_gen1", "https://github.com/MagiclabRobotics/magicbot-gen1_description", ["*.urdf", "*.xml"]),
    ("magiclab_z1", "https://github.com/MagiclabRobotics/magicbot-z1_description", ["*.urdf", "*.xml"]),
    ("magiclab_dog", "https://github.com/MagiclabRobotics/magicdog_description", ["*.urdf", "*.xml"]),
    ("agibot_a3", "https://github.com/AgibotTech/A3-A3U-robot-model", ["*.urdf", "*.xml"]),
    ("hightorque", "https://github.com/Hightorque-Robotics/HT_Robot_URDF", ["*.urdf", "*.xml"]),
    ("leju_kuavo", "https://github.com/LejuRobotics/kuavo-ros-opensource", ["*.urdf", "*.xml"]),
    ("engineai_desc", "https://github.com/engineai-robotics/engineai_robotics_description", ["*.urdf", "*.xml"]),
    ("booster_assets", "https://github.com/BoosterRobotics/booster_assets", ["*.urdf", "*.xml"]),
    # Quadrupeds with inertials and per-joint effort limits; exclusions are listed
    # in docs/robot-dataset.md -> Population.
    ("genisom", "https://github.com/zsibot/genisom_model", ["*.urdf", "*.xml"]),
    ("dobot_quad", "https://github.com/Dobot-Team/dobot_quad_sdk", ["*.urdf", "*.xml"]),
    ("dfki_quad", "https://github.com/dfki-ric-underactuated-lab/dfki-quad", ["*.urdf", "*.xml"]),
    # Aggregator; the only flat-URDF source for CyberDog 2.
    ("fiveages_quad", "https://github.com/fiveages-sim/robot-descriptions-quadruped", ["*.urdf", "*.xml"]),
]


def catalog_targets() -> list[str]:
    from robot_descriptions._descriptions import DESCRIPTIONS

    return sorted(n for n, d in DESCRIPTIONS.items()
                  if d.tags & {"humanoid", "biped", "quadruped"}
                  and n not in QUADRUPED_SKIP)


def fetch_catalog(name: str) -> dict:
    """Import a description module (which clones on first use) and resolve paths."""
    from robot_descriptions._cache import get_head_sha
    from robot_descriptions._descriptions import DESCRIPTIONS

    meta = DESCRIPTIONS[name]
    rec = {
        "key": name,
        "source": "robot_descriptions",
        "robot": meta.robot,
        "maker": meta.maker,
        "dof": meta.dof,
        "tags": sorted(meta.tags),
        "repository": meta.repository,
        "license_spdx": meta.license_spdx,
        "synthetic": name in SYNTHETIC,
    }
    try:
        mod = importlib.import_module(f"robot_descriptions.{name}")
    except Exception as exn:  # network, moved repo, broken pin
        rec["error"] = f"{type(exn).__name__}: {exn}"
        return rec
    for attr, key in (("URDF_PATH", "urdf"), ("MJCF_PATH", "mjcf")):
        path = getattr(mod, attr, None)
        if path and Path(path).exists():
            rec[key] = str(path)
    rec["repo_dir"] = str(getattr(mod, "REPOSITORY_PATH", "") or "")
    rec["commit"] = get_head_sha(rec["repo_dir"]) if rec["repo_dir"] else None
    return rec


#: The commit each direct clone was measured at. The catalogue pins its own
#: through the `robot_descriptions` version in constraints.txt; these are cloned
#: from a vendor's default branch, which moves, so without a pin a refit fits
#: whatever the vendor has pushed since.
EXTRA_COMMITS = {
    "openloong_azureloong": "4dd7a7e42a9cfd588afc78f3e429998ed8a30f4e",
    "engineai_pm01": "1a659aa0aa77445766ecea6d95708a5812e9f290",
    "hector": "edfdc7b1a9855c71887ac2cf2ebe0bb5dee5ea49",
    "bruce": "c23c1c819343831b07cf139f0d152383175c2995",
    "berkeley_humanoid_lite": "fc90fedd008b1e56a22e3c5221548d6b24f49707",
    "nao": "67476469a1371b00b17538eb6ea336367ece7d44",
    "unitree_ros_all": "daadf41ee9afce8f90fdc09a98506012691fa122",
    "menagerie": "da76818e269b82289eba39808e2fb91d679d6994",
    "deep_robotics": "e6753d2ef25e1e788d387ae3775fd283c199f1a3",
    "limx_humanoid": "02adfbdd206a56e684a60fd855f49aa214b4000c",
    "limx_tron2": "9939c22e69d27653ec0ba8a505859a2903dd1a71",
    "roboterax": "e8660e664e39e5b80ebb1f252e1b77b0e6929cff",
    "robotis_ai_sapiens": "7d62dff281f02b6265ebe07db6c0ae32ba8de381",
    "tienkung": "5c221783fb92fcc4af891ef1dc0502963caf2266",
    "agibot_x1": "e6651b9ab843fd1b1be70c087bfb7e8b28e44ccd",
    "agibot_x2": "77f43eb0904dae4c48ccd9154fee824f8ffd4d38",
    "noetix_n2": "153aad12dcc44aeafe378efe2f652873b02e0d89",
    "noetix_e1": "5a1d62d57d26499f10fc840df7b9f9d0331f5c72",
    "fiveages": "e02a822755a2913680d4d76397d8dcd45a33337a",
    "magiclab_gen1": "413a3a50cee2013f654f7c436ac131b935be4e84",
    "magiclab_z1": "f58c620ac323fe8c1ffda410d9fdb9074fe8adfc",
    "magiclab_dog": "e37cd1dca176b9aa19721bb5029dd377a1f3ed92",
    "agibot_a3": "589f508ff357447c610a3f3004419035ddc8f153",
    "hightorque": "5675b9aa238d5c0431bd376f9fc708308c617d7b",
    "leju_kuavo": "993699b3020371867ffdac34424590a941456e05",
    "engineai_desc": "1d2d7601f520f716863c32a83abe9c77b26ee06c",
    "booster_assets": "508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2",
    "genisom": "e6aa98e22d38ae3fdf4d448f79820295e78e83a5",
    "dobot_quad": "2c0eef369b4a8b956a679912178f71fb99e62d88",
    "dfki_quad": "dcf53c596339afd45b82f12c54b1e93e8273c2f4",
    "fiveages_quad": "98107c72260ba2344febf32dec262db12f6e20c7",
}


def fetch_extra(key: str, url: str, patterns: list[str]) -> dict:
    """Blobless + sparse clone: XML only, no mesh blobs ever downloaded."""
    dest = EXTRA_CACHE / key
    rec = {"key": key, "source": "direct_clone", "repository": url, "repo_dir": str(dest)}
    # Never prompt for credentials: a deleted repo would otherwise hang the fetch.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo",
           "GCM_INTERACTIVE": "never"}
    try:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)],
                check=True, capture_output=True, timeout=900, env=env,
            )
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *patterns],
                check=True, capture_output=True, timeout=900, env=env,
            )
        else:
            # Already there, perhaps checked out for another purpose: the twin
            # fetch (scripts/setup_data.py) takes only H2 out of unitree_ros_all.
            # Widen it, never narrow it, or every other robot in it silently
            # drops out of the fit.
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "add", *patterns],
                check=True, capture_output=True, timeout=900, env=env,
            )
        commit = EXTRA_COMMITS.get(key)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        if commit and head != commit:
            for cmd in (["fetch", "--depth", "1", "--filter=blob:none", "origin", commit],
                        ["checkout", "--quiet", "--detach", commit]):
                subprocess.run(["git", "-C", str(dest), *cmd],
                               check=True, capture_output=True, timeout=900, env=env)
        rec["commit"] = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        rec["files"] = sorted(
            str(p) for pat in ("*.urdf", "*.xml") for p in dest.rglob(pat)
        )
    except Exception as exn:
        # A vanished repository is a recorded skip, not a failure.
        shutil.rmtree(dest, ignore_errors=True)
        err = (exn.stderr or b"").decode(errors="replace").strip() \
            if isinstance(exn, subprocess.CalledProcessError) and exn.stderr else str(exn)
        low = err.lower()
        if any(s in low for s in ("not found", "could not read username",
                                  "repository not found", "authentication failed",
                                  "terminal prompts disabled")):
            rec["error"] = f"gone from {url} — deleted, renamed or made private"
        else:
            rec["error"] = f"{type(exn).__name__}: {err.splitlines()[-1] if err else exn}"
    return rec


def _portable_record(rec: dict) -> dict:
    """`rec` with every path made machine-independent, for the committed manifest."""
    rec = dict(rec)
    for key in ("urdf", "mjcf", "repo_dir"):
        if rec.get(key):
            rec[key] = portable(rec[key])
    if "files" in rec:
        rec["files"] = [portable(f) for f in rec["files"]]
    return rec


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    records = []
    targets = catalog_targets()
    print(f"[catalog] {len(targets)} humanoid/biped descriptions", flush=True)
    for i, name in enumerate(targets, 1):
        rec = fetch_catalog(name)
        status = rec.get("error", ",".join(k for k in ("urdf", "mjcf") if k in rec) or "NO PATH")
        print(f"  [{i:2d}/{len(targets)}] {name:34s} {status}", flush=True)
        records.append(rec)

    # Recorded, not fetched — so the skip is auditable in the manifest.
    for name, why in sorted(QUADRUPED_SKIP.items()):
        records.append({"key": name, "source": "robot_descriptions",
                        "skipped": why, "error": f"not fetched: {why}"})

    print(f"[extra] {len(EXTRA_REPOS)} direct clones", flush=True)
    for key, url, patterns in EXTRA_REPOS:
        rec = fetch_extra(key, url, patterns)
        n = len(rec.get("files", []))
        print(f"  {key:26s} {rec.get('error', str(n) + ' xml files')}", flush=True)
        records.append(rec)

    out = DATA / "manifest.json"
    write_json(out, [_portable_record(r) for r in records], indent=2)
    ok = sum(1 for r in records if "error" not in r)
    print(f"\nwrote {out}  ({ok}/{len(records)} ok)")


if __name__ == "__main__":
    sys.exit(main())
