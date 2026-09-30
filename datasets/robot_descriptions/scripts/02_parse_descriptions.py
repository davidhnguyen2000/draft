#!/usr/bin/env python3
"""Stage 2 — parse each description into a normalized segment table.

Pinocchio loads the URDF/MJCF (merging fixed joints). Co-located joints (e.g. a
3-DOF hip authored as three near-zero-offset links) are further merged into one
*segment*, so which link carries the motor mass stops mattering.

Per segment: mass, COM, inertia, carried joints and limits, `length` to the
next segment, and box/cylinder extents derived from the inertia (no meshes).

Output: `data/segments.json` (one record per robot, list of segments).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pinocchio as pin

from paths import CACHE, DATA, EXTRA_CACHE, ROOT, TRENDS_DATA, local, portable  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

# Two joints closer than this are treated as one physical joint cluster
# (3-DOF hips/shoulders/wrists are authored as ~zero-offset link chains).
COLOCATION_M = 0.02

EXTRA_CACHE = ROOT / "cache_extra"

# Models from EXTRA_REPOS (platforms outside the robot_descriptions catalog).
# (key, robot, maker, repo_key, relative path, tags); trailing comments give the
# parsed mass and DOF.
EXTRA_MODELS = [
    ("azureloong", "AzureLoong", "OpenLoong / Humanoid Robot (Shanghai)", "openloong_azureloong",
     "models/AzureLoong.urdf", ["humanoid"]),
    ("zq_sa01", "SA01", "ZhiQiang / EngineAI", "engineai_pm01",
     "resources/robots/zq_humanoid/urdf/zq_sa01.urdf", ["humanoid"]),
    ("berkeley_humanoid_lite", "Berkeley Humanoid Lite", "Hybrid Robotics", "berkeley_humanoid_lite",
     "data/robots/berkeley_humanoid/berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf", ["humanoid"]),
    ("nao_v50", "NAO V50", "Aldebaran / SoftBank", "nao",
     "nao_description/urdf/naoV50_generated_urdf/nao.urdf", ["humanoid"]),
    ("unitree_h2", "H2", "UNITREE Robotics", "unitree_ros_all",
     "robots/h2_description/H2.urdf", ["humanoid"]),
    ("unitree_h2_plus", "H2 Plus", "UNITREE Robotics", "unitree_ros_all",
     "robots/h2_plus/H2_Plus.urdf", ["humanoid"]),
    ("unitree_g1_d", "G1-D", "UNITREE Robotics", "unitree_ros_all",
     "robots/g1_d_description/g1_d.urdf", ["humanoid"]),
    # Inclusion in the fit is decided by stages 3/4, not this list.
    ("dr02_std", "DR02", "DeepRobotics", "deep_robotics",
     "DR02/urdf/standard/DR02-std.urdf", ["humanoid"]),                     # 67.6 kg, 21 DOF
    ("dr02_pro", "DR02 Pro", "DeepRobotics", "deep_robotics",
     "DR02/urdf/pro/DR02-pro.urdf", ["humanoid"]),                          # 78.1 kg, 31 DOF
    ("limx_oli_d03", "Oli HU_D03", "LimX Dynamics", "limx_humanoid",
     "HU_D03_description/urdf/HU_D03_03.urdf", ["humanoid"]),               # 51.7 kg, 31 DOF
    ("limx_oli_d04", "Oli HU_D04", "LimX Dynamics", "limx_humanoid",
     "HU_D04_description/urdf/HU_D04_01.urdf", ["humanoid"]),               # 52.9 kg, 31 DOF
    ("limx_tron2a_sf", "TRON2A sole-foot", "LimX Dynamics", "limx_tron2",
     "tron2/SF_TRON2A/urdf/robot.urdf", ["humanoid"]),                      # 34.8 kg, 10 DOF — legs only
    ("limx_tron2a_da", "TRON2A dual-arm", "LimX Dynamics", "limx_tron2",
     "tron2/DA_TRON2A/urdf/robot.urdf", ["humanoid"]),                      # 59.7 kg, 14 DOF
    ("limx_oli_d04_hand", "Oli HU_D04 +hand", "LimX Dynamics", "limx_humanoid",
     "HU_D04_description/urdf/HU_D04_01_with_hand.urdf", ["humanoid"]),     # 53.2 kg, 63 DOF
    ("limx_tron2a_wf", "TRON2A wheel-foot", "LimX Dynamics", "limx_tron2",
     "tron2/WF_TRON2A/urdf/robot.urdf", ["humanoid"]),                      # 34.9 kg, 10 DOF
    ("robotis_k1", "AI Sapiens K1", "ROBOTIS", "robotis_ai_sapiens",
     "ai_sapiens_description/urdf/k1_rev1/k1.urdf", ["humanoid"]),          # 34.8 kg, 23 DOF
    ("robotera_star1", "STAR1", "Robot Era", "roboterax",
     "star1/urdf/l3_with_hand_fixedpin_xml.urdf", ["humanoid"]),            # 64.0 kg, 55 DOF
    ("deeprobotics_lite3", "Lite3", "DeepRobotics", "deep_robotics",
     "Lite3/urdf/Lite3.urdf", ["quadruped"]),                               # 11.9 kg, 12 DOF
    ("deeprobotics_x30", "X30", "DeepRobotics", "deep_robotics",
     "X30/urdf/X30.urdf", ["quadruped"]),                                   # 55.8 kg, 12 DOF
    ("deeprobotics_m20", "M20", "DeepRobotics", "deep_robotics",
     "M20/urdf/M20.urdf", ["quadruped"]),                                   # 34.5 kg, 16 DOF — wheel-legged
    ("deeprobotics_m20s", "M20S", "DeepRobotics", "deep_robotics",
     "M20S/urdf/M20S.urdf", ["quadruped"]),                                 # 34.5 kg, 16 DOF — wheel-legged
    ("tienkung_lite", "Tien Kung Lite", "X-Humanoid", "tienkung",
     "lite_urdf_publish/urdf/humanoid_publish.urdf", ["humanoid"]),          # 42.5 kg, 20 DOF
    ("tienkung_pro", "Tien Kung Pro", "X-Humanoid", "tienkung",
     "pro_urdf_publish/pro_urdf_publish/urdf/humanoid.urdf", ["humanoid"]),  # 51.5 kg, 50 DOF
    ("tienkung2_pro", "Tien Kung 2 Pro", "X-Humanoid", "tienkung",
     "tiangong2pro_urdf/urdf/tiangong2.0_pro_urdf.urdf", ["humanoid"]),      # 68.0 kg, 30 DOF
    ("tienkung2_dex", "Tien Kung 2 Dex", "X-Humanoid", "tienkung",
     "tiangong2dex_urdf/urdf/tiangong2dex.urdf", ["humanoid"]),              # 63.3 kg, 31 DOF
    ("tienkung3", "Tien Kung 3", "X-Humanoid", "tienkung",
     "tiangong3_urdf/urdf/tiangong3.urdf", ["humanoid"]),                    # 60.8 kg, 25 DOF
    ("agibot_x1", "AgiBot X1", "AgiBot", "agibot_x1",
     "resources/robots/x1/urdf/x1.urdf", ["humanoid"]),                      # 35.3 kg, 12 DOF — legs only
    ("agibot_x2_ultra", "AgiBot X2 Ultra", "AgiBot", "agibot_x2",
     "X2_URDF-v1.4.0/X2-Ultra.urdf", ["humanoid"]),                          # 44.8 kg, 31 DOF
    ("noetix_n2", "N2", "Noetix Robotics", "noetix_n2",
     "resources/robots/N2/urdf/N2.urdf", ["humanoid"]),                      # 33.2 kg, 18 DOF
    ("noetix_e1", "E1", "Noetix Robotics", "noetix_e1",
     "source/NoetixE1/NoetixE1/assets/robots/e1/urdf/e1_24dof.urdf", ["humanoid"]),  # 41.6 kg, 24 DOF
    ("robotera_xbot", "XBot", "Robot Era", "fiveages",
     "humanoid/RobotEra/xbot_description/urdf/robot.urdf", ["humanoid"]),    # 53.0 kg, 28 DOF
    ("unitree_g1_revo2", "G1 Revo2", "UNITREE Robotics", "fiveages",
     "humanoid/Unitree/unitree_g1_description/urdf/unitree_g1_revo2.urdf", ["humanoid"]),  # 33.5 kg
    # Where a repo ships several revisions of one product, only the newest is taken.
    ("magicbot_gen1", "MagicBot Gen1", "MagicLab", "magiclab_gen1",
     "urdf/MAGICBOT.urdf", ["humanoid"]),                                       # 66.4 kg, 30 DOF
    ("magicbot_z1", "MagicBot Z1", "MagicLab", "magiclab_z1",
     "urdf/MagicBotZ1.urdf", ["humanoid"]),                                     # 39.3 kg, 23 DOF
    ("agibot_a3", "A3", "AgiBot", "agibot_a3",
     "a3_t3d0/urdf/model.urdf", ["humanoid"]),                                  # 56.8 kg, 31 DOF
    ("agibot_a3_ultra", "A3 Ultra", "AgiBot", "agibot_a3",
     "a3_ultra_t3d0/urdf/model.urdf", ["humanoid"]),                            # 60.0 kg, 31 DOF
    ("hightorque_hi", "Hi", "HighTorque", "hightorque",
     "hi_25dof/urdf/hi_25dof.urdf", ["humanoid"]),                              # 18.3 kg, 25 DOF
    ("hightorque_pi_plus", "Pi Plus", "HighTorque", "hightorque",
     "pi_plus_24dof/urdf/pi_plus_24dof.urdf", ["humanoid"]),                    # 11.0 kg, 24 DOF
    # `_gazebo` cut: same model as biped_s45.urdf, without non-standard <capsule> tags.
    ("kuavo_s45", "Kuavo 4 Pro", "Leju Robotics", "leju_kuavo",
     "src/kuavo_assets/models/biped_s45/urdf/biped_s45_gazebo.urdf", ["humanoid"]),  # 58.0 kg, 28 DOF
    ("kuavo_s53", "Kuavo 5", "Leju Robotics", "leju_kuavo",
     "src/kuavo_assets/models/biped_s200053/urdf/biped_s200053.urdf", ["humanoid"]),  # 69.8 kg, 37 DOF
    ("engineai_t800", "T800", "EngineAI", "engineai_desc",
     "t800/urdf/serial_t800.urdf", ["humanoid"]),                               # 85.0 kg, 25 DOF
    # Not `engineai_pm01`: that EXTRA_REPOS key names a different repository
    # (see `_check_key_collisions`).
    ("engineai_pm01_edu", "PM01 EDU", "EngineAI", "engineai_desc",
     "pm01_edu/urdf/serial_pm01_edu.urdf", ["humanoid"]),                       # 40.9 kg, 24 DOF
    ("booster_k1", "K1", "Booster Robotics", "booster_assets",
     "robots/K1/K1_22dof.urdf", ["humanoid"]),                                  # 19.7 kg, 22 DOF
    ("magicdog", "MagicDog", "MagicLab", "magiclab_dog",
     "urdf/magicdog.urdf", ["quadruped"]),                                      # 17.4 kg, 12 DOF
    # Quadrupeds: a fit population (docs/robot-dataset.md), not a holdout.
    ("unitree_a1", "A1", "UNITREE Robotics", "unitree_ros_all",
     "robots/a1_description/urdf/a1.urdf", ["quadruped"]),
    ("unitree_go1", "Go1", "UNITREE Robotics", "unitree_ros_all",
     "robots/go1_description/urdf/go1.urdf", ["quadruped"]),
    ("unitree_aliengo", "Aliengo", "UNITREE Robotics", "unitree_ros_all",
     "robots/aliengo_description/urdf/aliengo.urdf", ["quadruped"]),
    ("unitree_a2", "A2", "UNITREE Robotics", "unitree_ros_all",
     "robots/a2_description/urdf/a2.urdf", ["quadruped"]),
    ("unitree_go2", "Go2", "UNITREE Robotics", "unitree_ros_all",
     "robots/go2_description/urdf/go2_description.urdf", ["quadruped"]),
    ("unitree_b1", "B1", "UNITREE Robotics", "unitree_ros_all",
     "robots/b1_description/xacro/b1.urdf", ["quadruped"]),
    ("unitree_b2", "B2", "UNITREE Robotics", "unitree_ros_all",
     "robots/b2_description/urdf/b2_description.urdf", ["quadruped"]),
    ("unitree_laikago", "Laikago", "UNITREE Robotics", "unitree_ros_all",
     "robots/laikago_description/urdf/laikago.urdf", ["quadruped"]),
    ("barkour_v0", "Barkour v0", "Google DeepMind", "menagerie",
     "google_barkour_v0/barkour_v0.xml", ["quadruped"]),
    ("barkour_vb", "Barkour vB", "Google DeepMind", "menagerie",
     "google_barkour_vb/barkour_vb.xml", ["quadruped"]),
    ("unitree_as2", "AS2", "UNITREE Robotics", "unitree_ros_all",
     "robots/as2_description/urdf/as2.urdf", ["quadruped"]),                # 17.6 kg, 12 DOF
    ("zsibot_zsl1", "ZSL-1", "ZSIBot / GENISOM AI", "genisom",
     "zsl-1/urdf/ZSL-1.urdf", ["quadruped"]),                               # 15.2 kg, 12 DOF
    ("dobot_rover_x1", "Rover X1", "Dobot", "dobot_quad",
     "resources/dobot_quad/urdf/dobot_quad_ros.urdf", ["quadruped"]),       # 18.7 kg, 12 DOF
    ("dfki_quad", "DFKI Quad", "DFKI Bremen", "dfki_quad",
     "ws/src/common/model/urdf/quad.urdf", ["quadruped"]),                  # 10.9 kg, 12 DOF
    # Aggregator's flat expansion of Xiaomi's xacro; the mass identifies CyberDog 2.
    ("xiaomi_cyberdog2", "CyberDog 2", "Xiaomi", "fiveages_quad",
     "magiclab/cyberdog_description/urdf/cyberdog.urdf", ["quadruped"]),    # 8.8 kg, 12 DOF
]


def equivalent_extents(mass: float, inertia_com: np.ndarray) -> dict:
    """Size proxies derived from the inertia tensor — no meshes required.

    Returns the equivalent solid box a>=b>=c reproducing the principal
    inertias, plus the equivalent solid cylinder (r, L) about the long axis.
    `inertia_valid` is False when the principal inertias violate the triangle
    inequality, i.e. no solid body can have that tensor.
    """
    if mass <= 0:
        return {"inertia_valid": None, "box": None, "cyl_r": None, "cyl_L": None}
    eig = np.sort(np.linalg.eigvalsh(inertia_com))  # I1 <= I2 <= I3
    I1, I2, I3 = eig
    valid = bool(I1 > 0 and (I1 + I2 - I3) > -1e-12 * max(I3, 1e-12))
    sq = np.array([
        6.0 * (I2 + I3 - I1) / mass,   # a^2, longest
        6.0 * (I1 + I3 - I2) / mass,   # b^2
        6.0 * (I1 + I2 - I3) / mass,   # c^2, shortest
    ])
    box = np.sqrt(np.clip(sq, 0.0, None))
    a, b, c = box
    # Equivalent cylinder about the long axis a: I_axial = m r^2 / 2 -> r from I1
    r = float(np.sqrt(max(2.0 * I1 / mass, 0.0)))
    return {
        "inertia_valid": valid,
        "box": [float(x) for x in box],
        "box_a": float(a), "box_b": float(b), "box_c": float(c),
        "cyl_r": r,
        "cyl_L": float(a),
        "gyration_radius": float(np.sqrt(np.trace(inertia_com) / (2.0 * mass))),
    }


def load_model(rec: dict):
    """Build a pinocchio model with a free-flyer root; prefer URDF over MJCF."""
    errors = []
    for fmt, key in (("urdf", "urdf"), ("mjcf", "mjcf")):
        path = local(rec[key]) if rec.get(key) else None
        if not path or not Path(path).exists():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if fmt == "urdf":
                    model = pin.buildModelFromUrdf(path, pin.JointModelFreeFlyer())
                else:
                    model = pin.buildModelFromMJCF(path)
            return model, fmt, path
        except Exception as exn:
            errors.append(f"{fmt}: {type(exn).__name__}: {exn}")
    raise RuntimeError("; ".join(errors) or "no parsable file")


def segment_of(model) -> tuple[list[int], dict[int, int]]:
    """(seg_of, seg_root): each pinocchio joint's segment, and each segment's
    root joint. A joint closer than `COLOCATION_M` to its parent joins its segment."""
    seg_of = [0] * model.njoints  # index 0 = universe
    seg_root = {}
    for j in range(1, model.njoints):
        parent = model.parents[j]
        d = float(np.linalg.norm(model.jointPlacements[j].translation))
        if parent >= 1 and d < COLOCATION_M:
            seg_of[j] = seg_of[parent]
        else:
            seg_of[j] = len(seg_root)
            seg_root[seg_of[j]] = j
    return seg_of, seg_root


def build_segments(model, data) -> list[dict]:
    """Collapse the pinocchio joint tree into co-located segments."""
    njoints = model.njoints  # index 0 = universe

    # 1. Which joints are co-located with their parent -> same segment.
    seg_of, seg_root = segment_of(model)

    nseg = len(seg_root)
    segs = [{"idx": s, "root_joint": model.names[seg_root[s]], "joints": [],
             "pin_joints": [], "mass": 0.0} for s in range(nseg)]

    # 2. Accumulate inertia per segment, expressed in the segment-root frame.
    accum = [pin.Inertia.Zero() for _ in range(nseg)]
    for j in range(1, njoints):
        s = seg_of[j]
        # placement of joint j in the frame of its segment root
        M = pin.SE3.Identity()
        k = j
        while k != seg_root[s]:
            M = model.jointPlacements[k] * M
            k = model.parents[k]
        accum[s] += model.inertias[j].se3Action(M)

        jm = model.joints[j]
        nq, nv, idx_v = jm.nq, jm.nv, jm.idx_v
        if nv == 1:  # revolute / prismatic: the actuated case we care about
            segs[s]["joints"].append({
                "name": model.names[j],
                "shortname": jm.shortname(),
                "effort": float(model.effortLimit[idx_v]),
                "velocity": float(model.velocityLimit[idx_v]),
                "lower": float(model.lowerPositionLimit[jm.idx_q]),
                "upper": float(model.upperPositionLimit[jm.idx_q]),
                "colocated_with_root": bool(j != seg_root[s]),
            })
        segs[s]["pin_joints"].append(j)

    # 2b. Distance to the furthest fixed-joint frame: the length of a terminal
    # segment (foot, hand), which has no child joint to measure to.
    tip = [0.0] * nseg
    for f in model.frames:
        if f.type != pin.FrameType.FIXED_JOINT or f.parentJoint < 1:
            continue
        j, s = f.parentJoint, seg_of[f.parentJoint]
        M = pin.SE3.Identity()
        k = j
        while k != seg_root[s]:
            M = model.jointPlacements[k] * M
            k = model.parents[k]
        tip[s] = max(tip[s], float(np.linalg.norm((M * f.placement).translation)))

    # 3. Geometry: distance from this segment's root joint to each child segment.
    child_offsets = {s: [] for s in range(nseg)}
    parent_seg = {}
    for j in range(1, njoints):
        s, p = seg_of[j], seg_of[model.parents[j]] if model.parents[j] >= 1 else None
        if p is not None and s != p and j == seg_root[s]:
            # translation from parent segment root to this segment root
            M = pin.SE3.Identity()
            k = j
            while k != seg_root[p]:
                M = model.jointPlacements[k] * M
                k = model.parents[k]
            child_offsets[p].append(float(np.linalg.norm(M.translation)))
            parent_seg[s] = p

    for s, seg in enumerate(segs):
        I = accum[s]
        seg["mass"] = float(I.mass)
        seg["com"] = [float(x) for x in I.lever]
        Ic = np.array(I.inertia)  # pinocchio stores the rotational inertia at the COM
        seg["inertia_com"] = [[float(v) for v in row] for row in Ic]
        seg.update(equivalent_extents(seg["mass"], Ic))
        offs = child_offsets[s]
        seg["child_offsets"] = offs
        seg["tip_len"] = tip[s] or None
        seg["length"] = float(max(offs)) if offs else None
        seg["n_children"] = len(offs)
        seg["parent_seg"] = parent_seg.get(s)
        seg["n_dof"] = len(seg["joints"])
        # World placement of the segment root at the neutral configuration, used
        # for robot-level size metrics (standing height, limb spans).
        jid = model.getJointId(seg["root_joint"])
        seg["pos_neutral"] = [float(x) for x in data.oMi[jid].translation]
        seg.pop("pin_joints")
    return segs


def _check_key_collisions(urls):
    """A model key may equal a repo key only if it names that same repository.

    `draft/cli/fetch.py` merges rows by key with the manifest winning, so a
    colliding key would fetch the model from the wrong repository.
    """
    bad = [(k, repo) for k, _, _, repo, _, _ in EXTRA_MODELS
           if k in urls and urls.get(k) != urls.get(repo)]
    if bad:
        raise SystemExit(
            "EXTRA_MODELS key collides with a different EXTRA_REPOS repository:\n" +
            "\n".join(f"  {k!r} would fetch from {urls.get(k)} but belongs to "
                       f"{repo!r} ({urls.get(repo)})" for k, repo in bad) +
            "\n  -> rename the model key.")


def extra_records(manifest: list[dict]) -> list[dict]:
    """Turn EXTRA_MODELS into manifest-shaped records with resolved paths."""
    commits = {r["key"]: r.get("commit") for r in manifest if r["source"] == "direct_clone"}
    urls = {r["key"]: r.get("repository") for r in manifest if r["source"] == "direct_clone"}
    _check_key_collisions(urls)
    recs = []
    for key, robot, maker, repo_key, rel, tags in EXTRA_MODELS:
        path = EXTRA_CACHE / repo_key / rel
        if not path.exists():
            print(f"  MISSING {key}: {path}")
            continue
        # Menagerie ships MJCF; everything else in this table is URDF.
        fmt = "mjcf" if path.suffix == ".xml" else "urdf"
        recs.append({
            "key": key, "source": "direct_clone", "robot": robot, "maker": maker,
            "repository": urls.get(repo_key, repo_key), "commit": commits.get(repo_key),
            "license_spdx": None, "synthetic": False, "tags": tags, fmt: str(path),
        })
    return recs


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    records = [r for r in manifest if r["source"] == "robot_descriptions" and "error" not in r]
    records += extra_records(manifest)
    out, failed = [], []
    for rec in records:
        if not (rec.get("urdf") or rec.get("mjcf")):
            continue
        try:
            model, fmt, path = load_model(rec)
            data = model.createData()
            pin.forwardKinematics(model, data, pin.neutral(model))
            segs = build_segments(model, data)
        except Exception as exn:
            failed.append((rec["key"], str(exn)[:160]))
            continue
        total = sum(s["mass"] for s in segs)
        out.append({
            "key": rec["key"], "robot": rec["robot"], "maker": rec["maker"],
            "repository": rec["repository"], "commit": rec.get("commit"),
            "license_spdx": rec.get("license_spdx"), "synthetic": rec.get("synthetic", False),
            "tags": rec.get("tags", []), "format": fmt, "file": portable(path),
            "n_segments": len(segs), "n_dof": sum(s["n_dof"] for s in segs),
            "total_mass_kg": total, "segments": segs,
        })
        bad = sum(1 for s in segs if s["inertia_valid"] is False)
        print(f"{rec['key']:32s} {fmt:4s} segs={len(segs):3d} dof={sum(s['n_dof'] for s in segs):3d} "
              f"m={total:7.2f} kg  bad_inertia={bad}", flush=True)

    write_json(DATA / "segments.json", out, indent=1)
    print(f"\nparsed {len(out)} robots -> data/segments.json")
    for k, e in failed:
        print(f"  FAILED {k}: {e}")


if __name__ == "__main__":
    sys.exit(main())
