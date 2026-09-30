#!/usr/bin/env python3
"""Stage 3 — map every segment onto a canonical humanoid body part.

A segment is named for the joint at its proximal end (hip -> thigh, knee ->
shank), so `length` is its link length. Joint names go through `GENERIC_ROLES`,
or a per-robot map in `ROBOT_ROLES` for uninformative names. Unmatched joints
are reported.

Output: `data/segments_classified.json` plus a coverage report on stdout.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from paths import DATA, ROOT, TRENDS_DATA  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

# Joint role -> the segment that joint drives.
ROLE_TO_SEGMENT = {
    "hip": "thigh", "knee": "shank", "ankle": "foot",
    "shoulder": "upper_arm", "elbow": "forearm", "wrist": "hand",
    "torso": "torso", "neck": "head",
    "digit": "digit", "aux": "aux",
    # quadruped: the abduction body is fit with the humanoid hip link.
    "abad": "hip_link", "q_hip": "thigh", "q_knee": "shank",
}

#: Short link between two joints of the same role (e.g. inside a 3-DOF hip) is
#: `<role>_link`; the waist's is renamed so it cannot be read as `torso`.
JOINT_LINK = {"torso": "waist_link"}

# Ordered — first match wins.
GENERIC_ROLES = [
    (r"finger|thumb|index|middle|ring|little|pinky|gripper|palm|_f_joint|pinkie"
     r"|^[LR]Hand$", "digit"),
    (r"eye|hokuyo|camera|lidar|imu|sensor|multisense|range_joint|handeye|wheel", "aux"),
    (r"hip|_HAA|_HFE|_HR$", "hip"),          # HAA/HFE/HR: ODRI + Berkeley convention
    (r"knee|_KFE", "knee"),
    (r"ankle|_ank|foot|toe|_FFE|_FAA", "ankle"),
    (r"shoulder|_shld|_sho_", "shoulder"),
    (r"elbow|_elb|_el$", "elbow"),
    (r"wrist|forearm", "wrist"),
    (r"waist|torso|trunk|chest|spine|lumbar|abdomen|back_bk", "torso"),
    (r"neck|head", "neck"),
]

# Robots whose joint names carry no anatomical information.
ROBOT_ROLES = {
    "talos_description": [
        (r"leg_\w+_[123]_joint", "hip"), (r"leg_\w+_4_joint", "knee"),
        (r"leg_\w+_[56]_joint", "ankle"),
        (r"arm_\w+_[123]_joint", "shoulder"), (r"arm_\w+_4_joint", "elbow"),
        (r"arm_\w+_[567]_joint", "wrist"),
        (r"torso_", "torso"), (r"head_", "neck"), (r"gripper", "digit"),
    ],
    "jaxon_description": [
        (r"LEG_JOINT[012]", "hip"), (r"LEG_JOINT3", "knee"), (r"LEG_JOINT[45]", "ankle"),
        (r"ARM_F_JOINT", "digit"),
        (r"ARM_JOINT[012]", "shoulder"), (r"ARM_JOINT3", "elbow"),
        (r"ARM_JOINT[4567]", "wrist"),
        (r"CHEST", "torso"), (r"HEAD", "neck"), (r"RANGE", "aux"),
    ],
    "atlas_drc_description": [
        (r"leg_hp", "hip"), (r"leg_kny", "knee"), (r"leg_ak", "ankle"),
        (r"arm_sh", "shoulder"), (r"arm_el", "elbow"),
        (r"arm_(uwy|mwx|lwy|wr)", "wrist"),
        (r"back_bk", "torso"), (r"neck", "neck"),
    ],
    "r2_description": [
        (r"hand|finger|thumb|index|middle|ring|little", "digit"),
        (r"waist", "torso"), (r"neck|head", "neck"),
        (r"leg/joint[012]", "hip"), (r"leg/joint3", "knee"), (r"leg/joint[456]", "ankle"),
        (r"arm/joint[012]", "shoulder"), (r"arm/joint3", "elbow"),
        (r"arm/(joint4|wrist)", "wrist"),
    ],
    "azureloong": [
        (r"J_arm_\w_0[123]", "shoulder"), (r"J_arm_\w_04", "elbow"),
        (r"J_arm_\w_0[567]", "wrist"),
    ],
    "zq_sa01": [
        (r"leg_\w[123]_joint", "hip"), (r"leg_\w4_joint", "knee"),
        (r"leg_\w[56]_joint", "ankle"),
    ],
    "robotera_xbot": [
        (r"_leg_(roll|yaw|pitch)_joint", "hip"),   # the 3-DOF hip, spelled "leg"
        (r"_arm_yaw_joint", "shoulder"),           # 3rd shoulder DOF, spelled "arm"
    ],
    "robotera_star1": [
        (r"_arm_yaw_joint", "shoulder"),
        (r"_hand_mid_joint", "digit"),             # "mid", not "middle"
    ],
    # Unnamed arm joints 1 and 3 are shoulder pitch and yaw.
    "tienkung_pro": [
        (r"_joint[13]$", "shoulder"), (r"_joint[567]$", "wrist"),
    ],
    "tienkung2_pro": [(r"body_yaw_joint", "torso")],
    "limx_oli_d03": [(r"_hand_yaw_joint", "wrist")],
    # Numbered joints; the `_bar_` joints are passive linkage bars.
    "kuavo_s45": [
        (r"_bar_\d", "aux"),
        (r"leg_[lr][123]_joint", "hip"), (r"leg_[lr]4_joint", "knee"),
        (r"leg_[lr][56]_joint", "ankle"),
        (r"zarm_[lr][123]_joint", "shoulder"), (r"zarm_[lr]4_joint", "elbow"),
        (r"zarm_[lr][567]_joint", "wrist"),
        (r"zhead_", "neck"),
    ],
    # HighTorque names the third hip DOF `thigh` and the knee `calf`.
    "hightorque_hi": [
        (r"floating_base", "aux"),
        (r"_hip_(pitch|roll)", "hip"), (r"_hip_thigh", "hip"),
        (r"_hip_calf", "knee"),
        (r"_upper_arm_", "shoulder"),
    ],
    "hightorque_pi_plus": [
        (r"floating_base", "aux"),
        (r"_hip_(pitch|roll)", "hip"), (r"_thigh_joint", "hip"),
        (r"_calf_joint", "knee"),
        (r"_upper_arm_", "shoulder"),
    ],
    "romeo_description": [
        (r"Finger|Thumb|LHand|RHand", "digit"), (r"Eye", "aux"),
        (r"Trunk", "torso"), (r"Neck|Head", "neck"),
    ],
    # Quadrupeds whose words collide across vendors (Unitree `hip` = abduction,
    # ZSIBot `HIP` = hip flexion), so they need per-robot tables.
    "zsibot_zsl1": [
        (r"_ABAD", "abad"), (r"_HIP_", "q_hip"), (r"_KNEE_", "q_knee"),
    ],
    # DFKI names the hip-flexion joint `shoulder`.
    "dfki_quad": [
        (r"_abad", "abad"), (r"_shoulder", "q_hip"), (r"_knee", "q_knee"),
    ],
    "dobot_rover_x1": [
        (r"_abad", "abad"), (r"_thigh_pitch", "q_hip"), (r"_calf_pitch", "q_knee"),
    ],
}
ROBOT_ROLES["atlas_v4_description"] = ROBOT_ROLES["atlas_drc_description"]
ROBOT_ROLES["kuavo_s53"] = ROBOT_ROLES["kuavo_s45"]

# Quadruped naming -> (abad, q_hip, q_knee). First match wins; vendor-specific
# rules precede loose word matches.
QUAD_ROLES = [
    # MIT Mini Cheetah — joints named for the two bodies they connect
    (r"torso_to_abduct", "abad"), (r"abduct_\w+_to_thigh", "q_hip"),
    (r"thigh_\w+_to_knee", "q_knee"),
    # Explicit roll/pitch spelling
    (r"hip_roll|hip_ab", "abad"), (r"hip_pitch|hip_fe|hip_flex", "q_hip"),
    # Unitree (A1/A2/Go1/Go2/Aliengo/B1/B2/Laikago)
    (r"_hip_", "abad"), (r"_thigh_", "q_hip"), (r"_calf_", "q_knee"),
    # ANYmal B/C/D and ODRI Solo: HAA / HFE / KFE
    (r"_HAA", "abad"), (r"_HFE", "q_hip"), (r"_KFE", "q_knee"),
    # Boston Dynamics Spot: hx = abduction, hy = hip flexion, kn = knee
    (r"_hx$", "abad"), (r"_hy$", "q_hip"), (r"_kn$", "q_knee"),
    # Google Barkour v0/vB
    (r"abduction", "abad"), (r"^hip_", "q_hip"), (r"^knee_", "q_knee"),
    (r"knee", "q_knee"),
    # `abad` (ZSIBot, DFKI, Dobot)
    (r"abad", "abad"),
    (r"_foot|toe|wheel", "aux"),
    (r"payload|_pan|_tilt", "aux"),      # ANYmal D inspection head
    (r"head|neck", "aux"),               # MagicDog's pan head — a mast, not a body part
]

SIDE_PATTERNS = [(r"(^|[_/])l($|[_/])|left|^L(?=[A-Z])|_l_|(^|_)l_", "L"),
                 (r"(^|[_/])r($|[_/])|right|^R(?=[A-Z])|_r_|(^|_)r_", "R")]

# Robots excluded from the structural fit, with the reason (kept in the dataset).
EXCLUDE = {
    "atlas_drc_description": "hydraulic actuation — mass distribution not comparable to electric",
    "atlas_v4_description": "hydraulic actuation — mass distribution not comparable to electric",
    "r2_description": "space robot; 'legs' are 7-DOF climbing limbs ending in grippers",
    "mujoco_humanoid_mj_description": "synthetic benchmark model, not hardware",
    "simple_humanoid_description": "pedagogical model, box links",
    "rhea_description": "hobby build without vendor inertials",
    "spryped_description": "hobby build without vendor inertials",
    "cassie_description": "leaf-spring/linkage leg; motor mass sits in the pelvis, not at the joint",
    "cassie_mj_description": "leaf-spring/linkage leg; motor mass sits in the pelvis, not at the joint",
    "unitree_g1_d": "wheeled-base G1 variant — no legs to fit",
    "cookie_description": "wheeled balancer, 6 DOF, no upper body",
    "upkie_description": "wheeled balancer, 6 DOF, no upper body",
    "bolt_description": "6-DOF research biped, no upper body or ankle",
    # MJCF duplicates of a URDF sibling already in the set
    "booster_t1_mj_description": "duplicate of booster_t1_description (URDF preferred)",
    "g1_mj_description": "duplicate of g1_description",
    "h1_mj_description": "duplicate of h1_description",
    "h1_2_mj_description": "duplicate of h1_2_description",
    "n1_mj_description": "duplicate of n1_description",
    "jvrc_mj_description": "duplicate of jvrc_description",
    "toddlerbot_2xc_mj_description": "duplicate variant of toddlerbot_description",
    "toddlerbot_2xm_mj_description": "duplicate variant of toddlerbot_description",
    # Quadrupeds: Unitree comes from the unitree_ros clone (`unitree_*` keys), so
    # the catalog entries and MJCF siblings are duplicates.
    "a1_description": "duplicate of unitree_a1 (same unitree_ros URDF)",
    "aliengo_description": "duplicate of unitree_aliengo",
    "b1_description": "duplicate of unitree_b1",
    "b2_description": "duplicate of unitree_b2",
    "go1_description": "duplicate of unitree_go1",
    "go2_description": "duplicate of unitree_go2",
    "laikago_description": "duplicate of unitree_laikago",
    "a1_mj_description": "MJCF duplicate of unitree_a1",
    "aliengo_mj_description": "MJCF duplicate of unitree_aliengo",
    "go1_mj_description": "MJCF duplicate of unitree_go1",
    "go2_mj_description": "MJCF duplicate of unitree_go2",
    "anymal_b_mj_description": "MJCF duplicate of anymal_b_description",
    "anymal_c_mj_description": "MJCF duplicate of anymal_c_description",
    "hyq_description": "hydraulic actuation — mass distribution not comparable to electric",
}

# Models flagged in data/active_flags.json are excluded like EXCLUDE entries.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from active_flags import load_inactive  # noqa: E402

INACTIVE = load_inactive(ROOT / "data" / "active_flags.json",
                         expect_dataset="datasets/robot_descriptions/data/manifest.json")
if INACTIVE:
    print(f"active_flags: {len(INACTIVE)} model(s) flagged inactive")
    for k, why in sorted(INACTIVE.items()):
        print(f"  - {k}: {why}")


def joint_role(robot_key: str, name: str, is_quad: bool) -> str | None:
    # A robot's own table wins over the shared ones.
    tables = []
    if robot_key in ROBOT_ROLES:
        tables.append(ROBOT_ROLES[robot_key])
    if is_quad:
        tables.append(QUAD_ROLES)
    tables.append(GENERIC_ROLES)
    for table in tables:
        for pat, role in table:
            if re.search(pat, name, re.IGNORECASE):
                return role
    return None


def leg_length(segments: list[dict]) -> float | None:
    """Median thigh + shank length: the quadruped's size scale (`z_span` depends
    on the shipped neutral pose, so is not one)."""
    # Terminal shank: fall back to the foot frame, then the inertia box's long axis.
    def _len(s):
        return s.get("length") or s.get("tip_len") or s.get("box_a")

    thighs = [s for s in segments if s["seg_class"] == "thigh" and _len(s)]
    shanks = [s for s in segments if s["seg_class"] == "shank" and _len(s)]
    if not thighs or not shanks:
        return None
    legs = sorted(_len(t) for t in thighs)
    calves = sorted(_len(s) for s in shanks)
    n = min(len(legs), len(calves))
    pairs = sorted(legs[:n][i] + calves[:n][i] for i in range(n))
    return float(pairs[len(pairs) // 2])


def body_length(segments: list[dict]) -> float | None:
    """Fore-aft span of the leg attachment points — the trunk's structural length."""
    xs = [s["pos_neutral"][0] for s in segments
          if s["seg_class"] in ("hip_link", "thigh")]
    return float(max(xs) - min(xs)) if len(xs) >= 2 else None


def side_of(name: str) -> str | None:
    for pat, side in SIDE_PATTERNS:
        if re.search(pat, name):
            return side
    return None


def classify(r: dict, unmapped: Counter | None = None) -> dict:
    """Label one robot record in place: joint roles, segment classes, size metrics."""
    is_quad = "quadruped" in r.get("tags", [])
    r["excluded"] = EXCLUDE.get(r["key"]) or INACTIVE.get(r["key"])
    r["is_quadruped"] = is_quad
    for seg in r["segments"]:
        for j in seg["joints"]:
            j["role"] = joint_role(r["key"], j["name"], is_quad)
            if j["role"] is None and unmapped is not None:
                unmapped[f"{r['key']}:{j['name']}"] += 1
        root_role = next((j["role"] for j in seg["joints"]
                          if j["name"] == seg["root_joint"]), None)
        if root_role is None and seg["joints"]:
            root_role = seg["joints"][0]["role"]
        seg["root_role"] = root_role
        seg["side"] = side_of(seg["root_joint"])

    # A spread-out 3-DOF hip yields several `hip` segments; only the one whose
    # child is the knee is the thigh, the rest are `hip_link`. Same for other stacks.
    children = {}
    for seg in r["segments"]:
        children.setdefault(seg["parent_seg"], []).append(seg)
    for seg in r["segments"]:
        if seg["parent_seg"] is None:
            seg["seg_class"] = "pelvis"
            continue
        role = seg["root_role"]
        kids = children.get(seg["idx"], [])
        roles = {j["role"] for j in seg["joints"]}
        if is_quad and "q_hip" in roles:
            # Abduction and hip-flexion merged into one segment (e.g. Solo): a thigh.
            seg["seg_class"] = "thigh"
        elif role and any(k["root_role"] == role for k in kids):
            seg["seg_class"] = JOINT_LINK.get(role, f"{role}_link")
        else:
            seg["seg_class"] = ROLE_TO_SEGMENT.get(role, "unknown")

    # robot-level size metrics from the neutral-pose FK
    zs = [s["pos_neutral"][2] for s in r["segments"]]
    r["z_span_m"] = float(max(zs) - min(zs)) if zs else None
    r["leg_length_m"] = leg_length(r["segments"])
    r["body_length_m"] = body_length(r["segments"])
    r["class_counts"] = dict(Counter(s["seg_class"] for s in r["segments"]))
    return r


def main() -> None:
    robots = json.loads((DATA / "segments.json").read_text())
    unmapped = Counter()
    for r in robots:
        classify(r, unmapped)

    write_json(DATA / "segments_classified.json", robots, indent=1)

    kept = [r for r in robots if not r["excluded"]]
    print(f"{len(kept)} robots kept, {len(robots) - len(kept)} excluded\n")
    print(f"{'robot':32s} {'mass':>7s} {'zspan':>6s}  segment classes")
    for r in sorted(kept, key=lambda x: -x["total_mass_kg"]):
        cc = {k: v for k, v in sorted(r["class_counts"].items()) if k not in ("digit", "aux")}
        print(f"{r['key']:32s} {r['total_mass_kg']:7.2f} {r['z_span_m'] or 0:6.2f}  {cc}")
    if unmapped:
        print(f"\nUNMAPPED JOINTS ({len(unmapped)}):")
        for k in sorted(unmapped):
            print(f"  {k}")


if __name__ == "__main__":
    sys.exit(main())
