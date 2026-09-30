"""Move a model's floating base from one body to another.

mjlab reads base velocities and projected gravity off the root body, so a
vendor model (pelvis-rooted) must share its twin's root (``torso_link``) for a
policy comparison to be valid. The trunk chain is reversed: each body keeps its
frame, parent links flip, and each joint moves one body up the chain with its
axis negated so a given ``q`` means the same pose. :func:`verify_reroot` checks
the result by FK.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class _Frames:
    """World frames of every body at ``q = 0``, from the compiled model."""

    def __init__(self, model: mujoco.MjModel):
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        if model.nq >= 7 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)
        self.pos, self.quat = {}, {}
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self.pos[name] = np.array(data.xpos[i])
                self.quat[name] = np.array(data.xquat[i])
        self.jnt_anchor, self.jnt_axis = {}, {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.jnt_anchor[name] = np.array(data.xanchor[i])
                self.jnt_axis[name] = np.array(data.xaxis[i])

    def relative(self, parent: str, child: str) -> tuple[np.ndarray, np.ndarray]:
        """Transform of ``child`` expressed in ``parent``'s frame."""
        r_p = _mat(self.quat[parent])
        pos = r_p.T @ (self.pos[child] - self.pos[parent])
        quat = _quat_mul(_quat_conj(self.quat[parent]), self.quat[child])
        return pos, quat

    def into(self, body: str, point: np.ndarray, direction: np.ndarray):
        """A world point and direction, expressed in ``body``'s frame."""
        r_b = _mat(self.quat[body])
        return r_b.T @ (point - self.pos[body]), r_b.T @ direction


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.asarray(v).ravel())


def reroot(xml_path: Path, new_root: str, out_path: Path) -> Path:
    """Rewrite ``xml_path`` so ``new_root`` carries the free joint.

    ``new_root`` must descend from the current root through single-joint bodies;
    subtrees off that chain are carried along untouched.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    frames = _Frames(model)

    tree = ET.parse(str(xml_path))
    root_el = tree.getroot()
    worldbody = root_el.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path} has no <worldbody>")

    parent_of: dict[ET.Element, ET.Element] = {}
    by_name: dict[str, ET.Element] = {}
    for parent in root_el.iter():
        for child in parent:
            if child.tag == "body":
                parent_of[child] = parent
                by_name[str(child.get("name"))] = child

    old_root = next((b for b in worldbody if b.tag == "body"), None)
    if old_root is None:
        raise ValueError(f"{xml_path} has no top-level body")
    if new_root not in by_name:
        raise KeyError(f"body '{new_root}' not found in {xml_path}")

    # The trunk chain, old root -> new root.
    chain: list[ET.Element] = []
    node: ET.Element | None = by_name[new_root]
    while node is not None:
        chain.append(node)
        if node is old_root:
            break
        node = parent_of.get(node)
    else:
        raise ValueError(f"'{new_root}' is not a descendant of '{old_root.get('name')}'")
    chain.reverse()  # old_root ... new_root
    if len(chain) < 2:
        return Path(xml_path)

    names = [str(b.get("name")) for b in chain]

    # Detach the chain: each link stops being its parent's child.
    for body in chain[1:]:
        parent_of[body].remove(body)
    worldbody.remove(old_root)

    # The joint on edge chain[i] -> chain[i+1] moves from chain[i+1] to chain[i].
    moved_joints: list[list[ET.Element]] = []
    for body in chain[1:]:
        joints = [j for j in body if j.tag in ("joint", "freejoint")]
        for j in joints:
            body.remove(j)
        moved_joints.append(joints)
    # The old root's free joint goes to the new root.
    free_joints = [j for j in chain[0] if j.tag == "freejoint"
                   or (j.tag == "joint" and j.get("type") == "free")]
    for j in free_joints:
        chain[0].remove(j)

    # Rebuild in reverse order: new_root -> ... -> old_root.
    rebuilt = list(reversed(chain))
    rebuilt_names = list(reversed(names))
    for i, body in enumerate(rebuilt):
        if i == 0:
            worldbody.append(body)
            pos, quat = frames.pos[rebuilt_names[0]], frames.quat[rebuilt_names[0]]
            body.set("pos", _fmt(pos))
            body.set("quat", _fmt(quat))
            for j in free_joints:
                body.insert(0, j)
        else:
            parent_name, child_name = rebuilt_names[i - 1], rebuilt_names[i]
            rebuilt[i - 1].append(body)
            pos, quat = frames.relative(parent_name, child_name)
            body.set("pos", _fmt(pos))
            body.set("quat", _fmt(quat))
            # Moved joint: anchor/axis re-expressed in this frame, axis and range negated.
            for j in moved_joints[len(chain) - 1 - i]:
                jname = str(j.get("name"))
                anchor, axis = frames.into(child_name, frames.jnt_anchor[jname],
                                           frames.jnt_axis[jname])
                j.set("pos", _fmt(anchor))
                j.set("axis", _fmt(-axis))
                rng = j.get("range")
                if rng is not None:
                    lo, hi = (float(x) for x in rng.split())
                    j.set("range", _fmt([-hi, -lo]))
                body.insert(0, j)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="unicode", xml_declaration=False)
    return out_path


def verify_reroot(original: Path, rerooted: Path, n_samples: int = 24,
                  seed: int = 0) -> tuple[float, float]:
    """Compare FK of two models over random joint vectors, root-aligned.

    Returns (worst position error in m, worst orientation error in deg).
    """
    m_a = mujoco.MjModel.from_xml_path(str(original))
    m_b = mujoco.MjModel.from_xml_path(str(rerooted))
    d_a, d_b = mujoco.MjData(m_a), mujoco.MjData(m_b)

    def hinges(model):
        return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
                for i in range(model.njnt)
                if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,
                                         mujoco.mjtJoint.mjJNT_SLIDE)}

    h_a, h_b = hinges(m_a), hinges(m_b)
    shared_joints = sorted(set(h_a) & set(h_b))
    bodies = sorted(
        {mujoco.mj_id2name(m_a, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m_a.nbody)} &
        {mujoco.mj_id2name(m_b, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m_b.nbody)} -
        {None, "world"})

    rng = np.random.default_rng(seed)
    worst_pos = worst_ang = 0.0
    for k in range(n_samples):
        d_a.qpos[:] = 0.0
        d_b.qpos[:] = 0.0
        for d, m in ((d_a, m_a), (d_b, m_b)):
            if m.nq >= 7 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
                d.qpos[3] = 1.0
        if k:  # first sample is the neutral pose
            for name in shared_joints:
                q = float(rng.uniform(-0.4, 0.4))
                d_a.qpos[h_a[name]] = q
                d_b.qpos[h_b[name]] = q
        mujoco.mj_forward(m_a, d_a)
        mujoco.mj_forward(m_b, d_b)

        # Align on a shared body, then compare every other body relative to it.
        ref = bodies[0]
        ia = mujoco.mj_name2id(m_a, mujoco.mjtObj.mjOBJ_BODY, ref)
        ib = mujoco.mj_name2id(m_b, mujoco.mjtObj.mjOBJ_BODY, ref)
        r_a, r_b = _mat(d_a.xquat[ia]), _mat(d_b.xquat[ib])
        for name in bodies:
            ja = mujoco.mj_name2id(m_a, mujoco.mjtObj.mjOBJ_BODY, name)
            jb = mujoco.mj_name2id(m_b, mujoco.mjtObj.mjOBJ_BODY, name)
            pa = r_a.T @ (d_a.xpos[ja] - d_a.xpos[ia])
            pb = r_b.T @ (d_b.xpos[jb] - d_b.xpos[ib])
            worst_pos = max(worst_pos, float(np.linalg.norm(pa - pb)))
            qa = _quat_mul(_quat_conj(d_a.xquat[ia]), d_a.xquat[ja])
            qb = _quat_mul(_quat_conj(d_b.xquat[ib]), d_b.xquat[jb])
            dq = _quat_mul(_quat_conj(qa), qb)
            worst_ang = max(worst_ang,
                            float(np.degrees(2.0 * np.arccos(min(1.0, abs(dq[0]))))))
    return worst_pos, worst_ang
