"""Inertia-mismatch domain randomisation: each replica gets the inertias of a
plausible built copy of the design (see ``draft.twins.inertia_perturb``).

One startup event per anatomical group; each replica draws once per group and
applies it to all four legs. The math runs in numpy on the CPU, once at startup.

    from draft.tasks.quadruped.inertia_dr import add_inertia_mismatch_events
    add_inertia_mismatch_events(cfg, robot_xml, leg_length=ctx.leg_length, seed=0)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from draft.twins.inertia_perturb import (body_groups, leg_side, load_spec,
                                         perturb_body, resolve)

#: Fields the perturbation writes; declared so mjlab recomputes derived constants.
_FIELDS = ("body_mass", "body_ipos", "body_inertia", "body_iquat")


@requires_model_fields(*_FIELDS, recompute=RecomputeLevel.set_const)
def group_inertia_mismatch(
    env,
    env_ids,
    alpha_range: tuple[float, float],
    d_range: tuple[float, float],
    com_lo: tuple[float, float, float],
    com_hi: tuple[float, float, float],
    sides: tuple[float, ...],
    seed: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Perturb ONE anatomical group, one draw per replica shared over its bodies.

    `sides` is +1 / -1 per body, in `asset_cfg.body_names` order, and mirrors
    the lateral COM shift onto the right legs.
    """
    import mujoco

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    ids = asset.indexing.body_ids[asset_cfg.body_ids]
    n_env, n_body = len(env_ids), len(ids)
    assert len(sides) == n_body, (
        f"{n_body} bodies resolved but {len(sides)} sides given — the "
        f"SceneEntityCfg must carry preserve_order=True")

    def default(field):
        d = env.sim.get_default_field(field)
        if field in env.sim.per_world_default_fields:
            eg, bg = torch.meshgrid(env_ids, ids, indexing="ij")
            return d[eg, bg].detach().cpu().numpy()
        return np.broadcast_to(d[ids].detach().cpu().numpy()[None],
                               (n_env,) + tuple(d[ids].shape)).copy()

    mass0, ipos0 = default("body_mass"), default("body_ipos")
    inertia0, iquat0 = default("body_inertia"), default("body_iquat")

    # Seeded by run seed and group ranges, so groups get distinct reproducible streams.
    rng = np.random.default_rng([seed, int(abs(hash(tuple(alpha_range))) % (2 ** 31))])
    alpha = rng.uniform(*alpha_range, size=n_env)
    d = rng.uniform(*d_range, size=n_env)
    shift = rng.uniform(np.asarray(com_lo), np.asarray(com_hi), size=(n_env, 3))

    m_out = np.empty((n_env, n_body))
    c_out = np.empty((n_env, n_body, 3))
    I_out = np.empty((n_env, n_body, 3))
    q_out = np.empty((n_env, n_body, 4))
    R = np.zeros(9)
    for e in range(n_env):
        for b in range(n_body):
            mujoco.mju_quat2Mat(R, np.ascontiguousarray(iquat0[e, b], dtype=float))
            Rm = R.reshape(3, 3)
            I0 = Rm @ np.diag(inertia0[e, b]) @ Rm.T
            want = shift[e] * np.array([1.0, sides[b], 1.0])
            t = np.exp(alpha[e]) * (want - (np.exp(d[e]) - 1.0) * ipos0[e, b])
            m1, c1, I1 = perturb_body(mass0[e, b], ipos0[e, b], I0,
                                      alpha[e], d[e], t)
            w, V = np.linalg.eigh(I1)
            if np.linalg.det(V) < 0:
                V[:, 0] = -V[:, 0]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, V.flatten())
            m_out[e, b], c_out[e, b] = m1, c1
            I_out[e, b], q_out[e, b] = np.maximum(w, 1e-12), q

    eg, bg = torch.meshgrid(env_ids, ids, indexing="ij")
    dev, dt = env.device, env.sim.model.body_mass.dtype
    to = lambda a: torch.as_tensor(a, device=dev, dtype=dt)
    env.sim.model.body_mass[eg, bg] = to(m_out)
    env.sim.model.body_ipos[eg, bg] = to(c_out)
    env.sim.model.body_inertia[eg, bg] = to(I_out)
    env.sim.model.body_iquat[eg, bg] = to(q_out)


def add_inertia_mismatch_events(cfg, robot_xml: str | Path, leg_length: float,
                                spec: dict | None = None, seed: int = 0) -> dict:
    """Register one event per group; return what was registered, for the record.

    The spec is resolved once, against a freshly compiled copy of the robot.
    """
    import mujoco

    spec = spec or load_spec()
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    groups = body_groups(model)
    registered = {}
    for gr in resolve(model, leg_length, spec):
        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in gr.body_ids)
        sides = tuple(leg_side(model, b) for b in gr.body_ids)
        cfg.events[f"inertia_mismatch_{gr.group}"] = EventTermCfg(
            mode="startup", func=group_inertia_mismatch,
            params={"alpha_range": tuple(gr.alpha), "d_range": tuple(gr.d),
                    "com_lo": tuple(gr.com_lo), "com_hi": tuple(gr.com_hi),
                    "sides": sides, "seed": seed,
                    "asset_cfg": SceneEntityCfg("robot", body_names=names,
                                                preserve_order=True)})
        registered[gr.group] = {
            "bodies": list(names),
            "mass_ratio": [float(np.exp(2 * a)) for a in gr.alpha],
            "dilation": [float(x) for x in gr.d],
            "com_shift_m": [gr.com_lo.tolist(), gr.com_hi.tolist()],
        }
    if not registered:
        raise RuntimeError(f"no anatomical groups found in {robot_xml} — "
                           f"body_groups returned {list(groups)}")
    return registered
