"""The 3D half of the editor: a viser scene showing one generated model.

Owns the scene and physics loop and exposes the joints as data; the joint
sliders and simulate toggle live on the editor page (`editor/web/`), which runs
this class in the same process.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import viser
from mjviser.scene import ViserMujocoScene

from draft.generation import mjcf_assets


class Viewer3D:
    """A viser server showing one MuJoCo model. The editor page drives it."""

    def __init__(self, port: int = 8080, verbose: bool = True) -> None:
        self.server = viser.ViserServer(port=port)
        # viser's own panel cannot be hidden; `floating` overlays the canvas
        # instead of taking a column from it.
        self.server.gui.configure_theme(control_layout="floating",
                                        show_logo=False, show_share_button=False)
        # The bound port: viser moves to the next free one if `port` is taken.
        self.port = self.server.get_port()
        if verbose:
            print(f"[viewer] 3D scene on http://localhost:{self.port}", flush=True)

        self._mj_model: mujoco.MjModel | None = None
        self._mj_data: mujoco.MjData | None = None
        self._scene: ViserMujocoScene | None = None
        self._scene_lock = threading.Lock()
        self._simulate = False
        #: One entry per movable joint: name, limits, and its qpos address.
        #: Handed to the page, which draws the sliders.
        self._joints: list[dict[str, Any]] = []
        self._status = "Ready."

    # ── what the page drives ─────────────────────────────────────────────────

    def set_status(self, msg: str) -> None:
        self._status = msg

    @property
    def simulate(self) -> bool:
        return self._simulate

    def set_simulate(self, on: bool) -> None:
        """Run physics, or hold the slider pose (resetting on the way out)."""
        self._simulate = bool(on)
        if not self._simulate:
            self.reset_pose()

    def joints(self) -> list[dict]:
        """The movable joints of the loaded model, with their current angles."""
        out = []
        with self._scene_lock:
            for j in self._joints:
                q = (float(self._mj_data.qpos[j["qadr"]])
                     if self._mj_data is not None else 0.0)
                out.append({**j, "value": q})
        return out

    def set_joint(self, qadr: int, value: float) -> None:
        """Pose one joint. Ignored while physics is running, which owns qpos."""
        if self._simulate:
            return
        with self._scene_lock:
            if self._mj_data is None or self._mj_model is None:
                return
            if not 0 <= qadr < len(self._mj_data.qpos):
                return
            self._mj_data.qpos[qadr] = float(value)
            mujoco.mj_forward(self._mj_model, self._mj_data)
            if self._scene:
                self._scene.update_from_mjdata(self._mj_data)

    # ── model ─────────────────────────────────────────────────────────────────

    def load_model(self, mjcf_path: Path) -> bool:
        """Replace the 3D scene with a freshly-generated MJCF, in place."""
        scene_xml = Path(mjcf_path).parent / "scene.xml"
        load_path = scene_xml if scene_xml.is_file() else Path(mjcf_path)
        try:
            # Not `from_xml_path`: MuJoCo caches meshes by path, and every build
            # rewrites the same folder.
            model = mjcf_assets.load_model(load_path)
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            with self._scene_lock:
                self.server.scene.reset()
                self._scene = ViserMujocoScene(self.server, model, num_envs=1)
                self._mj_model = model
                self._mj_data = data
                self._scene.update_from_mjdata(data)
        except Exception as exc:
            self.set_status(f"**Error loading model:** {exc}")
            return False
        self._scan_joints()
        return True

    def reset_pose(self) -> None:
        with self._scene_lock:
            if self._mj_model and self._mj_data and self._scene:
                mujoco.mj_resetData(self._mj_model, self._mj_data)
                mujoco.mj_forward(self._mj_model, self._mj_data)
                self._scene.update_from_mjdata(self._mj_data)

    # ── joints ────────────────────────────────────────────────────────────────

    def _scan_joints(self) -> None:
        """Record the movable joints of the freshly-loaded model."""
        self._joints = []
        model = self._mj_model
        if model is None:
            return
        HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
        SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for jid in range(model.njnt):
            jtype = int(model.jnt_type[jid])
            if jtype not in (HINGE, SLIDE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            if model.jnt_limited[jid]:
                lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            else:
                # Unlimited: +-pi for a hinge, +-1 m for a slide.
                lo, hi = (-3.14159, 3.14159) if jtype == HINGE else (-1.0, 1.0)
            self._joints.append({
                "name": name, "lo": lo, "hi": hi,
                "qadr": int(model.jnt_qposadr[jid]),
                "unit": "rad" if jtype == HINGE else "m",
            })

    # ── loop ──────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Run the render/physics loop on a daemon thread."""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            with self._scene_lock:
                if self._scene and self._mj_data and self._mj_model:
                    if self._simulate:
                        mujoco.mj_step(self._mj_model, self._mj_data)
                    self._scene.update_from_mjdata(self._mj_data)
            dt = (self._mj_model.opt.timestep
                  if (self._simulate and self._mj_model) else 1.0 / 60.0)
            time.sleep(dt)
