"""One port serving the editor page (from `static/`) and its websocket state channel.

The page computes nothing: it sends edits and receives the whole re-solved
state, so the fitted trends are solved in one place.
"""

from __future__ import annotations

import asyncio
import json
import math
import mimetypes
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from draft.paths import repo_root
_REPO_ROOT = repo_root()
from draft.trends import motor_solve
from draft.editor.core.schema import ParamKind
from draft.editor.core.session import EditorSession
from draft.editor.core import fits
from draft.editor.shared import discover_robot_types
from draft.editor.viewer import Viewer3D


_STATIC = Path(__file__).parent / "static"

#: Seconds of quiet after an edit before the model is rebuilt.
REGEN_DEBOUNCE = 0.6

#: Points along the speed axis of the torque–speed plot.
_PLOT_SAMPLES = 72


def _finite(x) -> float | None:
    """JSON has no NaN. A gap in a curve is `null`, which the client skips."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class WebEditor:
    """The design being edited, as seen by any number of browser tabs.

    One :class:`~draft.editor.core.session.EditorSession` per robot type, kept
    so switching types does not lose edits.
    """

    def __init__(self, viewer: Viewer3D, repo_root: Path = _REPO_ROOT) -> None:
        self.repo_root = repo_root
        self.viewer = viewer
        self.generated_base = repo_root / "generated"
        self.generated_base.mkdir(parents=True, exist_ok=True)

        self.configs = discover_robot_types(repo_root)
        if not self.configs:
            raise RuntimeError(f"No robots found under {repo_root}/src/draft/robots")
        self._sessions: dict[str, EditorSession] = {}
        self.robot = sorted(self.configs)[0]
        self.status = {"text": "Ready.", "kind": "info"}
        self.report: dict | None = None

        self._clients: set[ServerConnection] = set()
        self._regen_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # ── sessions ──────────────────────────────────────────────────────────────

    @property
    def session(self) -> EditorSession:
        if self.robot not in self._sessions:
            s = EditorSession(self.configs[self.robot], self.generated_base)
            s.load_selected_folder()
            self._sessions[self.robot] = s
        return self._sessions[self.robot]

    # ── state → JSON ──────────────────────────────────────────────────────────

    def _param_json(self, spec) -> dict:
        s = self.session
        value = s.get(spec.key, spec.default)
        out: dict[str, Any] = {
            "key": spec.key,
            "kind": spec.kind.value,
            "owner": spec.owner.value,
            "hint": spec.hint,
            "value": value,
            "default": spec.default,
            "changed": _changed(value, spec.default),
        }
        if spec.kind is ParamKind.MOTOR_CLASS:
            out["options"] = list(s.schema.motor_classes)
        return out

    def _motors_json(self) -> dict:
        s = self.session
        try:
            designs = s.motors.designs()
        except Exception as exc:
            return {"error": str(exc), "classes": [], "plot": None}

        classes = []
        for d in designs:
            dens = d.densities()
            td, pd = dens["torque_density"], dens["power_density"]
            stated = s.motors.collect()
            classes.append({
                "cls": d.cls,
                "mode": d.point.mode,
                # Mode-stated inputs plus those no mode derives (aspect ratio).
                "active": list(motor_solve.MODE_INPUTS[d.point.mode]
                               + motor_solve.FREE_INPUTS),
                "tau": _finite(d.point.tau),
                "omega": _finite(d.point.omega),
                "gear": _finite(d.point.gear),
                "power": _finite(d.point.power),
                "gear_source": d.point.gear_source,
                "friction": stated.get(f"{d.cls}_motor_friction", 0.0),
                "damping": stated.get(f"{d.cls}_motor_damping", 0.0),
                "mass": _finite(d.mass),
                "r": _finite(d.sized["r"]),
                "L": _finite(d.sized["L"]),
                "aspect": _finite(d.sized["aspect"]),
                "volume": _finite(d.sized["volume"]),
                "armature": _finite(d.sized["armature"]),
                "category": dens["category"],
                "torque_density": _finite(td["value"]),
                "torque_status": td["status"],
                "torque_p90": _finite(td["p90"]),
                "torque_max": _finite(td["max"]),
                "power_density": _finite(pd["value"]),
                "power_status": pd["status"],
                "power_p90": _finite(pd["p90"]),
                "power_max": _finite(pd["max"]),
                "tau_ceiling": _finite(d.ceilings()["tau_max"]),
                "warnings": list(d.warnings),
                "errors": list(d.errors),
            })

        omegas = list(motor_solve.speed_axis(designs, n=_PLOT_SAMPLES))
        plot = {
            "omegas": [_finite(o) for o in omegas],
            "series": [{
                "cls": d.cls,
                "envelope": [_finite(v) for v in motor_solve.envelope_curve(d.point, omegas)],
                "frontier": [_finite(v) for v in motor_solve.frontier_curve(d, omegas)],
            } for d in designs],
        }
        return {
            "classes": classes,
            "plot": plot,
            "modes": [{"key": m, "label": motor_solve.MODE_LABELS[m]} for m in motor_solve.MODES],
            "total_mass": sum(d.mass for d in designs),
            "allow_hypothetical": s.allow_hypothetical,
        }

    def _densities_json(self) -> dict:
        s = self.session
        band = s.schema.link_density_band(s.values)
        return {
            "derived": s.schema.law_owned_densities,
            "band": ({"lo": band[0], "hi": band[1],
                      "value": s.get("link_rho"),
                      "population": s.values.get("mass_composition_reference")}
                     if band is not None and "link_rho" in s.values else None),
        }

    def state(self) -> dict:
        base = {
            "type": "state",
            "robots": sorted(self.configs),
            "robot": self.robot,
            "viser_url": f"http://localhost:{self.viewer.port}/?hideViserLogo",
            "status": self.status,
            "unsupported": None,
        }
        # A robot that fails to load (e.g. missing trend data) is reported on
        # the page; the other robots keep working.
        try:
            s = self.session
        except Exception as exc:
            base["unsupported"] = (f"{self.robot} could not be loaded — "
                                   f"{type(exc).__name__}: {exc}")
            return base
        # One solve feeds both the actuator table and the trend panels.
        motors = self._motors_json()
        base.update({
            "folders": s.folders.names,
            "folder": s.folders.selected,
            "groups": [{"title": g, "params": [self._param_json(sp) for sp in specs]}
                       for g, specs in s.schema.groups().items()
                       if g not in ("Actuator Design", "Density (kg/m³)")],
            "densityGroup": [self._param_json(sp)
                             for sp in s.schema.of_kind(ParamKind.DENSITY) if sp.editable],
            "motors": motors,
            # The static catalogue and curves are sent once, on connect.
            "design_fits": fits.design_points(motors),
            "densities": self._densities_json(),
            # The model's joints and whether physics is running.
            "joints": self.viewer.joints(),
            "simulate": self.viewer.simulate,
            "preset": self.configs[self.robot].preset_name,
            "report": self.report,
        })
        return base

    # ── broadcast ─────────────────────────────────────────────────────────────

    def _safe_state(self) -> dict:
        """`state()`, but a failure becomes an error message, not a dropped connection."""
        try:
            return self.state()
        except Exception as exc:
            return {"type": "state", "robots": sorted(self.configs), "robot": self.robot,
            "viser_url": f"http://localhost:{self.viewer.port}/?hideViserLogo",
                    "status": {"text": f"{type(exc).__name__}: {exc}", "kind": "error"},
                    "unsupported": f"The editor could not describe this robot — {exc}"}

    async def broadcast(self) -> None:
        if not self._clients:
            return
        payload = json.dumps(self._safe_state())
        await asyncio.gather(*(c.send(payload) for c in list(self._clients)),
                             return_exceptions=True)

    def _set_status(self, text: str, kind: str = "info") -> None:
        self.status = {"text": text, "kind": kind}

    # ── actions ───────────────────────────────────────────────────────────────

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")

        if kind == "hello":
            pass

        elif kind == "select_robot":
            if msg["robot"] in self.configs:
                self.robot = msg["robot"]
                self._cancel_regen()
                self.report = None
                self._set_status(f"{self.robot} selected.")
                self._show_selected()

        elif kind == "select_folder":
            s = self.session
            s.folders.select(msg["folder"])
            s.load_selected_folder()
            self.report = None
            self._set_status(f"Loaded {s.folders.selected}.")
            self._show_selected()

        elif kind == "refresh_folders":
            self.session.folders.refresh()

        elif kind == "patch":
            self.session.set(msg["key"], msg["value"])
            self._schedule_regen()

        elif kind == "motor_patch":
            cls, field = msg["cls"], msg["field"]
            value = msg["value"] if field == "mode" else float(msg["value"])
            self.session.motors.set({f"{cls}_motor_{field}": value})
            self._schedule_regen()

        elif kind == "new_from_defaults":
            s = self.session
            s.folders.new_folder()
            s.reset_to_defaults()
            self._set_status(f"New {self.robot} from defaults — generating…")
            await self.broadcast()
            await self._generate()
            return

        elif kind == "reset_defaults":
            self.session.reset_to_defaults()
            self._set_status("Reset to the robot's shipped parameters.")
            self._schedule_regen()

        elif kind == "generate":
            self._cancel_regen()
            await self._generate()
            return

        elif kind == "save_as":
            name = str(msg.get("name") or "").strip()
            if not name:
                self._set_status("Saving needs a name.", "error")
            else:
                self._set_status(f"Saving {name}…")
                await self.broadcast()
                out = await asyncio.to_thread(self.session.save_as, name)
                self._set_status(f"Saved → {out.name}", "ok")

        elif kind == "set_joint":
            # No broadcast: a drag sends one per frame.
            self.viewer.set_joint(int(msg["qadr"]), float(msg["value"]))
            return

        elif kind == "simulate":
            self.viewer.set_simulate(bool(msg.get("on")))
            self._set_status("Simulating." if self.viewer.simulate
                             else "Physics stopped; pose reset.")

        elif kind == "reset_pose":
            self.viewer.reset_pose()
            self._set_status("Pose reset.")

        elif kind == "view":
            self._show_selected()

        await self.broadcast()

    def _show_selected(self) -> None:
        try:
            mjcf = self.session.folders.mjcf()
        except Exception:
            return          # `state()` reports why; the 3D tab just stays empty
        if mjcf is not None:
            self.viewer.load_model(mjcf)
            self.viewer.set_status(f"Viewing: **{self.session.folders.selected}**")

    # ── generation ────────────────────────────────────────────────────────────

    def _cancel_regen(self) -> None:
        if self._regen_task is not None and not self._regen_task.done():
            self._regen_task.cancel()
        self._regen_task = None

    def _schedule_regen(self) -> None:
        self._cancel_regen()
        self._set_status("Parameters changed — regenerating…")
        self._regen_task = asyncio.create_task(self._debounced())

    async def _debounced(self) -> None:
        try:
            await asyncio.sleep(REGEN_DEBOUNCE)
        except asyncio.CancelledError:
            return
        await self._generate()

    async def _generate(self) -> None:
        """Build the current design, serialised (builds share an output folder)."""
        async with self._lock:
            s = self.session
            self._set_status(f"Generating {self.robot}…")
            await self.broadcast()
            result = await asyncio.to_thread(s.generate)
            if not result.ok:
                self.report = {"refused": str(result.error)}
                self._set_status(f"Generation failed: {result.error}", "error")
            else:
                r = result.report
                self.report = ({"total_mass_kg": r.total_mass_kg, "mass_kg": r.mass_kg,
                                "checks": r.checks, "warnings": r.warnings}
                               if r is not None else None)
                self.viewer.load_model(result.mjcf_path)
                self.viewer.set_status(f"Viewing: **{result.output_dir.name}**")
                self._set_status(f"Generated {result.output_dir.name}.", "ok")
            await self.broadcast()

    # ── serving ───────────────────────────────────────────────────────────────

    async def _connection(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        try:
            first = self._safe_state()
            try:
                first["fits"] = fits.catalogue()
            except Exception as exc:
                first["fits"] = {"error": f"{type(exc).__name__}: {exc}"}
            await ws.send(json.dumps(first))
            async for raw in ws:
                try:
                    await self.handle(json.loads(raw))
                except Exception as exc:          # one bad message must not
                    self._set_status(f"{type(exc).__name__}: {exc}", "error")
                    await self.broadcast()        # stop the editor
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    @staticmethod
    def _static(_conn: ServerConnection, request) -> Response | None:
        """Answer a plain GET from `static/`; let an upgrade through to the socket."""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        path = request.path.split("?")[0]
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (_STATIC / name).resolve()
        if not target.is_file() or _STATIC.resolve() not in target.parents:
            return Response(404, "Not Found", Headers({"Content-Length": "0"}), b"")
        body = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return Response(200, "OK", Headers({
            "Content-Type": ctype,
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
        }), body)

    async def serve(self, port: int = 8081, tries: int = 16) -> None:
        for attempt in range(tries):
            try:
                await self._serve_on(port + attempt)
                return
            except OSError as exc:
                if exc.errno not in (48, 98) or attempt == tries - 1:  # EADDRINUSE
                    raise
                print(f"[web] port {port + attempt} in use, trying {port + attempt + 1}",
                      flush=True)

    async def _serve_on(self, port: int) -> None:
        async with serve(self._connection, "localhost", port,
                         process_request=self._static) as server:
            actual = next(iter(server.sockets)).getsockname()[1]
            print(f"[draft] editor    http://localhost:{actual}   <- open this",
                  flush=True)
            print(f"[draft] 3D scene  http://localhost:{self.viewer.port}   "
                  "(embedded in the page)", flush=True)
            await asyncio.get_running_loop().create_future()   # run forever


def _changed(value, default) -> bool:
    """Is this value different from what the robot ships with?"""
    if isinstance(value, float) and isinstance(default, (int, float)):
        return abs(value - float(default)) > 1e-12
    return value != default
