#!/usr/bin/env python3
"""``scripts/robot_tuner.py`` — the browser editor.

Serves one page (parameters beside an embedded viser 3D scene); every derived
number is computed server-side by the same code the generator runs.

    python scripts/robot_tuner.py                      # page :8081, viser :8080
    python scripts/robot_tuner.py --port 9000 --viser-port 9001
"""

from __future__ import annotations

import argparse
import asyncio

from draft.paths import repo_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draft robot editor — one page, parameters beside the robot")
    ap.add_argument("--port", type=int, default=8081, help="editor page port")
    ap.add_argument("--viser-port", type=int, default=8080, help="port for the viser scene the page embeds")
    args = ap.parse_args()

    from draft.editor.viewer import Viewer3D
    from draft.editor.web.server import WebEditor

    viewer = Viewer3D(port=args.viser_port, verbose=False)
    viewer.start()

    editor = WebEditor(viewer, repo_root())
    # Best-effort: show the selected robot's last generated model, if any.
    editor._show_selected()

    # serve() prints the URL, since it may move to the next free port.
    try:
        asyncio.run(editor.serve(args.port))
    except KeyboardInterrupt:
        print("\n[draft] Shutting down.")

