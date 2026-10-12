"""Exercise the installed editor, including its optional viewer dependency."""
import asyncio
import json
import re
import selectors
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from draft.paths import repo_root


def test_editor_serves_page_and_state():
    pytest.importorskip("viser", reason="needs the editor extra")
    from websockets.asyncio.client import connect

    # viser reports get_port() == 0 when asked for port 0, so give it an unused
    # concrete port. The editor itself reports its actual ephemeral port.
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        viewer_port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/robot_tuner.py", "--port", "0",
         "--viser-port", str(viewer_port)],
        cwd=repo_root(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 30
            port = None
            while time.monotonic() < deadline:
                for key, _ in selector.select(timeout=0.2):
                    chunk = key.fileobj.read1(8192)
                    output.extend(chunk)
                    if not chunk:
                        pytest.fail(output.decode(errors="replace"))
                match = re.search(rb"\[draft\] editor\s+http://localhost:(\d+)", output)
                if match:
                    port = int(match[1])
                    break
            assert port is not None, output.decode(errors="replace")

        with urlopen(f"http://localhost:{port}/", timeout=5) as response:
            assert response.status == 200
            assert b'Draft' in response.read()

        async def state():
            async with connect(f"ws://localhost:{port}/") as socket:
                return json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

        first = asyncio.run(state())
        assert first["type"] == "state"
        assert first["unsupported"] is None
        assert {"quadruped", "humanoid"} <= set(first["robots"])
        assert first["motors"]["classes"]
        with urlopen(first["viser_url"], timeout=5) as response:
            assert response.status == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
