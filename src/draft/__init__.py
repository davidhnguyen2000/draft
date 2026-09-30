"""Draft — a parametric design tool for robot exploration.

A robot is a table of numeric parameters plus a kinematic tree whose fields are
expressions over it, compiled to MJCF. Unstated values come from trends fitted to
114 catalogued actuators and 49 published robots, and the result is checked
against what that hardware can do.

    draft.generation   the compiler: table + tree -> MJCF
    draft.trends       the fitted trends, and the feasibility checks they imply
    draft.robots       the shipped specifications (humanoid, quadruped)
    draft.twins        rebuilding four off-the-shelf platforms, end to end
    draft.tasks        the locomotion environments a generated design trains in
    draft.editor       the browser editor -- the normal way to build a design
"""

__version__ = "0.1.0"


def _load_env_libstdcxx() -> None:
    """In a conda environment, load its libstdc++ before a pip wheel loads the system's.

    conda-forge's Python links an ICU (through sqlite3) built against a newer
    libstdc++ than older Linux distributions ship. Pip wheels such as torch
    resolve ``libstdc++.so.6`` from the system instead, and a process keeps
    whichever copy loads first, so `import torch` followed by anything that
    imports sqlite3 (IPython, which mjlab's viewer pulls in) stops with
    "CXXABI_1.3.15 not found". The environment's copy is the newer and is
    backward compatible, so loading it first satisfies both.
    """
    import os
    import sys
    lib = os.path.join(sys.prefix, "lib", "libstdc++.so.6")   # absent in a venv
    if sys.platform == "linux" and os.path.exists(lib):
        import ctypes
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_load_env_libstdcxx()
