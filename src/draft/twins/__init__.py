"""Digital twins of four Unitree platforms (G1, H2, Go2, B2).

Each twin is generated from the original's kinematics and per-joint actuation
alone; every mass, envelope and inertia is derived by the trends, then scored
against the original. See `docs/twins.md`.

Modules: `targets` (registry), `measure` (shared measurement schema),
`synthesize` (measurement -> twin), `zero_align`, `reroot`, `render`,
`rl_adapter`, `dynamics_parity`.
"""
