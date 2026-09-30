"""Training commands and disturbances are the same for every design.

Designs are scored on one absolute battery (``draft.lineup.evaluate``), so the
velocity and push tasks must not size commands off the robot, else a design's
top speed just reflects its training ceiling. Checked on the AST (no mjlab).
"""

import ast
from pathlib import Path

import pytest

from draft.paths import repo_root

TASKS = repo_root() / "src" / "draft" / "tasks" / "quadruped"

#: ``BaseCtx`` attributes that carry a per-design number.
PER_DESIGN = ("v_max", "push_base_range", "s_sqrt", "s", "leg_length",
              "omega_nl", "init_height", "foot_radius", "command_scale")

#: task -> (allowed per-design reads, reason). Anything else fails.
ALLOWED = {
    "velocity": (
        frozenset({"v_max"}),
        "reward SHAPING (tracking std, forward-speed bonus scale) is sized off the "
        "design's own actuator estimate, because a tolerance sized off a 15 m/s "
        "ceiling nothing reaches is not a tolerance — an earlier sweep pointed all four uses "
        "at the shared ceiling and cost 56/51/22% of top speed. The COMMAND range is "
        "what has to stay common, and "
        "test_command_range_is_never_sized_off_the_design pins that separately",
    ),
    "push": (
        frozenset({"s_sqrt"}),
        "push INTERVAL still scales with sqrt(leg length): a fixed cadence gives "
        "the giraffe ~1.8x the disturbances per gait cycle. It is shared with the "
        "base policy this task warm-starts from, so changing it means retraining "
        "the base — a known, documented asymmetry, not an oversight",
    ),
}


def _literal(path: Path, name: str):
    """Value of a module-level literal constant, read from the AST (no mjlab)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} no longer defines {name}")


def _code_only(node: ast.AST) -> str:
    """Unparse a node with every docstring removed."""
    clone = ast.parse(ast.unparse(node))
    for n in ast.walk(clone):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
           and isinstance(body[0].value, ast.Constant) \
           and isinstance(body[0].value.value, str):
            body.pop(0)
    return ast.unparse(clone)


def _ctx_attrs(path: Path) -> set[str]:
    """Every ``ctx.<attr>`` read in the module (AST, so docstrings are ignored)."""
    tree = ast.parse(path.read_text())
    return {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "ctx"
    }


@pytest.mark.parametrize("task", sorted(ALLOWED))
def test_axis_task_is_not_scaled_by_the_design(task):
    allowed, why = ALLOWED[task]
    found = _ctx_attrs(TASKS / task / "__init__.py") & set(PER_DESIGN)
    assert found <= allowed, (
        f"{task} task sizes itself off the robot via ctx.{sorted(found - allowed)}. "
        f"The absolute battery then measures the curriculum, not the design. "
        f"Only {sorted(allowed) or 'nothing'} may remain here — {why}."
    )


def _shaping_names(path: Path) -> set[str]:
    """Names bound to a per-design ``ctx.<attr>`` (e.g. ``v_shape = ctx.v_max``)."""
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) \
           and isinstance(node.value.value, ast.Name) and node.value.value.id == "ctx" \
           and node.value.attr in PER_DESIGN:
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return out


def test_command_range_is_never_sized_off_the_design():
    """Per-design numbers may shape the reward but not the command range.

    Checks the twist ranges and `velocity_stages` read only the shared ceiling.
    """
    path = TASKS / "velocity" / "__init__.py"
    tree = ast.parse(path.read_text())
    shaping = _shaping_names(path)

    command_srcs = [
        src for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for src in [ast.unparse(node)]
        if "ranges.lin_vel_x" in src or "velocity_stages" in src
    ]
    assert command_srcs, (
        "no statement in the velocity task sets a twist range or velocity_stages, "
        "so this test is no longer looking at the command surface"
    )
    for src in command_srcs:
        leaked = sorted(n for n in shaping if n in
                        {x.id for x in ast.walk(ast.parse(src))
                         if isinstance(x, ast.Name)})
        assert not leaked, (
            f"the command range is sized off the design via {leaked}:\n  {src}\n"
            "An earlier sweep made exactly this mistake: a design trained to its own ceiling "
            "reports that ceiling as its top speed."
        )
        assert "COMMON_V_CEILING" in src or "v_max" in src, (
            f"the command range no longer reads the shared ceiling:\n  {src}"
        )


def test_velocity_ceiling_clears_every_measured_top_speed():
    """The shared ceiling is at least 2x the fastest speed any design has reached."""
    ceiling = _literal(TASKS / "speed_limits.py", "COMMON_V_CEILING")

    fastest_measured = 5.59
    assert ceiling >= 2.0 * fastest_measured, (
        f"COMMON_V_CEILING = {ceiling} is too close to the "
        f"{fastest_measured} m/s the bear already reached. A design that trains to "
        f"a ceiling reports that ceiling as its top speed — twice now."
    )


def test_push_schedule_is_a_schedule_and_not_feedback():
    """The push task uses the fixed `PushScheduleCurriculum`, not the fall-rate
    adaptive one (which would end each design at a different kick)."""
    path = TASKS / "push" / "__init__.py"
    # Names used in code (AST), not in docstrings.
    tree = ast.parse(path.read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        a.name for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert "PushDifficultyCurriculum" not in used, (
        "the push axis is back on the adaptive curriculum, which holds every "
        "design at its own fall rate and so ends each one at a different "
        "absolute kick — see PushScheduleCurriculum for what that cost."
    )
    assert "PushScheduleCurriculum" in used, (
        "the push axis no longer installs the fixed schedule"
    )
    stages = _literal(TASKS / "push" / "__init__.py", "_PUSH_STAGES")
    steps = [it for it, _ in stages]
    scales = [sc for _, sc in stages]
    assert steps[0] == 0, "the ramp needs a stage at iteration 0"
    assert steps == sorted(steps), "push stages must be ordered by iteration"
    assert scales == sorted(scales), "a ramp cannot step backwards"


def test_warm_start_zeroes_the_step_counter_and_resume_does_not():
    """In train.py, warm start zeroes ``common_step_counter`` (a new task's
    curriculum starts at zero); --resume keeps it."""
    tree = ast.parse((repo_root() / "scripts" / "train.py").read_text())

    branch = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.If)
         and "args.resume" in ast.unparse(n.test)),
        None,
    )
    assert branch is not None, "train.py no longer branches on args.resume"
    assert not _zeroes_counter_in(branch.body), (
        "the --resume path zeroes common_step_counter. A preempted run continuing "
        "itself must keep it, or every requeue restarts the curriculum ramp."
    )
    assert _zeroes_counter_in(branch.orelse), (
        "the warm-start path does not zero common_step_counter, so a warm-started "
        "run begins past the end of its own curriculum — see the docstring."
    )


def _zeroes_counter_in(body) -> bool:
    """True if any statement in ``body`` assigns 0 to a common_step_counter."""
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Assign) and ast.unparse(n).replace(" ", "").endswith(
                "common_step_counter=0"
            ):
                return True
    return False
