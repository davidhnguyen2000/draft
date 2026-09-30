"""No task may silently drop an `env_kwarg` its run config sets.

Each task's `make_env_cfg` names only what it reads and forwards the rest with
`**` to `make_base_env_cfg`, which has no `**kwargs`, so an unknown key raises.
AST-only, so it runs without mjlab.
"""

import ast
from pathlib import Path

import pytest

from draft.paths import repo_root

TASKS_DIR = repo_root() / "src" / "draft" / "tasks" / "quadruped"

#: (task name, module) for every registered quadruped task.
TASK_MODULES = {
    "quadruped": TASKS_DIR / "env.py",
    "quadruped_push": TASKS_DIR / "push" / "__init__.py",
    "quadruped_terrain": TASKS_DIR / "terrain" / "__init__.py",
    "quadruped_velocity": TASKS_DIR / "velocity" / "__init__.py",
}


def _func(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{path.name} defines no {name}()")


def _named_params(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)}


@pytest.fixture(scope="module")
def base_params() -> set[str]:
    return _named_params(_func(TASKS_DIR / "base_env.py", "make_base_env_cfg"))


def test_the_base_env_refuses_a_parameter_nobody_accepts(base_params):
    """The base has no `**kwargs`, so a misspelled key raises."""
    fn = _func(TASKS_DIR / "base_env.py", "make_base_env_cfg")
    assert fn.args.kwarg is None, (
        "make_base_env_cfg grew a **kwargs. It is the last thing between a "
        "misspelled env_kwarg and silence — every task forwards into it.")
    assert "terrain_scan_fix" in base_params and "terrain_mix" in base_params


@pytest.mark.parametrize("task", sorted(TASK_MODULES))
def test_every_task_forwards_what_it_does_not_read(task, base_params):
    """A task must end in `**something` AND pass that something on."""
    fn = _func(TASK_MODULES[task], "make_env_cfg")

    assert fn.args.kwarg is not None, (
        f"{task}: make_env_cfg has no ** catch-all, so any base parameter it "
        f"does not restate is a TypeError for the caller rather than reaching "
        f"the base.")
    catch_all = fn.args.kwarg.arg
    assert not catch_all.startswith("_"), (
        f"{task}: the catch-all is named {catch_all!r}. A leading underscore "
        f"says 'discard me', which is exactly the bug this guards.")

    call = next(
        (n for n in ast.walk(fn)
         if isinstance(n, ast.Call)
         and getattr(n.func, "id", None) == "make_base_env_cfg"), None)
    assert call is not None, f"{task}: never calls make_base_env_cfg"

    starred = [k.value.id for k in call.keywords
               if k.arg is None and isinstance(k.value, ast.Name)]
    assert catch_all in starred, (
        f"{task}: collects **{catch_all} but does not forward it to "
        f"make_base_env_cfg, so everything it did not name is dropped.")


@pytest.mark.parametrize("task", sorted(TASK_MODULES))
def test_a_task_only_names_parameters_the_base_has(task, base_params):
    """Every parameter a task names must exist on the base."""
    fn = _func(TASK_MODULES[task], "make_env_cfg")
    extra = _named_params(fn) - base_params
    assert not extra, (
        f"{task}: names {sorted(extra)}, which make_base_env_cfg does not have")


@pytest.mark.parametrize("task", sorted(TASK_MODULES))
def test_a_task_that_names_a_parameter_passes_it_on(task, base_params):
    """A named parameter is not in `**`, so it must be passed explicitly."""
    fn = _func(TASK_MODULES[task], "make_env_cfg")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "make_base_env_cfg")
    forwarded = {k.arg for k in call.keywords if k.arg}
    dropped = (_named_params(fn) & base_params) - forwarded
    assert not dropped, (
        f"{task}: names {sorted(dropped)} in its signature but never passes "
        f"{'it' if len(dropped) == 1 else 'them'} to make_base_env_cfg, so the "
        f"base uses its default while the config says otherwise.")
