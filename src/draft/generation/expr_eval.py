"""Evaluates ${expr} parameter expressions used in tree.yaml files."""
import math
import re
from typing import Any

from .mounting import (edge_outward_normal, edge_point, mount_euler, mount_pos,
                       rect_outline)

# Names an expression may use: arithmetic plus root-outline placement helpers.
_SAFE = {'pi': math.pi, 'abs': abs, 'min': min, 'max': max, 'sqrt': math.sqrt,
         'len': len, 'round': round,
         'mount_pos': mount_pos, 'mount_euler': mount_euler,
         'rect_outline': rect_outline,
         'edge_point': edge_point, 'edge_outward_normal': edge_outward_normal}


class _NS(dict):
    """Dict that also supports attribute access so ${motor_classes.L.r} works."""
    def __getattr__(self, name: str):
        try:
            v = self[name]
        except KeyError:
            raise AttributeError(name) from None
        return _NS(v) if isinstance(v, dict) else v


def attrdict(value: Any) -> Any:
    """Wrap a dict so ``${item.field}`` resolves (used for `for_each` loop items)."""
    return _NS(value) if isinstance(value, dict) else value


def resolve(value: Any, params: dict) -> Any:
    """Resolve ${...} expressions in a value; lists and scalars pass through."""
    if isinstance(value, str):
        value = value.strip()
        m = re.fullmatch(r'\$\{(.+)\}', value)
        if m:
            return _eval(m.group(1), params)
        # Inline substitution within a larger string
        return re.sub(r'\$\{([^}]+)\}', lambda m: _fmt(_eval(m.group(1), params)), value)
    if isinstance(value, list):
        return [resolve(v, params) for v in value]
    return value


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _eval(expr: str, params: dict) -> Any:
    # Leading whitespace would be an indentation error in eval.
    expr = expr.strip()
    ns = _NS({**_SAFE, **params})
    try:
        return eval(compile(expr, '<tree.yaml>', 'eval'), {'__builtins__': {}}, ns)
    except Exception as exc:
        raise ValueError(f"Cannot evaluate '${{{{ {expr} }}}}': {exc}") from exc
