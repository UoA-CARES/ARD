"""
AST-based reward injection for ard-isaaclab-tasks.

Each task env in ``ard-isaaclab-tasks`` isolates its reward in a single
``_get_rewards(self)`` method, documented as "the sole edit target for the ARD
framework". ARD's LLM proposes a replacement ``_get_rewards``; this module
splices it into the task env file via the AST.

Design — why we keep the pristine body
---------------------------------------
The **fixed evaluation metric** (``fitness_function``) no longer lives in
``_get_rewards`` at all: in ard-isaaclab-tasks it was moved into each env's
``_get_dones`` (via ``_log_fitness``), computed from environment state and
independent of the reward. So ARD replacing the reward can never alter the
scoreboard — that guarantee now holds at the task layer.

What ``_get_rewards`` still carries, in some tasks, are load-bearing **side
effects** the rest of the env depends on: franka calls
``self._compute_intermediate_values()`` (also feeding observations); inhand
re-samples the goal pose when reached and maintains ``consecutive_successes``;
forge updates ``self.prev_actions`` / ``self.success_pred_scale``.

Rather than disentangle those per task, we keep the **entire pristine body**,
demote its terminal ``return <expr>`` to a bare expression statement (so its
side effects still run), and append ``return self._ard_designed_reward()``. The
LLM's method becomes ``_ard_designed_reward(self)`` and supplies the reward that
is actually returned. This is uniform across all six tasks; preserving the body
keeps the task mechanics intact, and fitness is safe regardless because it is
logged from ``_get_dones`` before the reward runs.
"""

import ast
import logging
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

REWARD_METHOD = "_get_rewards"
DESIGNED_METHOD = "_ard_designed_reward"


class RewardInjectionError(ValueError):
    """Raised when the env file or the proposed reward can't be spliced."""


def _find_method(class_node: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_env_class(module: ast.Module):
    """Return (class_node, method_node) for the first class defining REWARD_METHOD."""
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef):
            method = _find_method(node, REWARD_METHOD)
            if method is not None:
                return node, method
    return None, None


def extract_method_source(env_source: str, method_name: str = REWARD_METHOD) -> str:
    """
    Return the verbatim source text of ``method_name`` (for use as an LLM template).

    Uses line ranges from the AST so comments and formatting are preserved.
    """
    module = ast.parse(env_source)
    _, method = _find_env_class(module)
    if method is None or method.name != method_name:
        # Fall back to an explicit search if the env class method differs.
        method = None
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                method = node
                break
    if method is None:
        raise RewardInjectionError(
            f"Could not find method {method_name!r} in the env source"
        )
    lines = env_source.splitlines()
    # Include any decorators in the slice.
    start = min([method.lineno] + [d.lineno for d in method.decorator_list]) - 1
    end = method.end_lineno
    return "\n".join(lines[start:end])


def _parse_designed_method(designed_src: str) -> ast.FunctionDef:
    """Parse the LLM-proposed reward and return its FunctionDef, renamed + cleaned."""
    designed_src = textwrap.dedent(designed_src).strip()
    try:
        snippet = ast.parse(designed_src)
    except SyntaxError as e:
        raise RewardInjectionError(f"Proposed reward is not valid Python: {e}") from e

    func = next(
        (n for n in snippet.body if isinstance(n, ast.FunctionDef)), None
    )
    if func is None:
        raise RewardInjectionError(
            "Proposed reward contains no function definition"
        )

    # Force the signature to (self) and a neutral name; the env passes no args.
    if not func.args.args or func.args.args[0].arg != "self":
        raise RewardInjectionError(
            "Proposed reward method must take 'self' as its first parameter"
        )
    func.name = DESIGNED_METHOD
    func.decorator_list = []  # methods on the env are plain instance methods
    # Must actually return something (the reward).
    if not any(isinstance(n, ast.Return) and n.value is not None
               for n in ast.walk(func)):
        raise RewardInjectionError("Proposed reward method has no 'return'")
    return func


def _build_eval_method(original: ast.FunctionDef) -> ast.FunctionDef:
    """
    Rebuild ``_get_rewards`` so it runs the pristine body for its side effects
    (fitness logging, intermediate refresh) but returns the designed reward.
    """
    new = ast.FunctionDef(
        name=original.name,
        args=original.args,
        body=[],
        decorator_list=[],
        returns=original.returns,
        type_comment=None,
    )

    body = list(original.body)
    # Preserve a leading docstring as-is.
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        new.body.append(body[0])
        body = body[1:]

    if body and isinstance(body[-1], ast.Return) and body[-1].value is not None:
        # Demote the terminal `return <expr>` to a side-effecting expression so
        # fitness logging that lives inside it (e.g. franka) still runs.
        head, last = body[:-1], body[-1]
        new.body.extend(head)
        new.body.append(ast.Expr(value=last.value))
    else:
        logger.warning(
            "Pristine _get_rewards has no terminal 'return'; keeping its body verbatim"
        )
        new.body.extend(body)

    # Return the LLM-designed reward.
    designed_call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=DESIGNED_METHOD,
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )
    new.body.append(ast.Return(value=designed_call))
    return new


def inject_reward(env_source: str, designed_src: str) -> str:
    """
    Splice the LLM-proposed reward into ``env_source``.

    Returns the full, modified module source. Only the ``_get_rewards`` method
    region is rewritten; the rest of the file is preserved verbatim.

    Raises RewardInjectionError on any structural problem.
    """
    module = ast.parse(env_source)
    class_node, original = _find_env_class(module)
    if original is None:
        raise RewardInjectionError(
            f"No class defining {REWARD_METHOD!r} found in env source"
        )

    designed = _parse_designed_method(designed_src)
    eval_method = _build_eval_method(original)

    # Render the two methods, indented to the original method's column.
    indent = " " * original.col_offset
    rendered = "\n\n".join(
        textwrap.indent(ast.unparse(ast.fix_missing_locations(m)), indent)
        for m in (eval_method, designed)
    )

    # Textually replace the original method's line span (preserves the rest).
    lines = env_source.splitlines()
    start = original.lineno - 1          # 0-based, inclusive
    end = original.end_lineno            # exclusive
    new_lines = lines[:start] + rendered.splitlines() + lines[end:]
    new_source = "\n".join(new_lines) + "\n"

    # Validate the result parses and both methods are present.
    try:
        check = ast.parse(new_source)
    except SyntaxError as e:
        raise RewardInjectionError(f"Injected source does not parse: {e}") from e
    chk_class, chk_method = _find_env_class(check)
    if chk_method is None or _find_method(chk_class, DESIGNED_METHOD) is None:
        raise RewardInjectionError(
            "Injected source is missing the expected methods after splice"
        )
    return new_source
