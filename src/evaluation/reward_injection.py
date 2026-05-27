"""
AST-based reward injection for ard-isaaclab-tasks.

Each task env in ``ard-isaaclab-tasks`` isolates its reward in a single
``_get_rewards(self)`` method, documented as "the sole edit target for the ARD
framework". ARD's LLM proposes a replacement ``_get_rewards``; this module
splices it into the task env file via the AST.

Design — direct replacement
---------------------------
The **fixed evaluation metric** (``fitness_function``) no longer lives in
``_get_rewards`` at all: in ard-isaaclab-tasks it was moved into each env's
``_get_dones`` (via ``_log_fitness``), computed from environment state and
independent of the reward. So ARD replacing the reward can never alter the
scoreboard — that guarantee now holds at the task layer.

The task's ``_get_rewards`` has likewise been **cleaned** of the load-bearing
side effects it used to carry (intermediate-value refresh, goal re-sampling,
``prev_actions`` bookkeeping, …); those now live in their own hooks. With
nothing left in ``_get_rewards`` but the reward computation itself, we simply
**replace the whole method** with the LLM's proposed ``_get_rewards`` — no
pristine body to preserve, no auxiliary ``_ard_designed_reward`` indirection.
"""

import ast
import logging
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

REWARD_METHOD = "_get_rewards"


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


def _parse_reward_method(designed_src: str) -> ast.FunctionDef:
    """Parse the LLM-proposed reward and return its FunctionDef, cleaned."""
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

    # The env calls ``self._get_rewards()`` with no extra args; enforce (self).
    if not func.args.args or func.args.args[0].arg != "self":
        raise RewardInjectionError(
            "Proposed reward method must take 'self' as its first parameter"
        )
    func.name = REWARD_METHOD
    func.decorator_list = []  # methods on the env are plain instance methods
    # Must actually return something (the reward).
    if not any(isinstance(n, ast.Return) and n.value is not None
               for n in ast.walk(func)):
        raise RewardInjectionError("Proposed reward method has no 'return'")
    return func


def inject_reward(env_source: str, designed_src: str) -> str:
    """
    Splice the LLM-proposed reward into ``env_source``.

    Returns the full, modified module source. Only the ``_get_rewards`` method
    region is rewritten; the rest of the file is preserved verbatim. The
    proposed method replaces the original ``_get_rewards`` outright.

    Raises RewardInjectionError on any structural problem.
    """
    module = ast.parse(env_source)
    _, original = _find_env_class(module)
    if original is None:
        raise RewardInjectionError(
            f"No class defining {REWARD_METHOD!r} found in env source"
        )

    reward_method = _parse_reward_method(designed_src)

    # Render the method, indented to the original method's column.
    indent = " " * original.col_offset
    rendered = textwrap.indent(
        ast.unparse(ast.fix_missing_locations(reward_method)), indent
    )

    # Textually replace the original method's line span (preserves the rest).
    lines = env_source.splitlines()
    start = original.lineno - 1          # 0-based, inclusive
    end = original.end_lineno            # exclusive
    new_lines = lines[:start] + rendered.splitlines() + lines[end:]
    new_source = "\n".join(new_lines) + "\n"

    # Validate the result parses and the reward method is still present.
    try:
        check = ast.parse(new_source)
    except SyntaxError as e:
        raise RewardInjectionError(f"Injected source does not parse: {e}") from e
    _, chk_method = _find_env_class(check)
    if chk_method is None:
        raise RewardInjectionError(
            f"Injected source is missing {REWARD_METHOD!r} after splice"
        )
    return new_source
