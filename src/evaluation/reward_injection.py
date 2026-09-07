"""
AST-based reward injection for ard-isaaclab-tasks.

Each task env in ``ard-isaaclab-tasks`` isolates its reward in a single
``compute_reward(self)`` method, documented as "the ARD edit target". ARD's LLM
proposes a replacement ``compute_reward``; this module splices it into the task
env file via the AST.

Design — direct replacement
---------------------------
``compute_reward`` is the *only* method replaced. Two other things in the env
stay fixed around it, which is what makes replacing the whole method safe:

- ``_get_rewards`` is a framework hook, not an edit target. It calls
  ``self.compute_reward()``, hands the result to ``log_reward_components`` (so
  every component the LLM named reaches TensorBoard as ``Episode/components_<name>``),
  and returns the total. Injection never touches it, so the component logging
  cannot be dropped by a candidate that simply forgets about it.
- The **fixed evaluation metric** (``fitness_function``) lives in each env's
  ``_get_dones``, computed from environment state and independent of the reward.
  So ARD replacing the reward can never alter the scoreboard — that guarantee
  holds at the task layer.

``compute_reward`` has likewise been **cleaned** of the load-bearing side effects
the old ``_get_rewards`` carried (intermediate-value refresh, goal re-sampling,
``prev_actions`` bookkeeping, …); those now live in their own hooks. With nothing
left in it but the reward computation itself, we simply **replace the whole
method** — no pristine body to preserve, no auxiliary indirection.

The two-output contract
-----------------------
Following Eureka, a proposal must return ``(total_reward, reward_components)``:
the per-env reward, plus a dict naming each term that went into it. The dict is
what makes the reward observable across iterations, so :func:`_parse_reward_method`
*enforces* it — a candidate returning a bare tensor would unpack wrong at every
simulation step and waste an entire training job before failing.
"""

import ast
import logging
import textwrap
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

REWARD_METHOD = config.REWARD_METHOD_NAME


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

    # The env calls ``self.compute_reward()`` with no extra args; enforce (self).
    if not func.args.args or func.args.args[0].arg != "self":
        raise RewardInjectionError(
            "Proposed reward method must take 'self' as its first parameter"
        )
    func.name = REWARD_METHOD
    func.decorator_list = []  # methods on the env are plain instance methods
    _check_returns_pair(func)
    return func


def _check_returns_pair(func: ast.FunctionDef) -> None:
    """Reject a proposal that does not return ``(total_reward, components)``.

    ``_get_rewards`` unpacks the result into two names on every simulation step, so
    a proposal returning a bare tensor raises deep inside the training container —
    after the image is built, the job dispatched, and (on HPC) a cluster slot spent.
    The check is cheap and static, so do it here instead.

    Only *syntactically* visible returns can be judged. A return whose value is a
    plain name (``return result``) or a call is accepted rather than guessed at; the
    check rejects the shapes that are unambiguously wrong — no return at all, or a
    return of a literal tuple whose length is not 2.
    """
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        raise RewardInjectionError("Proposed reward method has no 'return'")

    for node in returns:
        if isinstance(node.value, ast.Tuple) and len(node.value.elts) != 2:
            raise RewardInjectionError(
                f"Proposed reward returns a {len(node.value.elts)}-tuple at line "
                f"{node.lineno}; it must return exactly "
                "(total_reward, reward_components)"
            )

    # At least one return has to be a literal 2-tuple, otherwise the method never
    # demonstrably produces the pair the framework unpacks.
    if not any(isinstance(n.value, ast.Tuple) and len(n.value.elts) == 2 for n in returns):
        raise RewardInjectionError(
            "Proposed reward never returns a (total_reward, reward_components) "
            "pair; every return must be a 2-tuple of the total reward and the "
            "dict of its named components"
        )


def inject_reward(env_source: str, designed_src: str) -> str:
    """
    Splice the LLM-proposed reward into ``env_source``.

    Returns the full, modified module source. Only the ``compute_reward`` method
    region is rewritten; the rest of the file — ``_get_rewards``, the component
    logging it performs, and the ``fitness_function`` metric — is preserved
    verbatim. The proposed method replaces the original ``compute_reward`` outright.

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
