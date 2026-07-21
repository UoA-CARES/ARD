# ARD Stage 2 — Architecture

ARD's reward-refinement loop is built around one external repo and a local runner:

- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)** — the IsaacLab task substrate.
  Six tasks registered as `Isaac-ARD-*`, each isolating its reward in a single
  `_get_rewards` method (the sole ARD edit target) and logging a fixed
  `fitness_function` evaluation metric.
- **`LocalRunner`** — ARD builds each candidate's `Dockerfile` and `docker run`s
  the training image on this machine, one candidate at a time, then reads the
  `logs/` it wrote to its own work dir. There is no remote scheduler and no job
  queue; ARD is a thin single-machine driver.

## What changed from the old pipeline

| Concern | Old | New |
|---|---|---|
| Distribution | `ParallelExecutor` SSH'd into `machines_pool.txt` and ran `docker/run_remote_pipeline.sh` per task | Build + `docker run` each candidate locally (`LocalRunner`), one candidate at a time |
| Reward injection | git-checkout an in-tree project + **regex** replace of a `@torch.jit.script` reward fn | Copy the tasks repo + **AST** rewrite of `_get_rewards` (`reward_injection.py`) |
| Eval metric | `Episode/consecutive_successes` | `fitness_function` (logged by every task; matched by tag suffix) |
| Result source | local TensorBoard path on the training host | per-job **work dir** read in place (`<tag>/logs/…/summaries/`) |
| LLM target | a standalone reward fn returning `(total_reward, components)` | a whole `_get_rewards(self)` method returning the reward |

## Fitness isolation (task layer)

The fixed evaluation metric (`fitness_function`) is **isolated in the task repo**,
out of `_get_rewards`. Each `Isaac-ARD-*` env computes it in a `_log_fitness()`
method called from `_get_dones` (a per-step method ARD never edits), from pure
environment state. So ARD rewriting `_get_rewards` cannot alter or drop the
scoreboard — that guarantee holds at the task layer, not just by convention.

## Reward injection — direct method replacement

Two properties of the `ard-isaaclab-tasks` env layer let ARD swap rewards safely:

- The fixed **evaluation metric** (`fitness_function`) no longer lives in
  `_get_rewards`. Each env computes it in `_get_dones` (via `_log_fitness`), from
  environment state and independent of the reward — so rewriting the reward can
  never alter the scoreboard.
- `_get_rewards` has been **cleaned** of the load-bearing side effects it used to
  carry (intermediate-value refresh, goal re-sampling, `prev_actions`
  bookkeeping); those now live in their own hooks.

With nothing left in `_get_rewards` but the reward computation itself,
`reward_injection.inject_reward` simply **replaces the whole method** with the
LLM's proposed `_get_rewards`, keeping the rest of the env file verbatim — no
pristine body to preserve, no `_ard_designed_reward` indirection.

## Flow (one refinement iteration)

```
EurekaAgent.func_gen  ──►  N candidate _get_rewards methods
        │
WorkspaceManager.build_codebase  ──►  per-candidate ard-isaaclab-tasks .tar.gz (reward injected)
        │
LocalRunner.run  ──►  docker build + docker run each candidate, one at a time (env={TASK,SEED})
        │
ResultProcessor.capture  ──►  read <output_dir>/<tag>/logs + scalar summary
        │
FitnessScorer.score_all / select_best  ──►  read fitness_function, pick the batch winner
        │
EurekaAgent.receive_feedback  ──►  fold the winner's summary back in (run phase, then eval phase)
```

## Module map (`src/`)

- `evaluation/local_runner.py` — builds + `docker run`s each candidate locally (one blocking `run`: build → run → result).
- `evaluation/reward_injection.py` — AST splice of `_get_rewards` (+ fitness preservation).
- `evaluation/workspace_manager.py` — builds per-candidate job codebases.
- `evaluation/result_processor.py` — reads the job's logs in place, writes the scalar summary.
- `evaluation/scorer.py` — `FitnessScorer`: reads `fitness_function`, ranks candidates.
- `evaluation/evaluator.py` — `RewardEvaluator`, the dispatch + capture orchestrator.
- `refinement/llm_agent.py` — `EurekaAgent` (proposes `_get_rewards`, folds in feedback).
- `refinement/agent_config/*.txt` — LLM prompt templates.

## Configuration

- `configs/settings.yaml` — `tasks_repo`, `output_dir`, and the `runner` block
  (`use_gpu`, `timeout_seconds`, `image`, optional
  `env`/`build_args`/`command_template`). Each job's Dockerfile is built locally —
  there is no prebuilt image tag.
- `configs/taskconfig.yaml` — `task`, `env_file` (the injection target), `description`, `max_iterations`.
- `configs/refineconfig.yaml` — `iteration`, `num_eval`, `base_seed`, and the `agent` (LLM) block.

The only secret is `OPENROUTER_API_KEY` (LLM). Each job's training image is built
locally from the staged codebase's `Dockerfile` — nothing is prebuilt.
