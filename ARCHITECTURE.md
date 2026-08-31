# ARD Stage 2 — Architecture

ARD's reward-refinement loop is built around one external repo and a pluggable
execution backend (`runner.backend` in `configs/settings.yaml`):

- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)** — the IsaacLab task substrate.
  Six tasks registered as `Isaac-ARD-*`, each isolating its reward in a single
  `_get_rewards` method (the sole ARD edit target) and logging a fixed
  `fitness_function` evaluation metric.
- **`LocalRunner`** (`backend: local`) — ARD builds each candidate's `Dockerfile`
  and `docker run`s the training image on this machine, one candidate at a time,
  then reads the `logs/` it wrote to its own work dir. A thin single-machine driver.
- **`HPCRunner`** (`backend: hpc`) — for the CARES HPC Scheduler. ARD builds +
  pushes each candidate's image to the CARES registry, submits the whole batch,
  and the jobs train **concurrently** on the cluster. As each finishes, its
  artifacts are recycled from the NAS mount into `./runs` and read by the same
  `ResultProcessor`. The evaluator, scorer, workspace staging, and result capture
  are backend-agnostic — only dispatch differs (blocking `run` vs
  `submit`/`poll`/`collect`).

## HPC backend — reward delivery and the submit/monitor split

The CARES scheduler *pulls a prebuilt image* and does not build from a working
tree, so each candidate's injected `_get_rewards` is baked into **its own image
tag**: ARD reuses the exact `.tar.gz` `WorkspaceManager` builds for local as the
`docker build` context, tags it `<registry>/<image_repo>:<candidate-tag>`, and
pushes it (an incremental push — only the `ard_tasks` + editable-install layers
change). Two other scheduler quirks shape the code:

- **Config rides the `command`, not `env`.** The scheduler drops the job `env`
  block, so `task`/`seed`/tunables are packed into `hpc_entrypoint.sh` flags;
  `runner.env`'s `MAX_ITERATIONS`/`NUM_ENVS` become `--max_iterations`/`--num_envs`.
- **Outputs come back via the NAS.** Only `/workspace/output` is preserved, to
  `/cares-nas/hpc/outputs/<upi>/<job_id>` (mounted locally at `runner.hpc.nas_outputs`).
  The evaluator submits every candidate, polls each `job_id` until a terminal
  status, then `collect`s `<nas_outputs>/<job_id>/` into
  `./runs/<task>/<timestamp>/<tag>/`.

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
   dispatch (backend):
     local │  LocalRunner.run   ──►  docker build + docker run each candidate, one at a time (env={TASK,SEED})
     hpc   │  HPCRunner.submit  ──►  docker build + push per-candidate image, submit all → cluster runs them concurrently
           │  HPCRunner.poll / collect  ──►  await each job, recycle NAS/<job_id> → <output_dir>/<tag>/
        │
ResultProcessor.capture  ──►  read <output_dir>/<tag>/logs + scalar summary
        │
FitnessScorer.score_all / select_best  ──►  read fitness_function, pick the batch winner
        │
EurekaAgent.receive_feedback  ──►  fold that same run's summary back in (code and numbers from one run)
```

## Eval phase & warm-starting — once per iteration

Each iteration's run-phase winner (`FitnessScorer.select_best` over that
iteration's `sample` candidates) is re-trained `num_eval` times, on different
seeds, before anything is committed to feedback or warm-starting — a single
run's fitness is seed-noisy, so this de-noises it. `select_best` over those
eval records picks the iteration's actual winner (`best_eval`, falling back to
the run-phase `best` if every eval retrain failed to score); each seed stays in
the history as its own record. Cost per task is `iteration * (sample + num_eval)`
trainings.

That winner's checkpoint (`RewardRecord.checkpoint_path`, set by
`RewardEvaluator` after each successful run — see `find_checkpoint` in
`result_processor.py`) is then carried into the *next* iteration as
`warm_start_checkpoint`, when `warm_start` is enabled: every candidate in the
next iteration's run and eval phases resumes training from it instead of
random weights (baked into that candidate's build tarball, delivered via
`--checkpoint`; see `evaluator.py`'s `_build_env`/`_build_hpc_command` and
`_effective_max_iterations`, which extends the configured epoch budget by the
checkpoint's own inherited epoch count). Iteration 1 always cold-starts, since
no previous winner exists yet; `warm_start_checkpoint` also only lives for one
continuous `--refine` invocation, not across separate runs.

## Module map (`src/`)

- `evaluation/local_runner.py` — builds + `docker run`s each candidate locally (one blocking `run`: build → run → result).
- `evaluation/hpc_runner.py` — `HPCRunner`: builds + pushes each candidate's image and drives the CARES scheduler (`submit`/`poll`/`collect`).
- `evaluation/reward_injection.py` — AST splice of `_get_rewards` (+ fitness preservation).
- `evaluation/workspace_manager.py` — builds per-candidate job codebases.
- `evaluation/result_processor.py` — reads the job's logs in place, writes the scalar summary.
- `evaluation/scorer.py` — `FitnessScorer`: reads `fitness_function`, ranks candidates, summarises eval seeds.
- `evaluation/evaluator.py` — `RewardEvaluator`, the dispatch + capture orchestrator.
- `refinement/llm_agent.py` — `EurekaAgent` (proposes `_get_rewards`, folds in feedback).
- `refinement/agent_config/*.txt` — LLM prompt templates.

## Configuration

- `configs/settings.yaml` — `tasks_repo`, `output_dir`, and the `runner` block.
  `runner.backend` picks `local` (`use_gpu`, `timeout_seconds`, `image`, optional
  `env`/`build_args`/`command_template`; Dockerfile built locally, no prebuilt tag)
  or `hpc` (a `runner.hpc` sub-block: `registry`, `upi` (or `$ARD_UPI`), `nas_outputs`,
  `max_runtime_hours`, `poll_seconds`, `datasets`, …).
- `configs/taskconfig.yaml` — `task`, `env_file` (the injection target), `description`, `max_iterations`.
- `configs/refineconfig.yaml` — `iteration`, `num_eval`, `base_seed`, and the `agent` (LLM) block.

The only secret is `OPENROUTER_API_KEY` (LLM). Each job's training image is built
locally from the staged codebase's `Dockerfile` — nothing is prebuilt.
