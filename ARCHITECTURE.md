# ARD Stage 2 — Architecture

ARD's reward-refinement loop is built around one external repo and a pluggable
execution backend (`runner.backend` in `configs/settings.yaml`):

- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)** — the IsaacLab task substrate.
  Three tasks registered as `Isaac-ARD-*`, each isolating its reward in a single
  `compute_reward` method (the sole ARD edit target) that returns
  `(total_reward, reward_components)`, logging every component plus a fixed
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
tree, so each candidate's injected `compute_reward` is baked into **its own image
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
| Reward injection | git-checkout an in-tree project + **regex** replace of a `@torch.jit.script` reward fn | Copy the tasks repo + **AST** rewrite of `compute_reward` (`reward_injection.py`) |
| Eval metric | `Episode/consecutive_successes` | `fitness_function` (logged by every task; matched on the tag's final path segment) |
| Result source | local TensorBoard path on the training host | per-job **work dir** read in place (`<tag>/logs/…/summaries/`) |
| LLM target | a standalone `@torch.jit.script` fn returning `(total_reward, components)` | a `compute_reward(self)` method on the env class, returning the same pair |

## Reward component exposure (task layer)

This is Eureka's central mechanism, and it lives in the task repo. `compute_reward`
returns **two** things: the total reward, and a dict naming each individual term
that went into it. Each task's `_get_rewards` — a fixed framework hook ARD never
edits — calls it and passes the result to `log_reward_components`
(`ard_tasks/utils/reward_logging.py`), which reduces every component to its mean
over envs and writes it to `self.extras["log"]` as `components_<name>`, plus the aggregate
as `components_total`. IsaacLab's rl_games wrapper renames `log` -> `episode`, and
rl_games' `IsaacAlgoObserver` writes each key to TensorBoard as `Episode/components_<name>`.

ARD reads those scalars back out in `ResultProcessor.summarise_tensorboard` and
shows each one's statistics and trend over training to the LLM as feedback. That summary is a
*selection*, not a dump of the event file: rl_games writes ~30 scalars per run, and
pasting the optimiser internals (`a_loss`, `kl`, `e_clip`, the `performance/*` group,
the `/step` and `/time` duplicates) into every message would spend context on noise
and bury the components. Kept are the whole `Episode/` scope — everything the env
logs through `extras["log"]`, so any task-specific metric comes along for free — plus
four rl_games tags: `episode_lengths/iter`, `rewards/iter`, `losses/entropy`,
`info/last_lr` (`config.SUMMARY_TAG_ALLOWLIST`). That closes the loop the feedback prompt
depends on: "if a component's values are near identical throughout, RL cannot
optimise it — rescale, rewrite, or discard it" only means something when the LLM can
actually see each component it wrote.

Three properties of that pipeline are load-bearing and easy to break:

- `self.extras` is never cleared by `DirectRLEnv`, and the rl_games wrapper pops
  `"log"` out of a *copy* of it. So the env's own `extras["log"]` dict survives every
  step; left alone, one dict object is mutated in place and appended to the
  observer's `ep_infos` once per step, and the epoch's TensorBoard value collapses to
  the last step's reading. Every task therefore calls `reset_episode_log` from
  `_get_dones` (which `DirectRLEnv.step` runs before `_get_rewards`).
- `IsaacAlgoObserver.after_print_stats` takes its key list from `ep_infos[0]` and
  indexes every later reading with it, so a component key that appears on some steps
  and not others raises `KeyError` mid-training. `log_reward_components` pins the key
  set on first use and reconciles later steps against it.
- Components share one flat `Episode/` namespace with the evaluation metric, so they
  are prefixed `components_`, and ARD's `FitnessScorer._resolve_tag` matches
  `fitness_function` on a whole path segment rather than by `endswith`. Together those
  two make it impossible for an LLM-named component to shadow the scoreboard.

## Fitness isolation (task layer)

The fixed evaluation metric (`fitness_function`) is **isolated in the task repo**,
out of `compute_reward`. Each `Isaac-ARD-*` env computes it in `_get_dones` (a
per-step method ARD never edits), from pure environment state. So ARD rewriting
`compute_reward` cannot alter or drop the scoreboard — that guarantee holds at the
task layer, not just by convention.

## Reward injection — direct method replacement

Two properties of the `ard-isaaclab-tasks` env layer let ARD swap rewards safely:

- The fixed **evaluation metric** (`fitness_function`) no longer lives in
  `compute_reward`. Each env computes it in `_get_dones`, from environment state
  and independent of the reward — so rewriting the reward can never alter the
  scoreboard.
- `compute_reward` has been **cleaned** of the load-bearing side effects the old
  `_get_rewards` carried (intermediate-value refresh, goal re-sampling,
  `prev_actions` bookkeeping); those now live in their own hooks.

With nothing left in `compute_reward` but the reward computation itself,
`reward_injection.inject_reward` simply **replaces the whole method** with the
LLM's proposal, keeping the rest of the env file verbatim — including
`_get_rewards` and the component logging it performs, which a candidate therefore
cannot drop. `_parse_reward_method` statically rejects a proposal that does not
return a `(total_reward, reward_components)` pair, so a wrong-arity return fails
here rather than after an image build and a spent GPU slot.

## Flow (one refinement iteration)

```
EurekaAgent.func_gen  ──►  N candidate compute_reward methods
                              (each returns total_reward + component dict)
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
`warm_start_checkpoint`, when `warm_start.enabled` is set: every candidate in the
next iteration's run and eval phases starts training from it instead of
random weights (baked into that candidate's build tarball, delivered via
`--checkpoint`; see `evaluator.py`'s `_build_env`/`_build_hpc_command` and
`_effective_max_iterations`, which extends the configured epoch budget by the
checkpoint's own inherited epoch count). Iteration 1 always cold-starts, since
no previous winner exists yet; `warm_start_checkpoint` also only lives for one
continuous `--refine` invocation, not across separate runs.

### Transfer, not resume

The reward function differs between the checkpoint and the run that loads it, so
warm start is a *transfer*. It travels a path in rl_games kept deliberately
parallel to — and separate from — checkpoint resume:

```
refineconfig.yaml  warm_start: {...}
  → evaluator.py   _warm_start_flags()      → --warm_start --warm_start_reset_* ...
  → train.py       agent_cfg[params][config][warm_start]
  → torch_runner   _restore()               → agent.load_warmstart(ckpt)
  → a2c_common     set_warmstart_weights()
```

Plain `--checkpoint` without `--warm_start` still reaches `agent.restore()` →
`set_full_state_weights()`, unchanged, so resuming an interrupted run keeps
restoring the complete state exactly as it always did.

What the transfer applies:

- **Always:** the network weights (with the normalizer buffers they carry) and
  the epoch/frame counters — `_effective_max_iterations` is sized against the
  inherited epoch count, so it stays correct.
- **Never:** `last_mean_rewards` and `env_state`. The best-ever score gates the
  "best" checkpoint save, and one earned under the previous reward would suppress
  every save of the new run — leaving the *following* iteration nothing to warm
  start from. The environment is freshly built with a different reward.
- **Configured:** the optimizer state (and the AMP loss scale with it), the lr
  schedule (`last_lr`/`entropy_coef`), the observation normalizer, and the value
  normalizer. Defaults reset everything except the observation statistics, which
  are still valid because the environment did not change.

The normalizers are worth a note: they are `RunningMeanStd` **buffers of the
model**, so they ride inside the checkpoint's `model` blob rather than as keys of
their own. Resetting one is an in-place overwrite after the load, not an omitted
key — see `_reset_running_mean_std` in `a2c_common.py`.

### Critic warmup

`warm_start.critic_warmup_epoch_count` makes the first N epochs after a transfer
critic-only. The problem it solves: the transferred value function was fitted to
the *old* reward, so the very first PPO updates compute advantages from a stale
baseline and push the policy — the one thing worth transferring — in an arbitrary
direction. The warmup lets the critic re-fit before the actor is allowed to move.

What "the actor is held still" means depends on the network shape, and the ARD
tasks are split across both cases:

| `network.separate` | Frozen | Trains |
| --- | --- | --- |
| `true` (shadow_hand) | actor trunk + actor head | critic trunk + value head |
| `false` (cartpole, shadow_hand_vision) | the **shared trunk** + actor head | the value head alone |

With a shared trunk there is no critic-side trunk to train, so letting the trunk
move would move the policy — freezing it is what makes the window actually
critic-only. But it also means the warmup can only refit a linear readout on a
frozen representation, which is most of why shadow_hand runs separate trunks: there
the critic can actually re-learn a representation for the new reward.
`_critic_warmup_frozen_parameters` in `a2c_common.py` expresses the freeze as
"every model parameter that is not the critic's", which puts the shared trunk on
the right side without special-casing it and picks up the critic trunk under
`separate: true` for free.

Flipping `network.separate` changes the model's state_dict keys, so checkpoints do
not carry across it in either direction. `set_weights` detects that specific
mismatch and raises naming the cause rather than letting PyTorch report a bare list
of missing keys.

Two details that are not obvious:

- The window is anchored to the epoch the checkpoint restored
  (`checkpoint_epoch + N`), not to 0, because a transfer inherits the epoch
  counter. A window measured from 0 would already be over before the first epoch
  of every iteration after the first.
- The lr schedule is frozen for the same window. With the policy fixed, measured
  KL is exactly 0, and an adaptive scheduler reads that as "far below target" and
  raises the lr on every mini-epoch — so the actor would unfreeze onto a learning
  rate near `lr_schedule_max`, the opposite of what the warmup is for.

Plasticity monitoring is suspended over the window too: a frozen trunk produces no
weight change, so its units read as dormant and could be replaced for no reason
other than having been deliberately held still.

The warmup epochs come out of the run's existing budget — `_effective_max_iterations`
is unchanged — so candidates stay comparable on equal sample budget.

## Module map (`src/`)

- `evaluation/local_runner.py` — builds + `docker run`s each candidate locally (one blocking `run`: build → run → result).
- `evaluation/hpc_runner.py` — `HPCRunner`: builds + pushes each candidate's image and drives the CARES scheduler (`submit`/`poll`/`collect`).
- `evaluation/reward_injection.py` — AST splice of `compute_reward` (+ two-output validation).
- `evaluation/workspace_manager.py` — builds per-candidate job codebases.
- `evaluation/result_processor.py` — reads the job's logs in place, writes the scalar summary (reward components first, each with its statistics and trend).
- `evaluation/scorer.py` — `FitnessScorer`: reads `fitness_function`, ranks candidates, summarises eval seeds.
- `evaluation/evaluator.py` — `RewardEvaluator`, the dispatch + capture orchestrator.
- `refinement/llm_agent.py` — `EurekaAgent` (proposes `compute_reward`, folds in feedback).
- `refinement/agent_config/*.txt` — LLM prompt templates.

## Configuration

- `configs/settings.yaml` — `tasks_repo`, `output_dir`, and the `runner` block.
  `runner.backend` picks `local` (`use_gpu`, `timeout_seconds`, `image`, optional
  `env`/`build_args`/`command_template`; Dockerfile built locally, no prebuilt tag)
  or `hpc` (a `runner.hpc` sub-block: `registry`, `upi` (or `$ARD_UPI`), `nas_outputs`,
  `max_runtime_hours`, `poll_seconds`, `datasets`, …).
- `configs/taskconfig.yaml` — `task`, `env_file` (whose `compute_reward` is the injection target), `description`, `max_iterations`.
- `configs/refineconfig.yaml` — `iteration`, `num_eval`, `base_seed`, and the `agent` (LLM) block.

The only secret is `OPENROUTER_API_KEY` (LLM). Each job's training image is built
locally from the staged codebase's `Dockerfile` — nothing is prebuilt.
