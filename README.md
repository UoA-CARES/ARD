# ARD — Autonomous RL Designer

ARD is an LLM-driven reward-design pipeline for reinforcement learning in NVIDIA
Isaac Lab. Given a task described in plain language, an LLM proposes reward
functions, ARD trains each one with PPO and scores it on a fixed evaluation
metric, then feeds the results back to the LLM to iterate — an Eureka-style loop
that searches for a reward that actually solves the task.

ARD is a **thin orchestrator**. It does not train locally and does not manage
machines. It is a client of two companion repos:

- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)** — the RL task substrate. Six
  tasks registered as `Isaac-ARD-*`, each isolating its reward in a single
  `_get_rewards` method (ARD's edit target) and logging a fixed `fitness_function`
  evaluation metric from `_get_dones`, independent of the reward.
- **[`parallel_coordination_system`](../parallel_coordination_system)** (PCS) — a
  coordinator that schedules containerised training jobs across a pool of GPU
  workers. ARD submits jobs to it over HTTP.

## How the loop works

```
                    ┌──────────────────────── ARD (this repo) ───────────────────────┐
  task description ─►  EurekaAgent ── proposes N _get_rewards methods                  │
                    │      ▲                          │                                │
                    │      │ feedback             AST inject each into a fresh         │
                    │      │ (fitness +           copy of ard-isaaclab-tasks           │
                    │      │  scalar summary)         │                                │
                    │   best run                  tar.gz codebase per candidate        │
                    │      │                          │                                │
                    │   ResultProcessor ◄── artifacts ── CoordinatorClient ──POST /jobs┼──► PCS coordinator
                    └──────────────────────────────────────────────────────────────────┘        │
                                                                                         schedules across
                                                                                         GPU workers; trains
                                                                                         PPO (rl_games); scores
                                                                                         by fitness_function
```

One refinement iteration:

1. **Generate.** The LLM proposes `sample` candidate `_get_rewards(self)` methods.
2. **Inject.** Each candidate is spliced into a fresh copy of `ard-isaaclab-tasks`
   via AST and packed into a `.tar.gz` codebase.
3. **Dispatch.** Each codebase is submitted as a job to the coordinator, which
   runs them concurrently across its GPU workers (`bash quickstart.sh <TASK>`).
4. **Score.** Finished jobs are downloaded; each is scored by its
   `fitness_function` (read from the training TensorBoard logs).
5. **Re-evaluate & feed back.** The best candidate is retrained `num_eval` times,
   and its training summary is fed back to the LLM to inform the next iteration.

The evaluation metric is **isolated in the task layer** — it lives in each task's
`_get_dones`, not `_get_rewards` — so the LLM can rewrite the reward freely
without ever altering the scoreboard it is judged on. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the injection mechanism and design
rationale.

## Prerequisites

- A **PCS coordinator** reachable over HTTP, with GPU workers registered and the
  training image (`pcs-isaaclab-ard:2.3.2`, built from `ard-isaaclab-tasks/Dockerfile`)
  present on those workers. See the PCS repo for standing one up.
- A local checkout of **`ard-isaaclab-tasks`** (referenced by `configs/settings.yaml`).
- An LLM endpoint (OpenRouter-compatible by default).
- Python 3.10+. ARD's own dependencies are light (no Isaac Lab needed locally):

```bash
pip install -r requirements.txt
```

## Configuration

Three YAML files under `configs/`:

| File | What it sets |
|---|---|
| `settings.yaml` | `tasks_repo`, `output_dir`, and the `coordinator` block (`base_url`, `token_env`, `docker_image`, `gpus`, `timeout_seconds`, `command_template`, …). |
| `taskconfig.yaml` | The task: `task` (e.g. `Isaac-ARD-Cartpole-v0`), `env_file` (the env whose `_get_rewards` is rewritten), `description` (the LLM's brief), `max_iterations`. |
| `refineconfig.yaml` | The loop: `iteration`, `num_eval`, `base_seed`, and the `agent` block (`model`, `base_url`, `sample`, `temperature`). |

Secrets come from the environment, never the configs:

```bash
export PCS_TOKEN=pcs_...           # coordinator bearer token
export OPENROUTER_API_KEY=...      # LLM key
```

## Running

```bash
python main.py --refine
# explicit configs:
python main.py --refine --settings configs/settings.yaml \
               --taskconfig configs/taskconfig.yaml \
               --refineconfig configs/refineconfig.yaml
# several tasks in sequence:
bash scripts/runrefine.sh
```

To refine a different task, point `taskconfig.yaml` at it (`task` + `env_file`):

| Task ID | Env file (under `ard-isaaclab-tasks`) |
|---|---|
| `Isaac-ARD-Cartpole-v0` | `…/tasks/direct/cartpole/cartpole_env.py` |
| `Isaac-ARD-Humanoid-v0` | `…/tasks/direct/locomotion/locomotion_env.py` |
| `Isaac-ARD-Franka-Cabinet-v0` | `…/tasks/direct/franka_cabinet/franka_cabinet_env.py` |
| `Isaac-ARD-Allegro-Repose-v0` | `…/tasks/direct/inhand_manipulation/inhand_manipulation_env.py` |
| `Isaac-ARD-Forge-NutThread-v0` | `…/tasks/direct/forge/forge_env.py` |
| `Isaac-ARD-Shadow-Hand-Over-v0` | `…/tasks/direct/shadow_hand_over/shadow_hand_over_env.py` |

## Output

Per task, under `output_dir/<task>/` (default `./runs/<task>/`):

- downloaded job artifacts (`<tag>.tar.gz`) and their extracted `logs/` trees,
- per-run `training_record/training_summary.txt` — the scalar summary fed to the LLM,
- console logs reporting each iteration's best candidate and its fitness.

## Repository layout

```
main.py                       CLI entry point + the refinement loop
configs/
  settings.yaml               coordinator endpoint, tasks_repo, output_dir
  taskconfig.yaml             task id, env file, description, max_iterations
  refineconfig.yaml           iterations, eval count, LLM agent settings
scripts/runrefine.sh          run the loop over one or more task configs
src/
  evaluation/
    coordinator_client.py     PCS HTTP client (submit / poll / artifacts / cancel)
    reward_injection.py       AST splice of the LLM reward into _get_rewards
    workspace_manager.py      build per-candidate job codebases (.tar.gz)
    result_processor.py       unpack artifacts, read fitness_function, summarize
    evaluator.py              RewardEvaluator — the orchestrator
  refinement/
    llm_agent.py              EurekaAgent — proposes rewards, folds in feedback
    agent_config/*.txt        LLM prompt templates
ARCHITECTURE.md               design notes: dispatch, injection, fitness isolation
```

## Notes

- Training itself never runs in this repo — it runs in the coordinator's workers
  inside the Isaac Lab image. ARD only needs an HTTP client and TensorBoard to
  read results, so it installs nothing from the Isaac Lab / rl_games stack.
- The coordinator does not auto-retry failed jobs; a failed candidate is recorded
  with its error and the loop continues with the others.
