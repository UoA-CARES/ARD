# ARD (Autonomous RL Designer)

ARD is an LLM-driven reward-design pipeline for reinforcement learning in NVIDIA Isaac Lab. You describe a task in plain language, an LLM proposes reward functions, ARD trains each one with PPO and scores it on a fixed evaluation metric, then feeds the results back to the LLM so it can improve the next batch. This is an Eureka-style loop: it searches for a reward that actually solves the task.

ARD is a **thin orchestrator**. It does not carry the Isaac Lab / rl_games stack itself; each candidate trains inside a docker container. It depends on one companion repo:

- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)**, the RL task substrate. Three tasks are registered as `Isaac-ARD-*`, each isolating its reward in a single `_get_rewards` method (ARD's edit target) and logging a fixed `fitness_function` evaluation metric from `_get_dones`, independent of the reward.

For each candidate, ARD builds that repo's `Dockerfile`, then either `docker run`s it on the local machine one candidate at a time (`LocalRunner`), or builds, pushes, and submits it to the CARES HPC Scheduler, where the whole batch trains concurrently (`HPCRunner`). Which path is used is set by `runner.backend` in `configs/settings.yaml`. LLM generation is fanned out across threads either way; only the training step differs between backends.

## What's new in v0.3.0

- **HPC support.** Set `runner.backend: hpc` to train a whole batch of candidates at once on the CARES HPC Scheduler, instead of one at a time locally. See [Prerequisites](#prerequisites) and [Configuration](#configuration) below.
- **Updated task set.** `ard-isaaclab-tasks` dropped its older multi-task suite (Humanoid, Franka-Cabinet, Allegro-Repose, Forge-NutThread, Shadow-Hand-Over; still available at tag `v0.2.0`) in favor of three tasks: Cartpole, Shadow Hand Repose, and Shadow Hand Repose Vision. See the [task table](#running) below.

## How the loop works

```
                    ┌───────────────────── ARD (this repo) ─────────────────────┐
  task description ─►  EurekaAgent: proposes N _get_rewards methods              │
                    │      ▲                        │                           │
                    │      │ feedback           AST inject each into a fresh    │
                    │      │ (fitness +         copy of ard-isaaclab-tasks      │
                    │      │  scalar summary)       │                           │
                    │   best run                tar.gz codebase per candidate   │
                    │      │                        │                           │
                    │   ResultProcessor ◄── logs ── Runner (local or HPC)       │
                    └──────────────────────────────────────────────────────────┘
                        local backend: builds + docker runs one candidate at a time
                        hpc backend:   builds, pushes, and submits the whole batch at once
                        either way, trains PPO (rl_games) and scores by fitness_function
```

One refinement iteration:

1. **Generate.** The LLM proposes `sample` candidate `_get_rewards(self)` methods.
2. **Inject.** Each candidate is spliced into a fresh copy of `ard-isaaclab-tasks` via AST and packed into a `.tar.gz` codebase. With `warm_start` on and a previous iteration's winner already known, that winner's checkpoint is baked into the same tarball (see step 6), so every candidate this iteration resumes from it instead of random weights.
3. **Run.** Each codebase (with its `Dockerfile`) is trained according to `runner.backend`: the local backend builds and `docker run`s each candidate in turn; the HPC backend builds, pushes, and submits the whole batch to the CARES scheduler and trains it concurrently. Either way the task is selected via the job's config (`TASK`, plus `SEED` for eval runs).
4. **Score.** Each finished job's `logs/` are read from its work dir; each is scored by its `fitness_function` (from the training TensorBoard logs).
5. **Re-evaluate & feed back.** The iteration's best candidate is retrained `num_eval` times to de-noise its score, and its training summary is fed back to the LLM to inform the next iteration.
6. **Warm-start.** With `warm_start` enabled (`refineconfig.yaml`'s `warm_start: true`, or pass `--warm-start` — off by default), the de-noised winner's checkpoint is carried forward as the next iteration's starting point — read back in at step 2 above. Iteration 1 always cold-starts, since no previous winner exists yet.

This repeats every iteration, not just once at the end — each iteration both scores a winner and hands its checkpoint forward. Total trainings per task = `iteration * (sample + num_eval)`.

The evaluation metric is **isolated in the task layer**: it lives in each task's `_get_dones`, not `_get_rewards`, so the LLM can rewrite the reward freely without ever altering the scoreboard it is judged on. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the injection mechanism and design rationale.

## Prerequisites

Common to both backends:

- Python 3.10+. ARD's own dependencies are light (no Isaac Lab needed on the host):

  ```bash
  pip install -r requirements.txt
  ```

- A local checkout of **`ard-isaaclab-tasks`** (path set in `configs/settings.yaml`).
- An LLM endpoint (OpenRouter-compatible by default).

**Local backend** (`runner.backend: local`, the default):

- **docker** on the local machine, with the NVIDIA container runtime for GPU jobs (`--gpus all`). ARD builds the `ard-isaaclab-tasks` `Dockerfile` per job, so no training image needs to be prebuilt.

**HPC backend** (`runner.backend: hpc`, for the CARES HPC Scheduler):

- **docker** on the local machine, to build and push each candidate's image.
- the **hpc-client** package, installed and logged in:

  ```bash
  pip install -e /path/to/hpc-client
  hpc-client configure --scheduler-url http://<scheduler-host>:8080
  hpc-client login <upi>
  ```

  Also set `HPC_PASSWORD` so ARD can relogin automatically if this session expires mid-run; see [Secrets](#secrets) below.

- the CARES registry trusted as an insecure registry so `docker push` works (see `ard-isaaclab-tasks/docs/HPC.md` for the one-time Docker config).

## Configuration

Three YAML files under `configs/`:

| File | What it sets |
|---|---|
| `settings.yaml` | `tasks_repo`, `output_dir`, `build_root`, and the `runner` block (see below). |
| `taskconfig.yaml` | The task: `task` (e.g. `Isaac-ARD-Cartpole-v0`), `env_file` (the env whose `_get_rewards` is rewritten), `description` (the LLM's brief), `max_iterations`. |
| `refineconfig.yaml` | The loop: `iteration`, `num_eval`, `base_seed`, `warm_start` (default `false` — resume each iteration from the previous iteration's de-noised winner instead of random weights, when enabled; see [How the loop works](#how-the-loop-works)), and the `agent` block (`model`, `base_url`, `sample`, `temperature`). |

`runner.backend` in `settings.yaml` picks how candidates train:

- `local`: `use_gpu`, `timeout_seconds`, `image`, and optional `env` / `build_args` / `command_template`.
- `hpc`: an `hpc` block with `registry`, `image_repo`, `nas_outputs`, `max_runtime_hours`, `poll_seconds`, `datasets`, `job_name_prefix`, `extra_args`.

See the comments in `configs/settings.yaml` for what each key does.

### Secrets

Secrets come from the environment, never the configs. Add them to your shell's startup file so every new terminal has them, instead of exporting them by hand each time:

1. Open `~/.bashrc` in an editor.
2. Add these lines at the end:

   ```bash
   export OPENROUTER_API_KEY=...      # LLM key, required for every run
   export HPC_PASSWORD=...            # HPC backend only, lets ARD relogin automatically
   ```

3. Reload it in your current shell:

   ```bash
   source ~/.bashrc
   ```

`OPENROUTER_API_KEY` is always required. `HPC_PASSWORD` only matters with `runner.backend: hpc`: without it, an expired `hpc-client` session stops the run and asks you to run `hpc-client login` again by hand instead of ARD recovering on its own.

## Running

```bash
python main.py --refine
# a specific task by name (resolves its ard_meta.yaml in tasks_repo):
python main.py --refine --task cartpole
# explicit configs:
python main.py --refine --settings configs/settings.yaml \
               --taskconfig configs/taskconfig.yaml \
               --refineconfig configs/refineconfig.yaml
# enable warm-starting (off by default), overriding refineconfig.yaml's warm_start:
python main.py --refine --task cartpole --warm-start
```

To refine a different task, either point `taskconfig.yaml` at it (`task` + `env_file`) or pass `--task` with the directory name or the registered task ID:

| Task (`--task`) | Task ID | Env file (under `ard-isaaclab-tasks`) | Use for |
|---|---|---|---|
| `cartpole` | `Isaac-ARD-Cartpole-v0` | `…/tasks/direct/cartpole/cartpole_env.py` | smoke test |
| `shadow_hand` | `Isaac-ARD-Repose-Cube-Shadow-Direct-v0` | `…/tasks/direct/shadow_hand/shadow_hand_env.py` | full task, state observations |
| `shadow_hand_vision` | `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0` | `…/tasks/direct/shadow_hand_vision/shadow_hand_vision_env.py` | full task, vision observations |

The vision task does not run on the HPC backend yet (an RTX renderer issue on the worker GPUs); run it locally or in Docker instead.

## Output

Per task, under `output_dir/<task>/` (default `./runs/<task>/`):

- one directory per candidate, `<tag>/`, holding its `logs/` tree. The local backend writes here directly; the HPC backend copies the finished job's artifacts down from the NAS first. Either way, nothing is packed or re-downloaded once it lands.
- per-run `training_record/training_summary.txt`, the scalar summary fed to the LLM.
- console logs reporting each iteration's best candidate and its fitness.

### Sharing results

A finished task folder is mostly rl_games `.pth` checkpoints — for Shadow Hand Repose, 13 GB of the 16 GB — which nobody reading the metrics needs. `scripts/get_run_metrics.py` zips a run folder without them:

```bash
# one candidate: writes iter4_run_6/iter4_run_6.zip (205 MB -> 8.7 MB)
python scripts/get_run_metrics.py runs/Isaac-ARD-Repose-Cube-Shadow-Direct-v0/iter4_run_6
# a whole task folder, somewhere else (16 GB -> ~700 MB)
python scripts/get_run_metrics.py runs/Isaac-ARD-Repose-Cube-Shadow-Direct-v0 -o /tmp/shadow_hand.zip
```

Point it at either level — a task folder or a single candidate directory. By default the archive is written **inside that folder** as `<folder-name>.zip`, so it sits with the metrics it came from; `-o` puts it elsewhere. It keeps `status.json`, the hydra configs, `params/*.yaml`, `training_summary.txt`, container logs, and the TensorBoard event files, so the recipient can open the full training curves; it drops every `.pth`. Stdlib only, so it runs without ARD's dependencies installed.

Existing `.zip` files are never packaged, so zipping a whole task folder won't swallow archives you made for individual candidates, and re-running with `--force` replaces an archive rather than nesting it.

| Flag | Effect |
|---|---|
| `-o PATH` | Output zip, or a directory to write `<folder-name>.zip` into. Default is the run folder itself. |
| `--force` | Overwrite an existing output file (otherwise it refuses). |
| `--no-tfevents` | Also drop the TensorBoard event files. These are ~99% of what survives dropping `.pth`, so this turns a candidate's archive from 8.7 MB into ~114 KB of configs and logs. |
| `--exclude GLOB` | Drop anything else matching the glob; repeatable. |
| `--dry-run` | Report what would be packaged, and how much it saves, without writing. |

Run `--dry-run` first on a whole task folder — the scan is quick, the write is not.

## Repository layout

```
main.py                       CLI entry point and the refinement loop
configs/
  settings.yaml                runner backend (local/hpc), tasks_repo, output_dir
  taskconfig.yaml               task id, env file, description, max_iterations
  refineconfig.yaml             iterations, eval count, LLM agent settings
src/
  evaluation/                  builds and trains each candidate (local docker or
                                HPC), stages per-candidate codebases, captures results
  refinement/
    llm_agent.py                EurekaAgent: proposes rewards, folds in feedback
scripts/
  get_run_metrics.py           zip a run folder for sharing, minus the .pth checkpoints
ARCHITECTURE.md               design notes: execution, injection, fitness isolation
```

## Notes

- Training itself runs inside the per-job task container, not in this repo. ARD only needs the docker CLI (and, for the HPC backend, the `hpc-client`) plus TensorBoard to read results, so it installs nothing from the Isaac Lab / rl_games stack.
- A failed candidate is recorded with its error and the loop continues with the others; nothing is auto-retried.
