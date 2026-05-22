# ARD Stage 2 — Architecture (post-coordinator migration)

ARD's reward-refinement loop was rebuilt around two external repos:

- **[`parallel_coordination_system`](../parallel_coordination_system)** (PCS) — a
  generic remote job runner. One coordinator (FastAPI + scheduler + SQLite) owns
  GPU-slot scheduling, SSH-to-worker dispatch, `docker run`, log capture, cleanup
  and artifact collection. ARD is now a **pure HTTP client** of it.
- **[`ard-isaaclab-tasks`](../ard-isaaclab-tasks)** — the IsaacLab task substrate.
  Six tasks registered as `Isaac-ARD-*`, each isolating its reward in a single
  `_get_rewards` method (the sole ARD edit target) and logging a fixed
  `fitness_function` evaluation metric.

## What changed from the old pipeline

| Concern | Old | New |
|---|---|---|
| Distribution | `ParallelExecutor` SSH'd into `machines_pool.txt` and ran `docker/run_remote_pipeline.sh` per task | Submit jobs to the PCS coordinator (`CoordinatorClient`); it owns scheduling + workers |
| Reward injection | git-checkout an in-tree project + **regex** replace of `@torch.jit.script compute_rewards` | Copy the tasks repo + **AST** rewrite of `_get_rewards` (`reward_injection.py`) |
| Eval metric | `Episode/consecutive_successes` | `fitness_function` (logged by every task; matched by tag suffix) |
| Result source | local TensorBoard path on the training host | downloaded job **artifacts tarball** (`logs/…/summaries/`) |
| LLM target | a `compute_rewards(...)` fn returning `(total_reward, components)` | a whole `_get_rewards(self)` method returning the reward |

## Fitness isolation (task layer)

The fixed evaluation metric (`fitness_function`) is **isolated in the task repo**,
out of `_get_rewards`. Each `Isaac-ARD-*` env computes it in a `_log_fitness()`
method called from `_get_dones` (a per-step method ARD never edits), from pure
environment state. So ARD rewriting `_get_rewards` cannot alter or drop the
scoreboard — that guarantee holds at the task layer, not just by convention.

## Reward injection — why the pristine body is kept

With fitness gone, `_get_rewards` still carries load-bearing **side effects** in
some tasks: franka refreshes intermediate values (also feeding observations);
inhand re-samples the goal on success and maintains `consecutive_successes`;
forge updates `prev_actions` / `success_pred_scale`.

So `reward_injection.inject_reward`:

1. keeps the **entire pristine `_get_rewards` body** (preserving those side effects),
2. demotes its terminal `return <expr>` to a bare expression statement,
3. appends `return self._ard_designed_reward()`, and
4. adds the LLM's proposed method as `_ard_designed_reward(self)`.

The LLM-designed reward is what gets returned; the task mechanics keep running.
Fitness is safe regardless, since it is logged from `_get_dones` before the
reward runs. (Once the remaining per-task mechanics are also relocated out of
`_get_rewards`, the injector could be simplified to a clean full replacement.)

## Flow (one refinement iteration)

```
EurekaAgent.func_gen  ──►  N candidate _get_rewards methods
        │
WorkspaceManager.build_codebase  ──►  per-candidate ard-isaaclab-tasks .tar.gz (reward injected)
        │
CoordinatorClient.submit_job  ──►  POST /jobs  (command: bash quickstart.sh <TASK>)
        │                          coordinator schedules across GPU workers
CoordinatorClient.wait_for_all
        │
CoordinatorClient.download_artifacts  ──►  <tag>.tar.gz
        │
ResultProcessor.process_artifacts  ──►  fitness_function + scalar summary
        │
RewardEvaluator picks best ──► EurekaAgent.receive_feedback (run phase, then eval phase)
```

## Module map (`src/`)

- `evaluation/coordinator_client.py` — PCS HTTP client (submit / poll / artifacts / cancel).
- `evaluation/reward_injection.py` — AST splice of `_get_rewards` (+ fitness preservation).
- `evaluation/workspace_manager.py` — builds per-candidate job codebases.
- `evaluation/result_processor.py` — unpacks artifacts, reads `fitness_function`, summarizes.
- `evaluation/evaluator.py` — `RewardEvaluator`, the orchestrator.
- `refinement/llm_agent.py` — `EurekaAgent` (proposes `_get_rewards`, folds in feedback).
- `refinement/agent_config/*.txt` — LLM prompt templates.

## Configuration

- `configs/settings.yaml` — `tasks_repo`, `output_dir`, and the `coordinator` block
  (`base_url`, `token_env`, `docker_image`, `gpus`, `timeout_seconds`, `command_template`).
- `configs/taskconfig.yaml` — `task`, `env_file` (the injection target), `description`, `max_iterations`.
- `configs/refineconfig.yaml` — `iteration`, `num_eval`, `base_seed`, and the `agent` (LLM) block.

Secrets come from the environment: `PCS_TOKEN` (coordinator bearer token) and
`OPENROUTER_API_KEY` (LLM). The training image is assumed prebuilt on the workers.
