#!/usr/bin/env bash
# Run the ARD reward-refinement loop for one or more tasks.
#
# Prereqs:
#   - A PCS coordinator reachable at settings.coordinator.base_url, with GPU
#     workers registered and the training image (settings.coordinator.docker_image)
#     present on them.
#   - export TOKEN=pcs_...          # coordinator bearer token
#   - export OPENROUTER_API_KEY=...     # LLM key
#
# Usage:
#   scripts/runrefine.sh                       # refine all tasks (auto-discovered)
#   scripts/runrefine.sh --tasks cartpole forge  # refine specific tasks by dir name
#   scripts/runrefine.sh --config configs/taskconfig.yaml  # refine a single config
set -euo pipefail
cd "$(dirname "$0")/.."

SETTINGS="${SETTINGS:-configs/settings.yaml}"

# Extract tasks_repo from settings.yaml
TASKS_REPO="$(python3 -c "
import yaml, sys
with open('$SETTINGS') as f:
    cfg = yaml.safe_load(f)
print(cfg['tasks_repo'])
")"

TASK_DIRS_ROOT="$TASKS_REPO/source/ard_tasks/ard_tasks/tasks/direct"

# --- resolve task config list ---
taskconfigs=()

if [[ $# -eq 0 ]]; then
    # Auto-discover all ard_meta.yaml files in the tasks repo
    while IFS= read -r meta; do
        taskconfigs+=("$meta")
    done < <(find "$TASK_DIRS_ROOT" -maxdepth 2 -name "ard_meta.yaml" | sort)

    if [[ ${#taskconfigs[@]} -eq 0 ]]; then
        echo "No ard_meta.yaml files found under $TASK_DIRS_ROOT" >&2
        exit 1
    fi
    echo "Auto-discovered ${#taskconfigs[@]} tasks."

elif [[ "$1" == "--config" ]]; then
    # Single explicit config file
    taskconfigs=("$2")

elif [[ "$1" == "--tasks" ]]; then
    # Named task directories, e.g. --tasks cartpole forge
    shift
    for name in "$@"; do
        meta="$TASK_DIRS_ROOT/$name/ard_meta.yaml"
        if [[ ! -f "$meta" ]]; then
            echo "No ard_meta.yaml found for task '$name' at: $meta" >&2
            exit 1
        fi
        taskconfigs+=("$meta")
    done
fi

# --- run ---
failed=()
for config in "${taskconfigs[@]}"; do
    task_label="$(basename "$(dirname "$config")")"
    echo ""
    echo "=== Refining: $task_label ($config) ==="
    if python main.py --refine --taskconfig "$config"; then
        echo "=== Done: $task_label ==="
    else
        echo "=== FAILED: $task_label ===" >&2
        failed+=("$task_label")
    fi
done

echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All ${#taskconfigs[@]} refinement runs completed."
else
    echo "${#failed[@]}/${#taskconfigs[@]} runs failed: ${failed[*]}"
    exit 1
fi
