#!/usr/bin/env bash
# Run the ARD reward-refinement loop for one or more task configs.
#
# Prereqs:
#   - A PCS coordinator reachable at settings.coordinator.base_url, with GPU
#     workers registered and the training image (settings.coordinator.docker_image)
#     present on them.
#   - export PCS_TOKEN=pcs_...          # coordinator bearer token
#   - export OPENROUTER_API_KEY=...     # LLM key
set -euo pipefail
cd "$(dirname "$0")/.."

# Task configs to refine (each is a configs/*.yaml with task/env_file/description).
taskconfigs=(
    "configs/taskconfig.yaml"
    # add more task configs here to refine several tasks in sequence
)

for config in "${taskconfigs[@]}"; do
    echo "=== Refining: $config ==="
    python main.py --refine --taskconfig "$config"
    echo "=== Done: $config ==="
done

echo "All refinement runs completed."
