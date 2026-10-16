#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Pass the full path of a newly registered sweep as the first argument.
# Example: bash launch_sweep.sh kjsong/karate-club-gcn-sweep/tlvylrap
sweep_path="${1:?Provide ENTITY/PROJECT/SWEEP_ID}"
python -m wandb agent --count 15 "$sweep_path"
