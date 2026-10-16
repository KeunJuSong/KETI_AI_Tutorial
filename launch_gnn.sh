#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate ai_tutorial before launching; use existing W&B login credentials.
python wandb_sweep_Karate_club_gnn.py \
    --config configs/config.yaml \
    --gnn_model_config configs/gnn_model_config.yaml \
    --model_type GCN \
    --wandb \
    --wandb_project karate-club-gcn-sweep \
    --wandb_entity kjsong \
    "$@"
