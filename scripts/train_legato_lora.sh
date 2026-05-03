#!/bin/bash
# Symmetric LoRA baseline: LoRA on both vision encoder and decoder cross-attn.
# Used for ablation against DLoRA (scripts/train_legato_dlora.sh).

PORT=31400
OMP_NUM_THREADS=16 WANDB_PROJECT=legato-dlora PYTHONPATH=. \
accelerate launch --config_file configs/zero2.yaml --main_process_port $PORT \
    scripts/train.py configs/legato-lora.json
