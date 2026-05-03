#!/bin/bash
# Fine-tune LEGATO with asymmetric DLoRA (DoRA on vision encoder,
# LoRA on decoder cross-attention) on the Smashcima synthetic
# handwritten dataset. Requires datasets/PDMX-SmashcimaHW/ to exist.

PORT=31400
OMP_NUM_THREADS=16 WANDB_PROJECT=legato-dlora PYTHONPATH=. \
accelerate launch --config_file configs/zero2.yaml --main_process_port $PORT \
    scripts/train.py configs/legato-dlora.json
