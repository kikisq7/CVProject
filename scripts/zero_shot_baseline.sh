#!/bin/bash
# Run the pretrained LEGATO checkpoint (M0 baseline) against all three test
# sets so we can quantify the handwritten-domain gap and the catastrophic
# forgetting delta introduced by fine-tuning. Outputs go under
# outputs/legato-baseline/.
#
# Requires:
#   - guangyangmusic/legato downloaded (or HF access)
#   - datasets/PDMX-Synth, datasets/PDMX-SmashcimaHW, datasets/MUSCIMA-pp
#
# After running, feed the resulting test_predictions.json files into
# scripts/compute_ER.py / compute_TEDn.py / compute_OMR-NED.py.

set -eu
PORT=31400
OUT=outputs/legato-baseline
mkdir -p "$OUT"

run_predict() {
    local dataset_path=$1
    local tag=$2
    local mini_file_arg=$3
    local out_dir="$OUT/$tag"
    mkdir -p "$out_dir"

    cat > "$out_dir/config.json" <<JSON
{
    "model_config": "guangyangmusic/legato",
    "pretrained_model": "guangyangmusic/legato",
    "dataset_path": "$dataset_path",
    $mini_file_arg
    "output_dir": "$out_dir",
    "remove_unused_columns": false,
    "run_name": "legato-baseline-$tag",
    "do_train": false,
    "do_eval": false,
    "do_predict": true,
    "dataloader_num_workers": 4,
    "ddp_find_unused_parameters": false,
    "per_device_eval_batch_size": 1,
    "bf16_full_eval": true,
    "predict_with_generate": true,
    "generation_max_length": 2048,
    "generation_num_beams": 3,
    "bf16": true,
    "report_to": "none",
    "log_level": "info"
}
JSON

    PYTHONPATH=. accelerate launch --config_file configs/inference.yaml --main_process_port $PORT \
        scripts/train.py "$out_dir/config.json"
}

run_predict datasets/PDMX-SmashcimaHW smashcima '"mini_test_file": "datasets/mini_test_hw.json",'
run_predict datasets/MUSCIMA-pp      muscima   ''
run_predict datasets/PDMX-Synth      pdmx      '"mini_test_file": "datasets/mini_test.json",'
