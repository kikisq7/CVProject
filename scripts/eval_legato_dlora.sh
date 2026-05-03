#!/bin/bash
# Run prediction + all three metric scripts for a trained DLoRA (or LoRA)
# adapter across all three evaluation sets:
#   1. datasets/PDMX-SmashcimaHW  (in-domain handwritten)
#   2. datasets/MUSCIMA-pp        (out-of-distribution real handwritten)
#   3. datasets/PDMX-Synth        (forgetting check on typeset)
#
# Usage:
#   bash scripts/eval_legato_dlora.sh outputs/legato-dlora
#
# After prediction, ABC outputs are converted to MusicXML via utils/convert.py
# and passed to compute_TEDn / compute_OMR-NED.

set -eu

ADAPTER_DIR=${1:-outputs/legato-dlora}
PORT=31400

run_predict_and_metrics() {
    local predict_config=$1
    local out_dir=$2
    local dataset_path=$3

    mkdir -p "$out_dir"

    PYTHONPATH=. accelerate launch \
        --config_file configs/inference.yaml --main_process_port $PORT \
        scripts/train.py "$predict_config"

    local preds="$out_dir/test_predictions.json"
    if [[ ! -f "$preds" ]]; then
        echo "[skip] no $preds; prediction may have failed" >&2
        return
    fi

    # SER / CER
    PYTHONPATH=. python scripts/compute_ER.py \
        --prediction_file "$preds" \
        --ground_truth "$dataset_path" \
        | tee "$out_dir/er.txt" || true

    # ABC -> MusicXML
    DISPLAY=${DISPLAY:-:0} python utils/convert.py --input_file "$preds" || true

    local xml_preds="${preds%.json}_xml.json"
    if [[ -f "$xml_preds" ]]; then
        PYTHONPATH=. python scripts/compute_TEDn.py \
            --prediction_file "$xml_preds" \
            --ground_truth "$dataset_path" \
            --num_workers 4 \
            | tee "$out_dir/tedn.txt" || true

        PYTHONPATH=. python scripts/compute_OMR-NED.py \
            --prediction_file "$xml_preds" \
            --ground_truth "$dataset_path" \
            | tee "$out_dir/omr_ned.txt" || true
    else
        echo "[warn] ABC -> MusicXML conversion produced no $xml_preds" >&2
    fi
}

# Patch predict configs to point at this specific adapter directory.
python - <<PY
import json, os
for cfg in [
    "configs/legato-dlora-predict-smashcima.json",
    "configs/legato-dlora-predict-muscima.json",
    "configs/legato-dlora-predict-pdmx.json",
]:
    with open(cfg) as f:
        d = json.load(f)
    d["peft_adapter_path"] = os.environ.get("ADAPTER_DIR", "$ADAPTER_DIR")
    with open(cfg, "w") as f:
        json.dump(d, f, indent=4)
PY

run_predict_and_metrics \
    configs/legato-dlora-predict-smashcima.json \
    "$ADAPTER_DIR/eval_smashcima" \
    datasets/PDMX-SmashcimaHW

run_predict_and_metrics \
    configs/legato-dlora-predict-muscima.json \
    "$ADAPTER_DIR/eval_muscima" \
    datasets/MUSCIMA-pp

run_predict_and_metrics \
    configs/legato-dlora-predict-pdmx.json \
    "$ADAPTER_DIR/eval_pdmx" \
    datasets/PDMX-Synth
