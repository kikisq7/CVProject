"""Run a trained T4 adapter (or the zero-shot baseline) across many test sets.

Designed for compute-budget-constrained settings: each test set gets a
small slice (default 100 samples) and only SER/CER are computed by default.
You still get a meaningful "model x test condition" matrix at the end.

Test conditions covered (auto-skipped if the dataset directory is missing):

    eval_smashcima_clean
    eval_smashcima_rotate
    eval_smashcima_blur
    eval_smashcima_noise
    eval_smashcima_jpeg
    eval_smashcima_downscale
    eval_pdmx                  (catastrophic-forgetting check)
    eval_muscima               (real handwritten OOD, if present)

Each condition produces:

    outputs/<run_tag>/<condition>/test_predictions.json
    outputs/<run_tag>/<condition>/er.txt

After all conditions finish, a Markdown summary table is printed and
``outputs/<run_tag>/summary.csv`` is written.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env={**os.environ, **(env or {})})


def _write_predict_config(
    out_path: str,
    *,
    model_config: str,
    pretrained_model: str,
    adapter_path: Optional[str],
    dataset_path: str,
    output_dir: str,
    load_in_4bit: bool,
    fp16: bool,
    generation_max_length: int,
    generation_num_beams: int,
    mini_test_file: Optional[str] = None,
):
    cfg = {
        "model_config": model_config,
        "pretrained_model": pretrained_model,
        "dataset_path": dataset_path,
        "dummy_data": False,
        "output_dir": output_dir,
        "remove_unused_columns": False,
        "run_name": f"eval-{os.path.basename(output_dir)}",
        "do_train": False,
        "do_eval": False,
        "do_predict": True,
        "dataloader_num_workers": 2,
        "ddp_find_unused_parameters": False,
        "per_device_eval_batch_size": 1,
        "predict_with_generate": True,
        "generation_max_length": generation_max_length,
        "generation_num_beams": generation_num_beams,
        "report_to": "none",
        "log_level": "info",
        "load_in_4bit": load_in_4bit,
        "bnb_4bit_compute_dtype": "float16",
    }
    if fp16:
        cfg["fp16"] = True
        cfg["fp16_full_eval"] = True
    else:
        cfg["bf16"] = True
        cfg["bf16_full_eval"] = True
    if adapter_path:
        cfg["peft_adapter_path"] = adapter_path
    if mini_test_file:
        cfg["mini_test_file"] = mini_test_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _grep_metric(text: str, key: str) -> Optional[float]:
    import re

    m = re.search(rf"(?i){re.escape(key)}[^0-9-]*([-+]?\d*\.\d+|\d+)", text)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_tag", default="legato-dlora-t4")
    parser.add_argument(
        "--adapter_path", default="outputs/legato-dlora-t4",
        help="Adapter directory; pass '' to evaluate the zero-shot base model.",
    )
    parser.add_argument("--model_config", default="guangyangmusic/legato-small")
    parser.add_argument("--pretrained_model", default="guangyangmusic/legato-small")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--generation_max_length", type=int, default=768)
    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--stress_root", default="datasets/stress")
    parser.add_argument("--smashcima_dataset", default="datasets/PDMX-SmashcimaHW")
    parser.add_argument("--pdmx_dataset", default="datasets/PDMX-Synth")
    parser.add_argument("--pdmx_mini_test", default="datasets/mini_test.json")
    parser.add_argument("--muscima_dataset", default="datasets/MUSCIMA-pp")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Subset of condition names to run; defaults to all available.",
    )
    args = parser.parse_args()

    out_root = os.path.join("outputs", args.run_tag)
    os.makedirs(out_root, exist_ok=True)
    adapter_path = args.adapter_path or None

    plan: List[Dict] = []
    # Smashcima stress-test conditions.
    for name in ("clean", "rotate", "blur", "noise", "jpeg", "downscale"):
        ds_path = os.path.join(args.stress_root, name)
        if os.path.isdir(ds_path):
            plan.append({
                "condition": f"eval_smashcima_{name}",
                "dataset_path": ds_path,
                "mini_test_file": None,
            })
    # Untouched original handwritten test split.
    if os.path.isdir(args.smashcima_dataset):
        plan.append({
            "condition": "eval_smashcima_full",
            "dataset_path": args.smashcima_dataset,
            "mini_test_file": None,
        })
    # Forgetting check on PDMX typeset.
    if os.path.isdir(args.pdmx_dataset):
        plan.append({
            "condition": "eval_pdmx",
            "dataset_path": args.pdmx_dataset,
            "mini_test_file": args.pdmx_mini_test if os.path.isfile(args.pdmx_mini_test) else None,
        })
    # Real handwritten OOD if available.
    if os.path.isdir(args.muscima_dataset):
        plan.append({
            "condition": "eval_muscima",
            "dataset_path": args.muscima_dataset,
            "mini_test_file": None,
        })

    if args.conditions:
        plan = [p for p in plan if p["condition"] in args.conditions]
    if not plan:
        print("No eval conditions are available; build datasets first.", file=sys.stderr)
        return 1

    summary_rows = []
    for entry in plan:
        cond = entry["condition"]
        out_dir = os.path.join(out_root, cond)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "predict_config.json")
        _write_predict_config(
            cfg_path,
            model_config=args.model_config,
            pretrained_model=args.pretrained_model,
            adapter_path=adapter_path,
            dataset_path=entry["dataset_path"],
            output_dir=out_dir,
            load_in_4bit=args.load_in_4bit,
            fp16=args.fp16,
            generation_max_length=args.generation_max_length,
            generation_num_beams=args.generation_num_beams,
            mini_test_file=entry["mini_test_file"],
        )
        rc = _run(["python", "scripts/train.py", cfg_path], env={"PYTHONPATH": "."})
        if rc != 0:
            print(f"[warn] prediction failed for {cond}; continuing.", file=sys.stderr)
            summary_rows.append({"condition": cond, "SER": None, "CER": None})
            continue

        preds = os.path.join(out_dir, "test_predictions.json")
        if not os.path.isfile(preds):
            print(f"[warn] no predictions at {preds}", file=sys.stderr)
            summary_rows.append({"condition": cond, "SER": None, "CER": None})
            continue

        er_path = os.path.join(out_dir, "er.txt")
        rc = _run(
            ["python", "scripts/compute_ER.py",
             "--prediction_file", preds,
             "--ground_truth", entry["dataset_path"]],
            env={"PYTHONPATH": "."},
        )
        if rc == 0 and os.path.isfile(preds):
            try:
                # Re-run capturing stdout for the summary table.
                output = subprocess.check_output(
                    ["python", "scripts/compute_ER.py",
                     "--prediction_file", preds,
                     "--ground_truth", entry["dataset_path"]],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                )
                with open(er_path, "w") as f:
                    f.write(output)
                summary_rows.append({
                    "condition": cond,
                    "SER": _grep_metric(output, "SER"),
                    "CER": _grep_metric(output, "CER"),
                })
                continue
            except subprocess.CalledProcessError:
                pass
        summary_rows.append({"condition": cond, "SER": None, "CER": None})

    # Markdown table + CSV.
    print("\n## Multi-condition evaluation summary")
    print("| condition | SER | CER |")
    print("|---|---|---|")
    for r in summary_rows:
        ser = f"{r['SER']:.4f}" if r["SER"] is not None else "-"
        cer = f"{r['CER']:.4f}" if r["CER"] is not None else "-"
        print(f"| {r['condition']} | {ser} | {cer} |")

    csv_path = os.path.join(out_root, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("condition,SER,CER\n")
        for r in summary_rows:
            ser = f"{r['SER']:.6f}" if r["SER"] is not None else ""
            cer = f"{r['CER']:.6f}" if r["CER"] is not None else ""
            f.write(f"{r['condition']},{ser},{cer}\n")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
