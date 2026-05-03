"""Aggregate SER / TEDn / OMR-NED across (model x test_set) into a table.

Expects each model's evaluation outputs to live under a directory tree like::

    outputs/
        legato-baseline/
            eval_smashcima/{er.txt,tedn.txt,omr_ned.txt}
            eval_muscima/{...}
            eval_pdmx/{...}
        legato-lora/
            eval_smashcima/{...}
            eval_muscima/{...}
            eval_pdmx/{...}
        legato-dlora/
            eval_smashcima/{...}
            eval_muscima/{...}
            eval_pdmx/{...}

Produces a Markdown table on stdout plus a CSV at ``results.csv``.
"""

import argparse
import csv
import os
import re
from typing import Dict, Optional


NUMBER_RE = re.compile(r"([-+]?\d*\.\d+|\d+\.\d*e[-+]?\d+|\d+)")


def _grep_metric(path: str, patterns: Dict[str, str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in patterns}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m is None:
            continue
        nums = NUMBER_RE.findall(m.group(0))
        if nums:
            out[key] = float(nums[-1])
    return out


def collect(eval_dir: str) -> Dict[str, Optional[float]]:
    er = _grep_metric(
        os.path.join(eval_dir, "er.txt"),
        {"SER": r"(?i)SER[^0-9]*[0-9.eE+-]+", "CER": r"(?i)CER[^0-9]*[0-9.eE+-]+"},
    )
    tedn = _grep_metric(
        os.path.join(eval_dir, "tedn.txt"),
        {"TEDn": r"(?i)TEDn[^0-9]*[0-9.eE+-]+"},
    )
    omr = _grep_metric(
        os.path.join(eval_dir, "omr_ned.txt"),
        {"OMR-NED": r"(?i)OMR[-_ ]?NED[^0-9]*[0-9.eE+-]+"},
    )
    return {**er, **tedn, **omr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="outputs",
        help="Directory containing model-name subdirectories.",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["legato-baseline", "legato-lora", "legato-dlora"],
    )
    parser.add_argument(
        "--test_sets", nargs="+",
        default=["eval_smashcima", "eval_muscima", "eval_pdmx"],
    )
    parser.add_argument(
        "--out_csv", default="results.csv",
    )
    args = parser.parse_args()

    metric_names = ["SER", "CER", "TEDn", "OMR-NED"]

    rows = []
    for model in args.models:
        for ts in args.test_sets:
            row = {"model": model, "test_set": ts}
            row.update(collect(os.path.join(args.root, model, ts)))
            rows.append(row)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_set", *metric_names])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    # Markdown table on stdout.
    header = "| model | test_set | " + " | ".join(metric_names) + " |"
    sep = "|" + "|".join(["-----"] * (2 + len(metric_names))) + "|"
    print(header)
    print(sep)
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "-"
        print(
            f"| {r['model']} | {r['test_set']} | "
            + " | ".join(fmt(r.get(m)) for m in metric_names)
            + " |"
        )
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
