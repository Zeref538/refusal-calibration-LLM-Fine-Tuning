"""FR-10: the tradeoff curve across every arm, as a markdown table plus an
ASCII plot of the two error directions against each other.

The single-number version of this project ("hallucination down 40%") is the
dishonest version. This is the deliverable: where each mix lands on the curve,
with intervals, including the mixes that landed badly.

Usage:
    python curve.py                       # every runs/*/responses.jsonl
    python curve.py runs/base runs/mix_50_50
"""
import glob
import json
import os
import sys

from eval import load_eval, load_records
from metrics import METRICS, ci

def base_arm_of(run_dir):
    """Over-refusal is defined against *the same family's* base model — comparing
    a 3B fine-tune to the 1.5B base would be meaningless. Generation writes the
    right one into meta.json; everything else defaults to `base`."""
    meta = os.path.join(run_dir, "meta.json")
    name = json.load(open(meta))["base_arm"] if os.path.exists(meta) else "base"
    return f"runs/{name}/responses.jsonl"


def main():
    dirs = sys.argv[1:] or sorted(os.path.dirname(p) for p in glob.glob("runs/*/responses.jsonl"))
    items = load_eval()
    arms = {os.path.basename(d): load_records(os.path.join(d, "responses.jsonl"), items,
                                              base_arm_of(d))
            for d in dirs}

    heads = list(METRICS)
    print("| run | " + " | ".join(heads) + " |")
    print("|" + "---|" * (len(heads) + 1))
    for name, records in arms.items():
        cells = []
        for metric in METRICS.values():
            point, lo, hi = ci(records, metric)
            cells.append("n/a" if point is None else f"{point:.1%} [{lo:.0%},{hi:.0%}]")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\nhallucination (x) vs over-refusal (y) — the knee is the deliverable")
    points = [(METRICS["hallucination_rate"](r)[0] or 0, METRICS["over_refusal_rate"](r)[0] or 0, n)
              for n, r in arms.items()]
    grid = [[" "] * 42 for _ in range(12)]
    marks = "0123456789ABCDEF"
    for m, (x, y, _) in zip(marks, points):
        col, row = min(41, int(x * 41)), min(11, int((1 - y) * 11))
        grid[row][col] = m
    for i, row in enumerate(grid):
        print(f"{(1 - i / 11):4.0%} |" + "".join(row))
    print("     +" + "-" * 42)
    print("      0%" + " " * 34 + "100%   hallucination")
    print("      key: " + ", ".join(f"{m}={n}" for m, (_, _, n) in zip(marks, points)))


if __name__ == "__main__":
    main()
