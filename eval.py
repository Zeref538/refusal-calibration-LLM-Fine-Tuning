"""Score a run. Generation happens wherever there's a GPU; scoring happens
here, once, for every arm — base, prompt-baseline and every fine-tune.

Lean ran the same metric twice (eval.py on CPU, eval_kaggle.ipynb on GPU) and
they could have drifted apart without anyone noticing. Here the notebook only
*generates* `{question, response, confidence}`; every number comes out of this
file and metrics.py, so all arms are comparable by construction.

Usage:
    python eval.py --run runs/base/responses.jsonl --name base
    python eval.py --run runs/mix_50_50/responses.jsonl --base runs/base/responses.jsonl \
        --name mix_50_50 --compare
"""
import argparse
import hashlib
import json

from metrics import METRICS, ci, paired_delta, reliability, summarize


def load_eval(path="data/eval.jsonl", lock="data/eval.lock"):
    """NFR-4, enforced: the eval set is frozen after week 1. If this raises,
    the honest move is a new eval file with a new lock, not a re-hash."""
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    expected = open(lock).read().strip()
    if digest != expected:
        raise SystemExit(
            f"eval set changed since it was frozen ({digest[:12]} != {expected[:12]}).\n"
            "Editing the eval after training invalidates every number. Freeze a new set instead."
        )
    return {json.loads(l)["question"]: json.loads(l) for l in open(path, encoding="utf-8")}


def load_records(responses_path, items, base_path=None):
    """Join generations onto the frozen eval items (+ base correctness)."""
    base_correct = {}
    if base_path:
        from refusal import answer_correct, parse
        for line in open(base_path, encoding="utf-8"):
            row = json.loads(line)
            item = items[row["question"]]
            base_correct[row["question"]] = answer_correct(parse(row["response"]), item["aliases"])

    records = []
    for line in open(responses_path, encoding="utf-8"):
        row = json.loads(line)
        item = items[row["question"]]
        records.append({
            "question": row["question"],
            "klass": item["klass"],
            "reason": item["reason"],
            "aliases": item["aliases"],
            "response": row["response"],
            "confidence": row.get("confidence", 0.0),
            "base_correct": base_correct.get(row["question"], item["klass"] == "answerable"),
        })
    missing = len(items) - len(records)
    if missing:
        print(f"warning: {missing} eval items have no response in {responses_path}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="JSONL of {question, response, confidence}")
    ap.add_argument("--base", help="base-model responses; needed for over-refusal and --compare")
    ap.add_argument("--name", default="run")
    ap.add_argument("--compare", action="store_true", help="paired CI vs the base arm")
    ap.add_argument("--reliability", action="store_true")
    args = ap.parse_args()

    items = load_eval()
    records = load_records(args.run, items, args.base)
    print(summarize(records, args.name))

    if args.compare:
        if not args.base:
            raise SystemExit("--compare needs --base")
        base_records = load_records(args.base, items, args.base)
        by_q = {r["question"]: r for r in base_records}
        paired = [(by_q[r["question"]], r) for r in records if r["question"] in by_q]
        a, b = [p[0] for p in paired], [p[1] for p in paired]
        print(f"\n== {args.name} - base  (paired bootstrap, n={len(paired)})")
        for label, metric in METRICS.items():
            d, lo, hi = paired_delta(a, b, metric)
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  Δ {label:22s} {d:+6.1%}  [{lo:+.1%}, {hi:+.1%}]{sig}")
        print("  * = 95% CI excludes zero")

    if args.reliability:
        print(f"\n== reliability ({args.name})")
        for mid, acc, n in reliability(records):
            print(f"  conf~{mid:.2f}  acc={acc:.1%}  n={n}")


if __name__ == "__main__":
    main()
