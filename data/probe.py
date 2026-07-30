"""FR-3 self-knowledge labeling: turn k-sample probe output into class labels.

The core technique. We do not decide what the model "should" know from a
dataset's difficulty tag — we sample the base model k times per question and
measure. Questions it gets right nearly always are the answerable class;
questions it never gets right are the abstain class; the mushy middle is
dropped from *training* (teaching either behaviour there is a coin flip) but
kept for the calibration analysis, where the middle is the whole point.

Input:  data/probe_raw.jsonl  {question, aliases, samples: [text, ...]}
        (produced on a free GPU by session1_data_train.ipynb)
Output: data/labeled.jsonl    {question, aliases, p_correct, label}

Usage:
    python -m data.probe
"""
import json
import sys
from collections import Counter

from refusal import answer_correct, normalize, parse

# ponytail: two flat thresholds, sensitivity-checked in the write-up rather
# than tuned. If the knee moves, sweep these before touching anything else.
KNOWN, UNKNOWN = 0.8, 0.05


def p_correct(samples, aliases):
    hits = sum(answer_correct(parse(s), aliases) for s in samples)
    return hits / len(samples) if samples else 0.0


def label(p):
    if p >= KNOWN:
        return "answerable"
    if p <= UNKNOWN:
        return "unknown"
    return "borderline"


def consistency(samples):
    """Self-consistency of the *answered* samples — the confidence signal used
    for ECE at eval time (FR-9). No logprobs needed, so it works identically
    for the base model, the prompt baseline and the fine-tune."""
    answers = [normalize(r["answer"]) for r in map(parse, samples) if r["answer"]]
    if not answers:
        return 0.0
    return Counter(answers).most_common(1)[0][1] / len(samples)


def main(inp="data/probe_raw.jsonl", out="data/labeled.jsonl"):
    counts = Counter()
    with open(inp, encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            p = p_correct(row["samples"], row["aliases"])
            lab = label(p)
            counts[lab] += 1
            f_out.write(json.dumps({
                "question": row["question"],
                "aliases": row["aliases"],
                "p_correct": round(p, 3),
                "confidence": round(consistency(row["samples"]), 3),
                "label": lab,
            }) + "\n")
    total = sum(counts.values())
    for lab, n in counts.most_common():
        print(f"{lab:12s} {n:6d}  {n/total:5.1%}")
    print(f"thresholds: known>={KNOWN} unknown<={UNKNOWN}  total={total}")


if __name__ == "__main__":
    main(*sys.argv[1:])
