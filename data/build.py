"""Assemble the frozen eval set and the three training mixes (FR-1/2/4/7).

Split discipline, in order, because Lean's one real methodological soft spot
was proving disjointness by narrative rather than by construction:

  1. Every source question goes to `train` or `eval` by a hash of its text —
     deterministic, no seed to lose, no leak possible on a rerun.
  2. Defective items inherit the split of the question they were derived from,
     so a question can't appear answerable in train and defective in eval.
  3. Defect *templates* are also split (data/defective.py) — eval only shows
     surface forms never trained on.
  4. eval.jsonl is hashed into eval.lock. eval.py refuses to run against a
     modified set (NFR-4 enforced in code, not in good intentions).

Usage:
    python -m data.build
"""
import hashlib
import json
import os
import random
from collections import Counter

from data.defective import forms_for, make
from refusal import target

EVAL_FRACTION = 0.15
# Eval cost is n_items x (k+1) generations x every arm, so the eval set is
# capped rather than left to grow with the question pool: past ~800 items the
# CIs barely tighten, and the quota is better spent on more arms. Sampled
# balanced across classes so no class ends up too thin to carry a CI.
MAX_EVAL_ITEMS = 800
REASONS_DEFECTIVE = ("false_premise", "missing_context", "ambiguous", "after_cutoff")
# abstain fraction of the training mix. The extremes are on purpose: they anchor
# both ends of the tradeoff curve, so the knee in the middle is interpretable.
MIXES = {"mix_10_90": 0.10, "mix_25_75": 0.25, "mix_50_50": 0.50,
         "mix_75_25": 0.75, "mix_90_10": 0.90}


def split_of(question):
    h = int(hashlib.sha256(question.encode("utf-8")).hexdigest(), 16)
    return "eval" if (h % 1000) / 1000 < EVAL_FRACTION else "train"


def load(path="data/labeled.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_items(rows, split, rng):
    """-> list of {question, klass, reason, aliases, prompt-ready}."""
    rows = [r for r in rows if split_of(r["question"]) == split]
    answerable = [r for r in rows if r["label"] == "answerable"]
    unknown = [r for r in rows if r["label"] == "unknown"]
    borderline = [r for r in rows if r["label"] == "borderline"]
    forms = forms_for(split)
    wrong_pool = [r["aliases"][0] for r in answerable]

    items = []
    for r in answerable:
        items.append({"question": r["question"], "klass": "answerable",
                      "reason": None, "aliases": r["aliases"]})
    for r in unknown:
        items.append({"question": r["question"], "klass": "unknown",
                      "reason": "unknown", "aliases": r["aliases"]})
    # defective items are derived from *answerable* sources on purpose: the
    # model demonstrably knows the underlying fact, so an abstention here is
    # caused by the injected defect and nothing else.
    for r, reason in zip(answerable, _cycle(REASONS_DEFECTIVE, len(answerable))):
        q = make(r, reason, forms, rng, wrong_pool)
        if q:
            items.append({"question": q, "klass": "defective",
                          "reason": reason, "aliases": r["aliases"]})
    # borderline: eval-only, and only for the calibration curve — never trained
    # on, never scored as hallucination or over-refusal.
    if split == "eval":
        for r in borderline:
            items.append({"question": r["question"], "klass": "borderline",
                          "reason": None, "aliases": r["aliases"]})
    return items


def cap_balanced(items, cap, rng):
    """Round-robin across classes until the cap, so a big class can't crowd out
    a small one — every metric needs its own slice to be large enough to bound."""
    if len(items) <= cap:
        return items
    pools = {}
    for i in items:
        pools.setdefault(i["klass"], []).append(i)
    for pool in pools.values():
        rng.shuffle(pool)
    picked, klasses = [], sorted(pools)
    while len(picked) < cap and any(pools.values()):
        for k in klasses:
            if pools[k] and len(picked) < cap:
                picked.append(pools[k].pop())
    return picked


def _cycle(seq, n):
    return [seq[i % len(seq)] for i in range(n)]


def completion(item):
    if item["klass"] == "answerable":
        return target(answer=item["aliases"][0])
    return target(reason=item["reason"])


def write_mix(items, ratio, path, rng):
    """Sample to the requested abstain:answer ratio, largest set that fits."""
    answer = [i for i in items if i["klass"] == "answerable"]
    abstain = [i for i in items if i["klass"] in ("unknown", "defective")]
    n = min(int(len(answer) / (1 - ratio)), int(len(abstain) / ratio)) if 0 < ratio < 1 else 0
    picked = rng.sample(answer, n - int(n * ratio)) + rng.sample(abstain, int(n * ratio))
    rng.shuffle(picked)
    with open(path, "w", encoding="utf-8") as f:
        for i in picked:
            f.write(json.dumps({"prompt": i["question"], "completion": completion(i)}) + "\n")
    return Counter(i["klass"] for i in picked)


def main():
    rng = random.Random(0)
    rows = load()
    os.makedirs("data", exist_ok=True)

    eval_items = build_items(rows, "eval", random.Random(1))
    # dedupe: two defect transforms can collide on the same surface string, and
    # a duplicated eval question would be silently double-counted.
    eval_items = list({i["question"]: i for i in eval_items}.values())
    eval_items = cap_balanced(eval_items, MAX_EVAL_ITEMS, random.Random(2))
    with open("data/eval.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(eval_items, key=lambda i: i["question"]):
            f.write(json.dumps(i) + "\n")
    digest = hashlib.sha256(open("data/eval.jsonl", "rb").read()).hexdigest()
    open("data/eval.lock", "w").write(digest + "\n")
    print("eval:", Counter(i["klass"] for i in eval_items), "sha256:", digest[:12])

    train_items = build_items(rows, "train", rng)
    for name, ratio in MIXES.items():
        counts = write_mix(train_items, ratio, f"data/{name}.jsonl", rng)
        print(f"{name}: {dict(counts)} total={sum(counts.values())}")

    # Ablation data (FR-2): same 50/50 items, but every abstention target
    # collapses to a bare "unknown". If the reason-naming version isn't better,
    # FR-2 was decoration and the write-up should say so.
    bare = [dict(i, reason="unknown" if i["klass"] != "answerable" else None) for i in train_items]
    counts = write_mix(bare, 0.5, "data/mix_50_50_bare.jsonl", random.Random(0))
    print(f"mix_50_50_bare: {dict(counts)} total={sum(counts.values())}")


if __name__ == "__main__":
    main()
