"""Run this before every upload to Kaggle. `python preflight.py`.

GPU quota is the scarce resource; a malformed JSONL or a config pointing at a
mix that was never built costs an entire session to discover. This checks the
whole chain on CPU in a couple of seconds and tells you which stage you're at.

Exit code 0 = safe to upload. Warnings don't fail the run: "fewer items than
I'd like" is a caveat for the write-up, not a reason to block a training
session. Only things that make results *wrong* — corrupt files, a broken eval
lock, a config pointing at data that doesn't exist — are hard failures.
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter

import yaml

from refusal import parse

problems, warnings, notes = [], [], []


def jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                problems.append(f"{path}:{n} is not valid JSON ({e})")
    return rows


def have(path):
    return os.path.exists(path)


# --- stage 1: question pool -------------------------------------------------
if not have("data/questions.jsonl"):
    problems.append("data/questions.jsonl missing — run: python -m data.fetch")
else:
    qs = jsonl("data/questions.jsonl")
    notes.append(f"questions: {len(qs)}")
    if len(qs) < 1000:
        problems.append(f"only {len(qs)} questions; probing fewer than ~3000 gives too few "
                        "unknown-class items to train on")
    if any(not r.get("aliases") for r in qs):
        problems.append("some questions have no gold aliases — they can never be scored")

# --- stage 2: probe ---------------------------------------------------------
if not have("data/probe_raw.jsonl"):
    notes.append("probe_raw.jsonl missing -> next step is session1_data_train.ipynb")
else:
    probe = jsonl("data/probe_raw.jsonl")
    notes.append(f"probed: {len(probe)}")
    ks = Counter(len(r.get("samples", [])) for r in probe)
    if len(ks) > 1:
        notes.append(f"WARNING mixed sample counts {dict(ks)} — a resumed run with a different k")
    parsed = sum(bool(parse(s)["answer"] or parse(s)["reason"]) for r in probe[:200] for s in r["samples"])
    total = sum(len(r["samples"]) for r in probe[:200])
    if total and parsed / total < 0.5:
        problems.append(f"only {parsed/total:.0%} of probe samples parse into the response format — "
                        "check SYSTEM_PROMPT and max_new_tokens in the notebook before labeling")

# --- stage 3: labels + mixes ------------------------------------------------
if have("data/labeled.jsonl"):
    labeled = jsonl("data/labeled.jsonl")
    dist = Counter(r["label"] for r in labeled)
    notes.append(f"labels: {dict(dist)}")
    for k in ("answerable", "unknown"):
        if dist[k] < 200:
            warnings.append(f"only {dist[k]} '{k}' items — CIs will be wide. Probe more "
                            f"questions, or adjust KNOWN/UNKNOWN in data/probe.py (and say so "
                            "in the write-up). Not fatal — training still runs.")
else:
    notes.append("labeled.jsonl missing -> next step is: python -m data.probe")

if have("data/eval.jsonl"):
    items = jsonl("data/eval.jsonl")
    notes.append(f"eval: {len(items)} items {dict(Counter(i['klass'] for i in items))}")
    digest = hashlib.sha256(open("data/eval.jsonl", "rb").read()).hexdigest()
    lock = open("data/eval.lock").read().strip() if have("data/eval.lock") else ""
    if digest != lock:
        problems.append("data/eval.jsonl does not match data/eval.lock — the frozen eval was "
                        "modified. Do not re-lock it after training; freeze a new set instead")
    if len({i["question"] for i in items}) != len(items):
        problems.append("duplicate questions in the eval set")
else:
    notes.append("eval.jsonl missing -> next step is: python -m data.build")

# --- stage 4: configs -------------------------------------------------------
missing = {}
for cfg_path in sorted(glob.glob("configs/*.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    data_path = cfg["train"]["data_path"]
    if not have(data_path):
        missing.setdefault(data_path, []).append(os.path.basename(cfg_path))
    elif os.path.getsize(data_path) == 0:
        problems.append(f"{data_path} is empty")
for data_path, cfgs in missing.items():
    problems.append(f"{data_path} does not exist (needed by {', '.join(cfgs)}) "
                    "— run: python -m data.build")

for mix in sorted(glob.glob("data/mix_*.jsonl")):
    rows = jsonl(mix)
    bad = [r for r in rows if "prompt" not in r or "completion" not in r]
    if bad:
        problems.append(f"{mix}: {len(bad)} rows missing prompt/completion")
    abstain = sum(json.loads(r["completion"])["answer"] is None for r in rows)
    notes.append(f"{os.path.basename(mix)}: {len(rows)} rows, {abstain/len(rows):.0%} abstain"
                 if rows else f"{mix}: EMPTY")

# --- report -----------------------------------------------------------------
for n in notes:
    print(" ", n)
if warnings:
    print("\nWARNINGS (caveats, not blockers):")
    for w in warnings:
        print("  !", w)
if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print("  x", p)
    sys.exit(1)
print("\npreflight ok — safe to upload")
