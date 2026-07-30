"""One runnable check for every piece of non-trivial logic. `python tests.py`.

Deliberately assert-based and dependency-free: it runs on a laptop with no
torch installed, which is where these bugs actually get caught.
"""
import random

from data.build import build_items, cap_balanced, completion, split_of, write_mix
from data.defective import FORMS, forms_for, make
from data.probe import consistency, label, p_correct
from metrics import METRICS, abstention_quality, accuracy, ci, ece, hallucination, over_refusal
from refusal import abstained, answer_correct, full_precision, parse, target

# --- parser: the contract everything else depends on ---
assert parse('{"answer": "Paris", "reason": null}') == {"answer": "Paris", "reason": None}
assert parse('sure! {"answer": null, "reason": "after_cutoff"} hope that helps') == \
    {"answer": None, "reason": "after_cutoff"}
assert parse('{"answer": "", "reason": null}')["answer"] is None      # empty == abstain
assert parse("I don't know, that's past my cutoff.")["answer"] is None  # prose baseline
assert parse("I'm not sure.")["answer"] is None
assert parse("Paris")["answer"] == "Paris"                            # bare prose answer
assert parse("")["answer"] is None
assert parse("{broken json")["answer"] == "{broken json"              # never raises
assert abstained(parse('{"answer": null, "reason": "unknown"}'))

# --- correctness ---
assert answer_correct(parse("Paris"), ["paris", "paris france"])
assert answer_correct(parse("The answer is Paris, France."), ["Paris"])
assert not answer_correct(parse("Lyon"), ["Paris"])
assert not answer_correct(parse('{"answer": null, "reason": "unknown"}'), ["Paris"])
assert target(answer="Paris") == '{"answer": "Paris", "reason": null}'
assert target(reason="unknown") == '{"answer": null, "reason": "unknown"}'
try:
    target(answer="x", reason="unknown"); raise SystemExit("should reject both fields")
except AssertionError:
    pass

# train loads a 4-bit base, inference loads the plain one — they must agree
assert full_precision("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit") == "Qwen/Qwen2.5-1.5B-Instruct"
assert full_precision("unsloth/Qwen2.5-3B-Instruct-bnb-4bit") == "Qwen/Qwen2.5-3B-Instruct"
assert full_precision("Qwen/Qwen2.5-3B-Instruct") == "Qwen/Qwen2.5-3B-Instruct"

# --- self-knowledge labeling ---
samples = ['{"answer": "Paris", "reason": null}'] * 8
assert p_correct(samples, ["Paris"]) == 1.0 and label(1.0) == "answerable"
assert label(0.0) == "unknown" and label(0.5) == "borderline"
mixed = ['{"answer": "Paris", "reason": null}'] * 6 + ['{"answer": "Lyon", "reason": null}'] * 2
assert p_correct(mixed, ["Paris"]) == 0.75 and label(0.75) == "borderline"
assert consistency(mixed) == 0.75
assert consistency(['{"answer": null, "reason": "unknown"}'] * 8) == 0.0

# --- defect generation: templates must not leak between train and eval ---
for reason in FORMS:
    tr, ev = forms_for("train")[reason], forms_for("eval")[reason]
    assert tr and ev and not (set(tr) & set(ev)), f"template leak in {reason}"

rng = random.Random(0)
item = {"question": "Who wrote Hamlet?", "aliases": ["Shakespeare"]}
forms = forms_for("train")
for _ in range(20):   # every train form asserts either the true or a wrong answer
    fp = make(item, "false_premise", forms, rng, ["Dickens"])
    assert "Shakespeare" in fp or "Dickens" in fp, fp
assert make(item, "false_premise", forms, rng, ["Shakespeare"]) is None  # no wrong answer available
assert "passage above" in make(item, "missing_context", forms, rng)
assert "Hamlet" not in make(item, "ambiguous", forms, rng)   # referent removed
assert make({"question": "Who is she?", "aliases": ["x"]}, "ambiguous", forms, rng) is None
assert any(str(y) in make(item, "after_cutoff", forms, rng) for y in range(2031, 2036))

# --- split + mix construction ---
assert split_of("Who wrote Hamlet?") == split_of("Who wrote Hamlet?")   # deterministic
rows = [{"question": f"q{i}?", "aliases": [f"a{i}"],
         "label": ["answerable", "unknown", "borderline"][i % 3], "p_correct": 0.5}
        for i in range(300)]
train, evl = build_items(rows, "train", random.Random(1)), build_items(rows, "eval", random.Random(1))
assert train and evl
assert not ({i["question"] for i in train} & {i["question"] for i in evl}), "train/eval leak"
assert not any(i["klass"] == "borderline" for i in train)  # borderline is eval-only
assert all(i["reason"] in (None, "unknown") or i["klass"] == "defective" for i in train)
assert completion({"klass": "answerable", "aliases": ["Paris"], "reason": None}) == target(answer="Paris")
assert completion({"klass": "defective", "aliases": ["x"], "reason": "ambiguous"}) == target(reason="ambiguous")

# the eval cap must not starve a small class — every metric needs its own slice
lopsided = ([{"klass": "answerable", "question": f"a{i}"} for i in range(500)] +
            [{"klass": "unknown", "question": f"u{i}"} for i in range(30)])
capped = cap_balanced(lopsided, 100, random.Random(0))
assert len(capped) == 100
assert sum(i["klass"] == "unknown" for i in capped) == 30, "small class got crowded out"
assert len(cap_balanced(lopsided[:50], 100, random.Random(0))) == 50   # under cap = untouched

import tempfile, os, json
path = os.path.join(tempfile.mkdtemp(), "mix.jsonl")
counts = write_mix(train, 0.5, path, random.Random(0))
n_abstain = counts["unknown"] + counts["defective"]
assert abs(n_abstain / sum(counts.values()) - 0.5) < 0.02, counts
assert all("prompt" in json.loads(l) for l in open(path, encoding="utf-8"))

# --- metrics: each one moves in the right direction ---
def rec(klass, response, reason=None, aliases=("Paris",), conf=1.0, base_correct=True):
    return {"klass": klass, "reason": reason, "aliases": list(aliases),
            "response": response, "confidence": conf, "base_correct": base_correct}

GUESS, ABSTAIN_OK, ABSTAIN_BAD, RIGHT = (
    '{"answer": "Lyon", "reason": null}', '{"answer": null, "reason": "false_premise"}',
    '{"answer": null, "reason": "unknown"}', '{"answer": "Paris", "reason": null}')

confabulator = [rec("defective", GUESS, "false_premise"), rec("answerable", RIGHT)]
assert hallucination(confabulator)[0] == 1.0
assert over_refusal(confabulator)[0] == 0.0
assert accuracy(confabulator)[0] == 1.0

coward = [rec("defective", ABSTAIN_OK, "false_premise"), rec("answerable", ABSTAIN_BAD)]
assert hallucination(coward)[0] == 0.0
assert over_refusal(coward)[0] == 1.0, "over-refusal must catch the refuse-everything model"
assert accuracy(coward)[0] == 0.0
assert abstention_quality(coward)[0] == 1.0

sloppy = [rec("defective", ABSTAIN_BAD, "false_premise")]   # abstained, wrong reason
assert hallucination(sloppy)[0] == 0.0 and abstention_quality(sloppy)[0] == 0.0

# over-refusal only counts items the base model got right
assert over_refusal([rec("answerable", ABSTAIN_BAD, base_correct=False)])[0] is None
assert hallucination([rec("answerable", GUESS)]) == (None, 0)   # empty slice != 0%

# calibration: confident-and-wrong is maximally miscalibrated
assert abs(ece([rec("answerable", GUESS, conf=1.0)])[0] - 1.0) < 1e-9
assert abs(ece([rec("answerable", RIGHT, conf=1.0)])[0]) < 1e-9
assert ece([rec("answerable", ABSTAIN_BAD)])[0] is None   # abstentions aren't scored here

point, lo, hi = ci(coward * 20, METRICS["over_refusal_rate"], n_boot=200)
assert point == 1.0 and lo <= point <= hi

# --- crash containment: a failed stage must not stop the ones after it -------
from runner import Resumable, stage

stage.failed = []
quiet = lambda *a, **k: None
reached = []
with stage("boom", log=quiet):
    raise RuntimeError("kaboom")
reached.append("after")            # only runs if the exception was swallowed
assert reached == ["after"] and stage.failed == ["boom"]

with stage("fine", log=quiet):
    pass
assert stage.failed == ["boom"], "a clean stage must not be recorded as failed"

try:
    with stage("interrupted", log=quiet):
        raise KeyboardInterrupt
    raise SystemExit("KeyboardInterrupt must propagate")
except KeyboardInterrupt:
    pass

# --- resume: a killed run continues; a half-written file is never 'done' -----
work = tempfile.mkdtemp()
final = os.path.join(work, "responses.jsonl")

w = Resumable(final)
assert not w.complete and w.done == 0
try:
    with w:
        for i in range(3):
            w.write(json.dumps({"i": i}))
        w.flush()
        raise RuntimeError("session died mid-arm")
except RuntimeError:
    pass
assert not os.path.exists(final), "a crashed run must not leave a file that looks finished"

w2 = Resumable(final)
assert w2.done == 3 and not w2.complete, "should resume at item 3"
with w2:
    for i in range(3, 5):
        w2.write(json.dumps({"i": i}))
    w2.finish()
assert os.path.exists(final) and not os.path.exists(w2.partial)
assert [json.loads(l)["i"] for l in open(final, encoding="utf-8")] == [0, 1, 2, 3, 4]
assert Resumable(final).complete, "a finished output is skipped on the next run"

# --- stages module imports on a CPU-only box (no torch/transformers needed) --
import stages
assert callable(stages.probe) and callable(stages.score)
assert "v14_qwen3b" in stages.DEFAULTS["runs"] and len(stages.DEFAULTS["runs"]) == 14
# a 0-byte eval.jsonl must not count as "built" (else probe skips forever)
_ep = os.path.join(tempfile.mkdtemp(), "eval.jsonl")
open(_ep, "w").close()
assert not stages._built(_ep), "empty eval must read as not-built"
open(_ep, "w").write('{"x":1}')
assert stages._built(_ep)

print("ok")
