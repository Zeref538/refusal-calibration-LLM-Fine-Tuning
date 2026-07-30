"""The deliverable (PRD §4.3): both error directions, always reported together,
always with an interval.

Lean's honest-but-thin spot was a bare point estimate on n=100 — enough to see
an 18-point effect, not enough to defend a 3-point one. Every rate here comes
with a bootstrap CI, and the comparison helpers report the CI of the *paired
difference*, which is what the claim actually rests on.

A record is:
    {klass, reason, response, confidence, aliases, base_correct}
    klass ∈ answerable | unknown | defective | borderline
"""
import random
from collections import Counter

from refusal import abstained, answer_correct, parse

UNANSWERABLE = ("unknown", "defective")


def _rate(records, predicate):
    """-> (rate, n). Rate is None when nothing qualifies — never 0.0, which
    would silently read as a perfect score on an empty slice."""
    hits = [predicate(r) for r in records]
    return (sum(hits) / len(hits), len(hits)) if hits else (None, 0)


def hallucination(records):
    """Answered a question that has no answer. The headline failure."""
    return _rate([r for r in records if r["klass"] in UNANSWERABLE],
                 lambda r: not abstained(parse(r["response"])))


def over_refusal(records):
    """Abstained on a question the *base model got right* — so the abstention
    cost real capability. Scoring against base-correct items rather than all
    answerable ones is the point: it separates "the fine-tune became cowardly"
    from "the model never knew this anyway."."""
    return _rate([r for r in records if r["klass"] == "answerable" and r["base_correct"]],
                 lambda r: abstained(parse(r["response"])))


def accuracy(records):
    return _rate([r for r in records if r["klass"] == "answerable"],
                 lambda r: answer_correct(parse(r["response"]), r["aliases"]))


def abstention_quality(records):
    """Of the correct abstentions, how many name the right reason (FR-8)?
    A model that abstains on everything with "unknown" scores badly here — which
    is the intended pressure."""
    correct_abstentions = [r for r in records
                           if r["klass"] in UNANSWERABLE and abstained(parse(r["response"]))]
    return _rate(correct_abstentions, lambda r: parse(r["response"])["reason"] == r["reason"])


def ece(records, buckets=10):
    """Expected calibration error over answerable + borderline items, using
    self-consistency as confidence. Borderline items are included here and
    nowhere else — the mushy middle is exactly where calibration is tested."""
    scored = [(r["confidence"], answer_correct(parse(r["response"]), r["aliases"]))
              for r in records
              if r["klass"] in ("answerable", "borderline") and not abstained(parse(r["response"]))]
    if not scored:
        return None, 0
    total = 0.0
    for b in range(buckets):
        lo, hi = b / buckets, (b + 1) / buckets
        bucket = [(c, ok) for c, ok in scored if (lo <= c < hi or (b == buckets - 1 and c == 1.0))]
        if bucket:
            acc = sum(ok for _, ok in bucket) / len(bucket)
            conf = sum(c for c, _ in bucket) / len(bucket)
            total += len(bucket) / len(scored) * abs(acc - conf)
    return total, len(scored)


def reliability(records, buckets=10):
    """Rows for the reliability diagram: (bucket_mid, accuracy, n)."""
    scored = [(r["confidence"], answer_correct(parse(r["response"]), r["aliases"]))
              for r in records
              if r["klass"] in ("answerable", "borderline") and not abstained(parse(r["response"]))]
    out = []
    for b in range(buckets):
        lo, hi = b / buckets, (b + 1) / buckets
        bucket = [ok for c, ok in scored if lo <= c < hi or (b == buckets - 1 and c == 1.0)]
        if bucket:
            out.append(((lo + hi) / 2, sum(bucket) / len(bucket), len(bucket)))
    return out


METRICS = {
    "hallucination_rate": hallucination,
    "over_refusal_rate": over_refusal,
    "accuracy": accuracy,
    "abstention_quality": abstention_quality,
    "ece": ece,
}


def ci(records, metric, n_boot=1000, seed=0):
    """Percentile bootstrap over records -> (point, lo95, hi95)."""
    point = metric(records)[0]
    if point is None:
        return None, None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(len(records))] for _ in range(len(records))]
        value = metric(sample)[0]
        if value is not None:
            draws.append(value)
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def paired_delta(a_records, b_records, metric, n_boot=1000, seed=0):
    """CI of (b - a) resampling the same item indices in both arms — the only
    honest way to say "run B beats run A" when both ran on the same eval set."""
    assert len(a_records) == len(b_records)
    point = (metric(b_records)[0] or 0) - (metric(a_records)[0] or 0)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(a_records)) for _ in range(len(a_records))]
        va = metric([a_records[i] for i in idx])[0]
        vb = metric([b_records[i] for i in idx])[0]
        if va is not None and vb is not None:
            draws.append(vb - va)
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def summarize(records, name=""):
    lines = [f"== {name}  (n={len(records)}, {dict(Counter(r['klass'] for r in records))})"]
    for label, metric in METRICS.items():
        point, lo, hi = ci(records, metric)
        n = metric(records)[1]
        if point is None:
            lines.append(f"  {label:22s} n/a (no qualifying items)")
        else:
            lines.append(f"  {label:22s} {point:6.1%}  [{lo:.1%}, {hi:.1%}]  n={n}")
    return "\n".join(lines)
