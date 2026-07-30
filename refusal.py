"""The contract: one response format, one parser, one correctness check.

Everything (probing, data build, eval, metrics) routes through here, so a
change to the abstention format can never drift between train and eval —
that drift is what makes most "hallucination fix" numbers unreproducible.

Decision (PRD §9): abstention is a **structured signal**, not prose.
    {"answer": "Paris", "reason": null}          -> answered
    {"answer": null, "reason": "after_cutoff"}   -> abstained, with a why

Structured wins because (a) downstream agent code can branch on it without
regexing prose, and (b) "abstention quality" becomes mechanically gradeable
instead of a human read.
"""
import json
import re
import unicodedata

# The reason taxonomy. An abstention naming a reason outside this set, or the
# wrong one for the item, counts as a *low-quality* abstention in the eval.
REASONS = {
    "unknown": "the model has no reliable knowledge of this",
    "after_cutoff": "answering needs information past the training cutoff",
    "false_premise": "the question asserts something untrue",
    "missing_context": "a required referent/document was never supplied",
    "ambiguous": "the question has several distinct valid readings",
}

SYSTEM_PROMPT = (
    'Answer with one JSON object and nothing else.\n'
    'If you know the answer: {"answer": "<answer>", "reason": null}\n'
    'If you do not: {"answer": null, "reason": "<one of: '
    + ", ".join(sorted(REASONS))
    + '>"}\n'
    "Never guess. An unsupported answer is worse than an abstention."
)

# Prose fallbacks — a base model with no fine-tune will not emit our JSON, and
# scoring its plain-English refusals as hallucinations would flatter the
# fine-tune. Both baselines get parsed by the same forgiving path.
_PROSE_ABSTAIN = re.compile(
    r"\b(i (don'?t|do not) know|i'?m not (sure|certain)|no (reliable )?information|"
    r"cannot (be )?(answer|determine)|can'?t (answer|determine)|unable to (answer|determine)|"
    r"there is no |unanswerable)\b",
    re.I,
)


def parse(text):
    """-> {"answer": str|None, "reason": str|None}. Never raises."""
    text = (text or "").strip()
    obj = _first_json_object(text)
    if isinstance(obj, dict) and ("answer" in obj or "reason" in obj):
        answer, reason = obj.get("answer"), obj.get("reason")
        answer = answer.strip() if isinstance(answer, str) and answer.strip() else None
        reason = reason.strip().lower() if isinstance(reason, str) and reason.strip() else None
        return {"answer": answer, "reason": reason}
    if not text or _PROSE_ABSTAIN.search(text):
        return {"answer": None, "reason": None}  # abstained, but named no reason
    return {"answer": text, "reason": None}


def _first_json_object(text):
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def abstained(resp):
    return resp["answer"] is None


def normalize(s):
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def answer_correct(resp, aliases):
    """Correct = answered, and the answer matches any gold alias.

    Containment (not equality) because a model that answers "Paris, France"
    to "Paris" is right; the eval set is short-answer QA, so the false-positive
    risk of containment is small and one-sided in the base model's favour —
    which is the safe direction when the claim is "the fine-tune didn't lose
    accuracy."
    """
    if abstained(resp):
        return False
    got = normalize(resp["answer"])
    return any(a and (a in got or got in a) for a in (normalize(x) for x in aliases))


def full_precision(base_id):
    """Training uses a 4-bit base; inference loads the plain one and applies the
    same LoRA deltas on top. Kept here because train and eval must agree on it —
    Lean lost an afternoon to an adapter whose recorded base needed bitsandbytes.

        unsloth/Qwen2.5-3B-Instruct-bnb-4bit -> Qwen/Qwen2.5-3B-Instruct
    """
    name = base_id.split("/")[-1].replace("-bnb-4bit", "").replace("-unsloth", "")
    return name if "/" in name else f"Qwen/{name}"


def target(answer=None, reason=None):
    """The training-label side of the same contract."""
    assert (answer is None) != (reason is None), "exactly one of answer/reason"
    assert reason is None or reason in REASONS, f"unknown reason: {reason}"
    return json.dumps({"answer": answer, "reason": reason})
