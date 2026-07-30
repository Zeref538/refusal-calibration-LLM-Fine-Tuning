"""Mechanically derive the *defective* unanswerable class from answerable
items — false premise, missing context, ambiguous referent, past cutoff.

Why mechanical: correct-by-construction (we know the defect because we
injected it), free (NFR-3, no teacher model), and reproducible from a seed.

The trap this design avoids: if training and eval share templates, the model
learns the template, not the defect, and the eval reports a fake win. So each
reason has several surface forms and they are **split disjointly** — the eval
set only ever sees forms the model never trained on. `forms_for("train")` and
`forms_for("eval")` cannot overlap; the test asserts it.
"""
import random
import re

# {q} = original question (with '?'), {qs} = stem without '?', {gold}, {wrong}, {year}
FORMS = {
    "false_premise": [
        "{qs}, given that it was not {gold}?",
        "I've read that the answer to \"{q}\" is {wrong}. Explain how that came about.",
        "Since {wrong} is the correct answer here, {qs}?",
        "{qs}, and why was {gold} ruled out despite being wrong?",
    ],
    "missing_context": [
        "According to the passage above, {qs}?",
        "Based on the attached document, {qs}?",
        "In the file I just shared, {qs}?",
        "As stated in section 3 of the report, {qs}?",
    ],
    "ambiguous": [
        "{ambig}",
        "For that one, {ambig_stem}?",
        "Same question as before, but for the other one: {ambig_stem}?",
        "About it — {ambig_stem}?",
    ],
    "after_cutoff": [
        "As of {year}, {qs}?",
        "{qs}, according to the {year} revision?",
        "What did the {year} update change about this: {q}",
        "In {year}, {qs}?",
    ],
}

FUTURE_YEARS = (2031, 2032, 2033, 2034, 2035)

# Proper-noun-ish spans: what we blank out to create a dangling referent.
_ENTITY = re.compile(r"\b(?:[A-Z][\w'-]*)(?:\s+(?:of|the|de|van)?\s*[A-Z][\w'-]*)*\b")
_PRONOUNS = ("it", "that one", "them", "the other one")


def forms_for(split):
    """Disjoint template halves: even indices train, odd indices eval."""
    keep = 0 if split == "train" else 1
    return {k: [f for i, f in enumerate(v) if i % 2 == keep] for k, v in FORMS.items()}


def _ambiguate(question, rng):
    """Replace the first entity mention (skipping a sentence-initial word) with
    a bare pronoun, leaving a question with no resolvable referent."""
    for m in _ENTITY.finditer(question):
        if m.start() == 0 and " " not in m.group():
            continue  # "Who", "What" — not an entity
        if len(m.group()) < 3:
            continue
        return question[:m.start()] + rng.choice(_PRONOUNS) + question[m.end():]
    return None


def make(item, reason, forms, rng, wrong_pool=()):
    """-> question string, or None if this item can't carry this defect."""
    q = item["question"].strip()
    if not q.endswith("?"):
        q += "?"
    qs = q[:-1]
    gold = item["aliases"][0]

    if reason == "ambiguous":
        ambig = _ambiguate(q, rng)
        if not ambig:
            return None
        fields = {"ambig": ambig, "ambig_stem": ambig.rstrip("?")}
    elif reason == "false_premise":
        wrong = next((w for w in rng.sample(list(wrong_pool), min(8, len(wrong_pool)))
                      if w.lower() != gold.lower()), None)
        if not wrong:
            return None
        fields = {"q": q, "qs": qs, "gold": gold, "wrong": wrong}
    elif reason == "after_cutoff":
        fields = {"q": q, "qs": qs, "year": rng.choice(FUTURE_YEARS)}
    else:
        fields = {"q": q, "qs": qs}

    return rng.choice(forms[reason]).format(**fields)
