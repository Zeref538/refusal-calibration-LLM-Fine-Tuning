# Refusal Calibration

Teach a small open-weights model (Qwen2.5-1.5B-Instruct, LoRA, free T4) to say
**"I don't know — and here's why"** instead of confabulating, *without* turning
it into a model that refuses everything.

Spec: [REFUSAL_CALIBRATION_PRD.md](REFUSAL_CALIBRATION_PRD.md).
Predecessor: [../Lean](../Lean) — same discipline, two known weak spots fixed
(see [What's different from Lean](#whats-different-from-lean)).

**The deliverable is a curve, not a number.** Any refusal fine-tune can drive
hallucinations to zero by refusing everything. So every result here reports
both error directions, on the same frozen eval, with intervals.

## The four numbers, always together

| metric | what it catches |
|---|---|
| **Hallucination rate** | answered something unanswerable |
| **Over-refusal rate** | abstained on something *the base model got right* |
| **Accuracy** (answerable) | did the capability survive at all |
| **Abstention quality** | did the refusal name the *right* reason |
| **ECE / reliability** | is stated confidence worth anything |

Over-refusal is measured against base-correct items on purpose: it separates
"the fine-tune became cowardly" from "the model never knew that anyway."

## The contract

One response format, one parser ([refusal.py](refusal.py)), used by training,
probing, eval and any downstream agent:

```json
{"answer": "Paris", "reason": null}          // answered
{"answer": null, "reason": "after_cutoff"}   // abstained, with a machine-readable why
```

Reasons: `unknown · after_cutoff · false_premise · missing_context · ambiguous`.
Structured, not prose, because (a) a planner can branch on it and (b) it makes
"was the refusal *good*?" mechanically gradeable. The parser also accepts plain
English refusals, so the un-finetuned baselines aren't scored as hallucinating
when they politely decline.

## Pipeline

Step-by-step commands, run order and troubleshooting: **[RUNBOOK.md](RUNBOOK.md)**.

**Three session notebooks** run it across Kaggle's 9h-per-session limit —
[session1_data_train.ipynb](session1_data_train.ipynb) (probe + build + train the
curve), [session2_train.ipynb](session2_train.ipynb) (train the rest),
[session3_generate_score.ipynb](session3_generate_score.ipynb) (generate + score).
They share one tested module ([stages.py](stages.py)) so they can't drift; every
stage is resumable, so "Run All" is always the right button.

```
prep [done, committed]   data.fetch -> 25k questions ;  make_configs -> 14 run configs

session 1  [GPU]   probe (k=16)  -> build frozen eval + 7 mixes  -> train the curve + seeds
session 2  [GPU]   train the rest (dosage, capacity/LR, ablation, 3B)
session 3  [GPU]   generate every arm on the frozen eval  -> score (both axes, CIs, curve)
```

Full scale (14 runs + ~16 eval arms) spans three Kaggle sessions; that's why
it's three notebooks. Every stage is **crash-contained and resumable** — a
failure (OOM, bad config, dead kernel) is isolated to its stage, everything
finished stays on disk, and rerunning skips it. Generation even resumes a killed
arm from the exact item it died on. The stage functions live in
[stages.py](stages.py), the containment/resume primitives in
[runner.py](runner.py), both covered by [tests.py](tests.py).

Only stages 1, 4 and 5 need the GPU; everything else — including every metric —
reruns on a laptop from the saved generations:

```
python eval.py --run runs/v3_mix50/responses.jsonl --base runs/base/responses.jsonl --name v3_mix50 --compare
python curve.py
```

`preflight.py` validates the whole chain on CPU in two seconds. GPU quota is the
scarce resource, and a malformed JSONL should never cost a session to discover.

Arms scored: `base`, `prompt` (base + "say I don't know if unsure" — if this
matches the fine-tune, that *is* the finding and it gets published), and every
fine-tune you have quota for.

**14 runs are configured**, each isolating exactly one variable against the
`v3_mix50` reference — five mix ratios (10/90 → 90/10) for the curve, **two seed
replicates** for a noise floor, two dosages, two LoRA ranks, one LR, one ablation
on whether naming the refusal reason buys anything, and a **3B** run to test
whether the result is size-dependent. Ordered by value in
[RUNBOOK.md](RUNBOOK.md): `v3` alone is publishable; `+v2/v4` is the curve;
`+v12/v13` is the noise floor that tells you which curve gaps are real.

**Step 2 is the technique.** The abstain class isn't picked from a difficulty
label — it's measured. Sample the base model 16× per question: gets it right
≥80% of the time → answerable; ≤5% → abstain; in between → *borderline*, which
is excluded from training (teaching either behaviour there is a coin flip) and
kept for calibration only, where the mushy middle is the entire point.

**Defective items are built, not collected** ([data/defective.py](data/defective.py)) —
false premise, missing context, dangling referent, past-cutoff — derived from
questions the model demonstrably *does* know, so an abstention there is caused
by the injected defect and nothing else. Free, seeded, and correct by
construction.

## What's different from Lean

Kept — the parts that made Lean's write-up credible:

- Both axes reported together; a shorter/safer wrong answer is a failure, not a saving.
- Falsified hypotheses published, not buried.
- Free hardware, committed configs and seeds, small honest eval.
- Self-distillation: label from what the model *actually does*, not from what a dataset asserts.

Fixed — where Lean was thin:

| Lean's weak spot | here |
|---|---|
| Point estimates on n=100, no intervals — fine for an 18-pt effect, useless for a 3-pt one | bootstrap CIs on every rate, **paired** CIs on every comparison ([metrics.py](metrics.py)) |
| Metric implemented twice (`eval.py` CPU + `eval_kaggle.ipynb` GPU) — free to silently drift | notebook **only generates**; every number for every arm comes from one scorer |
| Four near-identical training notebooks, one per run | one [train.py](train.py) + one config per run from a single table ([make_configs.py](make_configs.py)); only the named variable differs |
| No seed replicates — couldn't tell a real gap from run-to-run noise | three seeds of the reference (`v3/v12/v13`); the scoring cell prints the spread as the smallest interpretable gap |
| Single model size — findings could be a 1.5B artifact | a 3B run (`v14`), scored against its own base/prompt arms |
| One n=100 eval, generated once with no recovery | ~800-item balanced eval; every stage crash-contained and every long output resumable ([runner.py](runner.py)) so a dead session never restarts finished work |
| Train/eval disjointness argued in prose | split by hash of the question, defect templates split disjointly too, asserted in [tests.py](tests.py) |
| Eval set could be edited after the fact | `data/eval.lock` — [eval.py](eval.py) refuses to score a modified eval set (NFR-4 in code, not in good intentions) |
| No val split during training | val loss tracked every 50 steps, so collapse is visible mid-run |
| One small correctness test | [tests.py](tests.py) covers parser, labeling, defect generation, split leakage, mix ratios, every metric's direction, crash-containment and resume |

## Test

```
python tests.py
```

Dependency-free and assert-based — runs on a laptop with no torch, which is
where these bugs actually get caught. It asserts the failure modes directly: a
confabulating model must score 100% hallucination, a refuse-everything model
must score 100% over-refusal, and an empty metric slice must read `n/a` rather
than a flattering 0%.

## Stack

Python · Unsloth + PEFT (LoRA) · Qwen2.5-1.5B-Instruct · Kaggle T4 (free) ·
TriviaQA (`rc.nocontext`) · Hugging Face Hub · GGUF → Ollama. No paid API.
