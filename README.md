# Refusal Calibration

Teach a small open-weights model (Qwen2.5-1.5B-Instruct, LoRA, free T4) to say
**"I don't know — and here's why"** instead of confabulating, *without* turning
it into a model that refuses everything.

**→ [Read the case study](https://zeref538.github.io/refusal-calibration-LLM-Fine-Tuning/)**
(has a plain-English / technical toggle)

Spec: [REFUSAL_CALIBRATION_PRD.md](REFUSAL_CALIBRATION_PRD.md).
Predecessor: [../Lean](../Lean) — same discipline, two known weak spots fixed
(see [What's different from Lean](#whats-different-from-lean)).

**The deliverable is a curve, not a number.** Any refusal fine-tune can drive
hallucinations to zero by refusing everything. So every result here reports
both error directions, on the same frozen eval, with intervals.

**Status: complete.** 15 adapters trained, 19 arms generated and scored on the
frozen eval. Raw generations are committed under [runs/](runs/); the full scorer
output is [results/scores.txt](results/scores.txt).

## Results

All figures: point estimate [95% bootstrap CI], n=800 frozen eval items.

| arm | hallucination | over-refusal | accuracy | abstention quality | ECE |
|---|---|---|---|---|---|
| `base` (1.5B) | 95.5 [93.6, 97.3] | 0.0 | 100 *(by construction)* | 5.6 | 7.3 |
| `prompt` (1.5B) | 79.0 [74.7, 82.9] | 4.5 [1.9, 7.3] | 81.0 | 0.0 | 24.4 |
| `v3_mix50` (ref) | **3.0** [1.5, 4.9] | **61.5** [54.9, 68.4] | 29.0 | 51.0 | 17.4 |
| `base_3b` | 12.2 [9.2, 15.4] | 0.0 | 41.5 | 11.1 | 8.4 |
| `v14_qwen3b` (3B) | **6.8** [4.5, 9.3] | **9.6** [3.9, 17.0] | **45.5** | **64.6** | 7.3 |
| `v14_seed1` (3B, seed 1) | **9.5** [6.7, 12.4] | **7.2** [2.4, 13.0] | **49.5** | **64.9** | 11.8 |

Four findings, in the order they matter:

**1. The single-number headline is dishonest.** The reference fine-tune cuts
hallucination by 92.5 pp [−95.0, −90.0] — and pays 61.5 pp of over-refusal with
accuracy down 71.0 pp. Same checkpoint. Reporting only the first number
describes a model that got quieter, not better.

**2. The seed replicates invalidated the curve they were built to support.**
`v3`/`v12`/`v13` are the identical recipe under seeds 0/1/2:

| metric | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| hallucination | 3.0 | 13.8 | 12.8 | **10.8 pp** |
| over-refusal | 61.5 | 43.0 | 58.0 | **18.5 pp** |
| accuracy | 29.0 | 41.0 | 32.5 | **12.0 pp** |

Most gaps between the mix arms `v1`–`v5` are smaller than that spread, so the
ranking of 25 vs. 50 vs. 75% abstain training is **not resolvable at 1.5B** —
only the endpoints separate. Two runs that produced no headline of their own are
what makes the other twelve interpretable. Cost: 2.4 GPU-hours, ~13% of the
training budget.

**3. It's a capacity floor, not a broken method.** At 3B the same recipe cuts
hallucination 5.5 pp [−8.5, −2.5] while *gaining* 4.0 pp accuracy [−4.4, +12.4],
at 9.6 pp over-refusal instead of 61.5, and calibration held at base level
(ECE 8.4 → 7.3) — the only fine-tuned arm to manage that. A single-size study
would have published the wrong general claim. Finding 4 re-tests all of this
against a second seed; one claim survives intact and one does not.

**4. The 3B result replicated — and the stability is itself the finding.**
`v14` was one seed, which is the condition this project rejects everywhere else,
so it got a replicate. Seed spread at 3B versus the same spread at 1.5B:

| metric | 3B seed 0 | 3B seed 1 | spread | spread at 1.5B |
|---|---|---|---|---|
| hallucination | 6.8 | 9.5 | **2.7 pp** | 10.8 pp |
| over-refusal | 9.6 | 7.2 | **2.4 pp** | 18.5 pp |
| accuracy | 45.5 | 49.5 | **4.0 pp** | 12.0 pp |
| abstention quality | 64.6 | 64.9 | **0.3 pp** | 8.7 pp |
| ECE | 7.3 | 11.8 | 4.5 pp | 12.6 pp |

4–7× tighter. Both seeds hold over-refusal in single digits and both beat
`base_3b` on accuracy. So the wild seed sensitivity at 1.5B is a *capacity*
artifact, not a property of the recipe.

Two qualifications, because the replicate exists to produce exactly these:
seed 1's hallucination delta is −2.7 pp [−5.8, +0.3] — same direction as seed
0's −5.5 pp [−8.5, −2.5] but its CI crosses zero, so the hallucination
reduction is significant in one run of two. And ECE is the least stable axis
(7.3 vs 11.8 against a base of 8.4), so "calibration did not degrade" is
downgraded to "held on one seed, drifted mildly on the other." Over-refusal,
abstention quality and accuracy replicate cleanly.

Cost of the whole thing: **~31.2 GPU-hours, $0** on free Kaggle T4s (~$17 to
rent). The shortest path to the flattering headline was ~9.2 GPU-hours — the
extra 22 hours are the only reason it's knowable which gaps are real and which
are noise.

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
session 4  [GPU]   v14_seed1: replicate the 3B run, generate that one arm
```

Full scale (15 runs + 19 eval arms) spans four Kaggle sessions; that's why it's
four notebooks — [session4_seed_replicate.ipynb](session4_seed_replicate.ipynb)
trains one run and generates one arm, with the rest pre-seeded so they skip. Every stage is **crash-contained and resumable** — a
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
matched the fine-tune, that *would be* the finding and it would get published),
and all 14 fine-tunes. It didn't match: prompting bought a real −16.5 pp
[−21.4, −11.8] of the hallucination win for zero GPU-hours, but cost 19 pp of
accuracy, never emitted a valid reason code (abstention quality 0.0%), and
nearly tripled calibration error. At 3B it made hallucination *worse*.

Scoring prints to stdout and writes no file, and the session log came back
unreadable — so all 19 arms were re-scored locally on CPU from the
downloaded generations. That path is [run_score.py](run_score.py); its output is
committed as [results/scores.txt](results/scores.txt) and
[results/scores.json](results/scores.json).

**14 runs were configured**, each isolating exactly one variable against the
`v3_mix50` reference — five mix ratios (10/90 → 90/10) for the curve, **two seed
replicates** for a noise floor, two dosages, two LoRA ranks, one LR, one ablation
on whether naming the refusal reason buys anything, and a **3B** run to test
whether the result is size-dependent. Ordered by value in
[RUNBOOK.md](RUNBOOK.md): `v3` alone is publishable; `+v2/v4` is the curve;
`+v12/v13` is the noise floor that tells you which curve gaps are real.

That ordering turned out to be the single most valuable decision in the project:
the noise floor is what demoted the curve from "result" to "unresolvable at this
scale," and it would have been trivial to skip as the two runs with nothing new
to show.

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
| No seed replicates — couldn't tell a real gap from run-to-run noise | three seeds of the reference (`v3/v12/v13`); the spread came back at 18.5 pp on over-refusal, which is wider than most gaps in the curve — so the curve's ranking is reported as unresolvable rather than ranked |
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

Python · Unsloth + PEFT (LoRA) · Qwen2.5-1.5B/3B-Instruct · Kaggle T4 (free) ·
TriviaQA (`rc.nocontext`) · Hugging Face Hub. No paid API, no paid compute.

## Repo map

| path | what's in it |
|---|---|
| [docs/index.html](docs/index.html) | the [published case study](https://zeref538.github.io/refusal-calibration-LLM-Fine-Tuning/) — self-contained, no build step |
| [stages.py](stages.py) / [runner.py](runner.py) | the five pipeline stages; crash-containment and resume primitives |
| [data/](data/) | fetch, probe, defect generation, the frozen eval + `eval.lock` + training mixes |
| [configs/](configs/) | 17 run configs, all generated from the table in [make_configs.py](make_configs.py) |
| [runs/](runs/) | raw generations behind every number (adapters are gitignored) |
| [results/](results/) | scorer output — `scores.txt`, `scores.json` |
| [metrics.py](metrics.py) / [curve.py](curve.py) / [eval.py](eval.py) | every metric, CI and the curve; CPU-only |
| [tests.py](tests.py) | assert-based, dependency-free, runs without torch |
