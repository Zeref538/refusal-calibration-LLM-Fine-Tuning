# PRD — Refusal Calibration Fine-Tune

> Teach a small open-weights model to say **"I don't know"** when it doesn't,
> without turning it into a model that refuses everything.

**Status:** draft · **Target start:** after the interview agent ships ·
**Est. effort:** ~5 weeks part-time · **Name:** TBD

---

## 1. Problem

Small models confabulate. Asked something outside their knowledge — an
unanswerable question, a false premise, a missing passage — they produce a
confident, fluent, wrong answer instead of declining. This is the single most
cited reason teams distrust local models.

The naive fix (prompt it to be cautious, or train it on refusals) creates the
opposite failure: **over-refusal**, where the model declines questions it could
have answered correctly. That failure is less visible and just as harmful, and
most published "hallucination fixes" never measure it.

**The thesis of this project:** refusal is a *calibration* problem, not a
behavior to maximize. Both directions must be measured, and the honest result
is a tradeoff curve — not a single number.

## 2. Goals

- Fine-tune a small model (LoRA, free GPU) so that it abstains on
  unanswerable inputs and still answers answerable ones.
- Measure and publish **both** error directions plus a calibration metric.
- Produce a reusable recipe and an honest write-up, including where it fails.

### Non-goals

- Not a safety/harmful-content refusal project — this is about *epistemic*
  refusal (not knowing), not policy refusal (not allowed).
- Not RAG. Retrieval is a different fix for the same symptom; comparing against
  it is optional context, not the deliverable.
- No RLHF/DPO in v1 — supervised LoRA only, on free hardware.

## 3. Users

| user | need |
|---|---|
| Primary — builders of local-first agents (me, for [YODA]/[the butler agent]) | A planner/answerer that declines instead of inventing, since downstream code executes its output |
| Secondary — hiring managers | Evidence the candidate can diagnose a model behavior problem, design a two-sided eval, and report an honest tradeoff |

## 4. Requirements

### 4.1 Data
- **FR-1** Build a training set with three classes:
  - **Answerable** — model should answer (and be correct).
  - **Unanswerable-unknown** — outside knowledge; correct behavior is abstain.
  - **Unanswerable-defective** — false premise, missing context, ambiguous
    referent; correct behavior is abstain *and name the defect*.
- **FR-2** Abstention targets must be informative: *"I don't know — that
  requires data after my training cutoff"* beats a bare "I don't know."
- **FR-3** **Self-knowledge labeling:** determine what the base model actually
  knows by sampling it k times per question and checking correctness, rather
  than assuming. Questions it answers correctly ≥ threshold → answerable class;
  near-zero → abstain class. *(This is the core technique — the model is taught
  its own boundary, not an arbitrary one.)*
- **FR-4** Held-out eval sets, disjoint from training, for each class.

### 4.2 Training
- **FR-5** LoRA SFT on a 1.5–3B instruct base, free Kaggle/Colab T4.
- **FR-6** One committed config + seed per run; loss tracked on a val split.
- **FR-7** At least three runs to produce a tradeoff curve — e.g. varying the
  abstain:answer ratio in the training mix (25/75, 50/50, 75/25).

### 4.3 Evaluation (the deliverable)
- **FR-8** Report, always together:
  - **Hallucination rate** — confident wrong answers on unanswerable items.
  - **Over-refusal rate** — abstentions on items the base answered correctly.
  - **Accuracy on answerable items** — must not collapse.
  - **Abstention quality** — does the refusal name a valid reason?
- **FR-9** Report a **calibration metric**: bucket by model confidence and
  measure accuracy per bucket (ECE or a reliability diagram).
- **FR-10** Publish the **tradeoff curve** across runs, not just the best point.
- **FR-11** Compare against two cheap baselines: (a) base model, (b) base model
  with a "say I don't know if unsure" system prompt. *If prompting matches the
  fine-tune, say so plainly.*

### 4.4 Delivery
- **FR-12** Hugging Face model card: recipe, eval tables, limitations, license.
- **FR-13** GGUF export + Ollama Modelfile so it's usable in local agents.
- **FR-14** Write-up covering falsified hypotheses, in the style of the
  previous fine-tuning case study.

## 5. Non-functional requirements

- **NFR-1** Free hardware only (T4-class, runs < 4h each).
- **NFR-2** Fully reproducible: data-gen scripts, configs, seeds committed.
- **NFR-3** No paid API required; if a teacher model is used, it must be a free
  tier (NVIDIA NIM), and that dependency documented.
- **NFR-4** Eval frozen before training begins; no eval edits afterward.

## 6. Success metrics

| metric | target |
|---|---|
| Hallucination rate on unanswerable set | ≥ 40% relative reduction vs base |
| Over-refusal rate on answerable set | ≤ 5 pts increase vs base |
| Accuracy on answerable set | within 3 pts of base |
| Calibration (ECE) | improved vs base |
| Beats the prompt-only baseline | on at least hallucination rate |

Ship tiers:

- **Minimum:** one fine-tune that reduces hallucination with over-refusal
  measured and reported, even if the tradeoff is poor.
- **Good:** tradeoff curve across ≥ 3 mixes; beats the prompt-only baseline.
- **Headline:** ≥ 40% hallucination reduction at ≤ 5 pts over-refusal, improved
  calibration, model published and usable in a local agent.

## 7. Milestones

| week | deliverable | exit gate |
|---|---|---|
| 1 | Eval sets frozen; base + prompt-baseline measured on all four metrics | baseline table committed; eval tagged and untouched thereafter |
| 2 | Self-knowledge labeling run (k-sample probing) → answerable/abstain split | labeled corpus with per-question correctness distribution |
| 3 | Training set assembled in 3 mixes with informative abstention targets | data stats + rejection rates documented |
| 4 | Three LoRA runs, merged, exported | all runs produce loadable adapters; smoke-tested |
| 5 | Full eval, tradeoff curve, calibration analysis, model card, write-up | honest verdict published, including failures |

## 8. Risks

| risk | mitigation |
|---|---|
| Model learns to abstain on everything | Over-refusal is a first-class metric from week 1; mixes are varied deliberately to find the knee |
| Prompting alone works as well | Measured explicitly as a baseline — if true, that *is* the finding and gets published |
| Benchmark contamination on public QA sets | Report deltas vs the same base model, which cancels contamination |
| "Knows the answer" is fuzzy | k-sample probing with an explicit threshold, documented and sensitivity-checked |
| Free GPU limits | 1.5–3B + LoRA fits; cap runs at 2 epochs |

## 9. Open questions — resolved at implementation

- **Base model:** reuse Qwen2.5-1.5B-Instruct from the previous fine-tune.
  Comparability against a known-characterised base beats a newer model's
  headline numbers, and the whole eval reports deltas vs base anyway.
- **Structured abstention:** yes — `{"answer": ..., "reason": ...}`, defined
  once in `refusal.py` and shared by training, probing and eval. It makes
  abstention quality mechanically gradeable and drops straight into a planner.
  The parser also accepts prose refusals so the un-finetuned baselines aren't
  scored as hallucinating when they decline in English.
- **Domain:** general QA (TriviaQA `rc.nocontext`) for v1. The agent-planning
  slice is deferred — the structured output format is the part the planner
  needs, and it exists now; a planning-domain mix is v2 if the curve holds.
- **Confidence signal for ECE:** self-consistency across k samples, not
  logprobs — identical code path to the FR-3 probe, and works unchanged for
  base, prompt baseline and fine-tunes.

## 10. Implementation status

Scaffolding, data pipeline, metrics and eval harness are built and tested
(`python tests.py`, plus an end-to-end run on synthetic probe data). What
remains is the three GPU steps (probe → train → generate) and the write-up.

## 11. Stack

Python · Unsloth + PEFT (LoRA) · Qwen2.5-1.5B/3B-Instruct class base ·
Kaggle T4 · NVIDIA NIM free tier (optional teacher) · Hugging Face Hub ·
llama.cpp/GGUF → Ollama · pytest.
