# Runbook

Everything you need to run this without me. Order matters; each stage checks
the previous one. **GPU quota is the scarce resource** — `python preflight.py`
before every upload, it costs two seconds and saves whole sessions.

**Three session notebooks run the whole thing across Kaggle's 9h limit** —
`session1_data_train.ipynb`, `session2_train.ipynb`, `session3_generate_score.ipynb`,
in order. They share one tested module (`stages.py`). The sections below are for
when you want to re-score locally, choose which runs to spend quota on, or
something breaks.

Nothing here needs a paid API. Only three stages need the GPU (probe, train,
generate); everything else — every metric, every table — runs on your laptop
from the saved generations.

---

## 0. What's already done

- `data/questions.jsonl` — 8,000 TriviaQA questions + gold aliases, committed.
- `configs/*.yaml` — 11 runs, generated from the table in `make_configs.py`.
- `tests.py` passes; the whole pipeline has been smoke-tested end to end on
  synthetic probe data, so the plumbing is known good before you spend quota.

Re-fetch a different pool size with `python -m data.fetch 20000`.
Change or add runs by editing the `RUNS` table in `make_configs.py`, then
`python make_configs.py`.

---

## 1. Run it — three session notebooks

Full scale (14 runs + ~16 eval arms) does **not** fit one 9h Kaggle session, so
it's split into three notebooks you run in order. All three share one tested
module (`stages.py`), so they can't drift apart:

| notebook | does | ~time |
|---|---|---|
| `session1_data_train.ipynb` | probe → build the frozen eval + mixes → train the curve + seed replicates | ~8h |
| `session2_train.ipynb` | train the rest (dosage, capacity/LR, ablation, 3B) | ~8h |
| `session3_generate_score.ipynb` | generate every arm on the frozen eval → score | ~4h |

Each session:

1. Zip this repo, upload as a Kaggle **Dataset** named `refusal-calibration`
   (once; reuse it for all three).
2. New Notebook → Accelerator **GPU T4 x2**, Internet **on**, Add Data → that dataset.
3. Open the session's notebook, run the pip cell first (one expected kernel
   restart), then **Run All**.
4. At the end, **Save Version → Save & Run All** so `/kaggle/working` persists to
   the next session — or download `results.zip`/`probe.zip`/`adapters.zip` and
   re-upload them into the dataset. Either way, the next session resumes.

**Nothing finished is ever lost.** Each stage writes to disk as it runs and is
wrapped (`stages.py` → `runner.stage`) so a failure — OOM, bad config, dead
kernel — is contained to that stage; the rest of the notebook, including the zip
cell, still runs. On rerun, finished work is skipped: the probe skips done
questions, training skips runs that already have an adapter, generation resumes
an arm from the exact item it died on (`responses.partial` → atomic rename to
`responses.jsonl` only when complete). So "Run All" is always the correct button,
however many times it takes. This is covered by `tests.py`.

Session 1 is the only one that creates `data/eval.jsonl` — **keep it**, every
later session depends on it. Rehearse first if unsure: in Session 1 pass
`Ctx(n_questions=3000)` and hand `train` just `["v3_mix50"]` to run the whole
chain small in ~2h, then rerun full — nothing already done gets redone.

**Freeze the eval when you get it back.** `eval.lock` is what lets you claim the
eval was never touched after training, and `eval.py` enforces it:

```bash
git add data/eval.jsonl data/eval.lock data/mix_*.jsonl && git commit -m "freeze eval"
```

A healthy class split is roughly 30-60% answerable, 20-40% unknown. If it's 95%
either way, that's the *prompt or the thresholds*, not the model — see
Troubleshooting.

### Run order (priority) — 14 runs

Stop after any block and you still have a coherent result:

| runs | what you can claim |
|---|---|
| `v3_mix50` | one fine-tune, both error directions measured (PRD "Minimum" tier) |
| `+ v2, v4` | a real tradeoff curve (PRD "Good" tier) |
| `+ v1, v5` | curve with both anchors — the knee is interpretable |
| `+ v12, v13` | **noise floor** — 3 seeds of the reference. Do this before believing any gap on the curve; a difference smaller than the seed spread isn't real |
| `+ v6, v7` | dosage: is over-refusal just more training? |
| `+ v8, v9, v10` | capacity/LR: is it the data mix or the optimiser? |
| `+ v11` | ablation: was naming the refusal reason worth anything? |
| `+ v14` | scale: does the recipe hold at 3B? (auto-gets its own `base_3b`/`prompt_3b` eval arms) |

The scoring cell prints the seed spread explicitly — treat it as the smallest
gap worth interpreting.

## 2. Re-score locally — CPU, seconds, no GPU ever

The notebook already prints every number, but the generations are saved, so all
of it reruns on your laptop — and this is where you'll actually write it up:

```bash
python eval.py --run runs/base/responses.jsonl --name base --reliability
python eval.py --run runs/prompt/responses.jsonl   --base runs/base/responses.jsonl --name prompt   --compare
python eval.py --run runs/v3_mix50/responses.jsonl --base runs/base/responses.jsonl --name v3_mix50 --compare
python curve.py          # the deliverable: every arm, both axes, with intervals
```

`--compare` prints paired bootstrap deltas; `*` marks a 95% CI that excludes
zero. **A delta without a `*` is not a result** — say so in the write-up.

---

## Do I need to distil data?

Not for v1. This project's "distillation" is the self-knowledge probe (stage 1):
the model's own sampled answers are the label source, which is free and needs no
teacher. Reach for a teacher model only for these, in this order:

1. **Not enough `unknown` items.** Cheapest fix first: probe more questions
   (`python -m data.fetch 20000`, rerun stage 1). Only if that fails, generate
   obscure questions with a free-tier teacher (NVIDIA NIM) — and document the
   dependency, per NFR-3.
2. **Abstention wording is monotonous.** Optional: have a teacher paraphrase
   the reason strings. Skip it unless the eval shows the model pattern-matching
   the template rather than the defect — the reason field is a fixed enum
   anyway, so this affects style, not scoring.
3. **A harder answerable set.** Only if accuracy is near-ceiling and the
   over-refusal metric has nothing to bite on.

If you do add a teacher pass, write it as a new `data/*.py` that emits the same
`{question, aliases}` rows and feed it back through stage 1. Nothing downstream
changes — that's the point of the shared contract in `refusal.py`.

---

## Troubleshooting

### Training

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` during **training** | batch too big for a T4 | `batch_size: 1`, `grad_accum: 16` in the config (same effective batch), or `max_seq_len: 768` |
| `CUDA out of memory` during **probe** | concurrent seqs = `probe_batch_size × k_probe` too high | `Ctx(probe_batch_size=4)`. Default is 8 (×16 = 128 seqs); halve it. Probe writes per batch, so a crash loses nothing — just rerun |
| `CUDA out of memory` during **generate** (esp. the 3B `v14` arm) | `eval_batch_size × k_eval` too high | `Ctx(eval_batch_size=4)`. The failed arm resumes; other arms already done are skipped |
| `unsloth` import error / xformers ABI mismatch | Kaggle image drifted | `!pip install -q "unsloth[kaggle-new]"` or pin `!pip install -q unsloth==2024.11 trl==0.12.1`; restart the kernel after installing |
| Loss is `nan` after a few steps | LR too high for fp16 | use `configs/v10_lr1e4.yaml` (lr 1e-4); if it persists, `grad_accum: 16` |
| Loss drops to ~0 almost immediately | targets are near-identical short JSON — expected on the 90/10 mix | not a bug; check the val loss curve and the stage-3 smoke test output instead |
| Val loss rises while train loss falls | overfitting | 1 epoch (`v6_1epoch`) or lower LR (`v10_lr1e4`) |
| Session died mid-loop | Kaggle 9h/session, 30h/week | rerun the notebook — completed runs are skipped, so you resume where it stopped |
| `FileNotFoundError: data/mix_*.jsonl` | Session 1's build didn't run/persist | rerun Session 1's `build(ctx)` cell; carry `data/` into later sessions (persist `/kaggle/working` or re-upload `results.zip`) |
| Kernel wants to restart after `pip install` | pip cell ran after other imports | it's the first cell for exactly this reason — restart once, then Run All; every finished stage is skipped |
| A run fails, the loop keeps going | intentional — `stage()` contains it | read the traceback in the log; one bad config shouldn't cost you the session. The end-of-notebook summary lists what failed |
| `!!! <stage> FAILED ... Continuing` | a stage raised and was contained | every stage before it is still on disk. Fix the cause, rerun — finished work is skipped |
| `data ready: False`, training skipped | no frozen eval (build failed or wasn't run) | the one hard gate — training unscoreable adapters is worse than not training. Fix build first |

### Adapters / loading

| symptom | cause | fix |
|---|---|---|
| Adapter loads but output is gibberish | loaded the adapter dir as a full model | always `AutoModelForCausalLM.from_pretrained(BASE)` then `PeftModel.from_pretrained(base, adapter)` — the notebooks do this |
| `bitsandbytes` / 4-bit error on CPU | the adapter records the 4-bit base | load the full-precision base `Qwen/Qwen2.5-1.5B-Instruct`; LoRA deltas apply identically (Lean hit this exact wall) |
| `ValueError: Can't find tokenizer` in the adapter dir | tokenizer wasn't saved | `train.py` saves it; for older runs pass the base model id to `AutoTokenizer` instead |

### Generation / eval

| symptom | cause | fix |
|---|---|---|
| Responses are empty or truncated mid-JSON | `max_new_tokens` too small | raise to 64; anything longer than that is the model rambling, which is itself a finding |
| Base model never emits our JSON | expected — it was never trained to | the parser accepts prose refusals; don't "fix" this, it's what makes the baseline fair |
| `eval set changed since it was frozen` | `data/eval.jsonl` was edited after locking | do **not** re-lock. Restore the committed eval (`git checkout data/eval.jsonl`). If it genuinely must change, freeze a *new* set and rerun every arm |
| `warning: N eval items have no response` | generation stopped early | rerun the arm; it resumes from `responses.partial` at the exact item it reached |
| A `responses.partial` left behind | that arm was killed mid-write | not an error — rerun and it continues. It's never scored (only the renamed `.jsonl` is), so it can't corrupt results |
| 3B arm scored against the 1.5B base | comparing across model families | it isn't — `meta.json` records each arm's `base_arm`; v14 auto-gets `base_3b`/`prompt_3b`. Check `meta.json` if unsure |
| `abstention_quality n/a` | the model never abstained | that's the confabulator end of the curve — report it, don't patch it |
| Every metric identical across mixes | all arms read the same responses file | check the `--run` paths; each arm needs its own `runs/<arm>/responses.jsonl` |
| `KeyError` on a question in `eval.py` | responses generated against a different eval set | regenerate responses with the committed `data/eval.jsonl` |

### Data / labeling

| symptom | cause | fix |
|---|---|---|
| Almost everything labeled `unknown` | samples aren't parsing, so nothing scores correct | print a raw sample (last probe cell); if it's prose, the chat template or `SYSTEM_PROMPT` isn't being applied |
| Almost everything labeled `answerable` | question pool too easy | fetch more (`python -m data.fetch 20000`) — TriviaQA gets obscure fast — or raise `KNOWN` in `data/probe.py` |
| `borderline` swallows most items | k too small, so p_correct is coarse | k=8 gives 1/8 resolution; raise k to 16, or widen the thresholds and *say so* in the write-up |
| Too few training rows in a mix | the limiting class caps the mix size | build a bigger pool; the extreme mixes (10/90, 90/10) are always the smallest |
| Defective questions read unnaturally | templated by design | that's the documented limitation; the mitigation is the disjoint train/eval template split, asserted in `tests.py` |

### When something is wrong and you can't tell what

```bash
python tests.py       # is the logic intact?
python preflight.py   # is the data intact, and which stage am I at?
```

If both pass and results still look wrong, the bug is in the *interpretation*,
not the code: check that the arm you're comparing against is the base arm, and
that `--compare` shows a `*`. Lean's most useful lesson was that the surprising
result was real and the tidy explanation was wrong — check the number twice
before rewriting the story around it.
