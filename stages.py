"""The pipeline's stages as callable functions, so the three session notebooks
stay thin and can't drift apart (that copy-paste drift is exactly the Lean weak
spot this project set out to fix).

Notebook usage is three lines:

    from stages import Ctx, probe, build, train, generate, score
    ctx = Ctx()                 # sets cwd, sys.path, prints GPU
    probe(ctx); build(ctx); train(ctx, ctx.cfg["runs"])

Heavy imports (torch/transformers/peft) are done inside the functions that need
them, so `score()` — and this whole module's import — works on a laptop with no
GPU stack, which is what `tests.py` and `nbrun` exercise.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

from runner import Resumable, stage

DEFAULTS = {
    "repo_src": "/kaggle/input/refusal-calibration",
    "repo": "/kaggle/working/repo",
    "probe_base": "Qwen/Qwen2.5-1.5B-Instruct",
    # Sized so probe (~1.5h) + 7 training runs fit one 9h T4 session. 12k
    # questions × 8 samples still yields thousands of training examples, a full
    # 800-item eval, and a healthy unknown class. Scale up only on a faster GPU.
    "n_questions": 12000,
    "k_probe": 8,
    "temperature": 0.7,
    # concurrent sequences = batch_size × k. A T4 (16GB) holds the fp16 1.5B
    # model plus KV cache; keep batch×k around 128 or it OOMs. Lower to 8 if 16
    # overflows (e.g. the 3B model at generate time).
    "probe_batch_size": 16,
    "max_new_tokens": 64,
    "k_eval": 8,
    "eval_batch_size": 16,
    "prompt_baseline": (
        'Answer the question. If you are not sure, say "I don\'t know" and '
        "briefly say why. Never guess."
    ),
    "runs": ["v3_mix50", "v2_mix25", "v4_mix75", "v1_mix10", "v5_mix90",
             "v12_seed1", "v13_seed2", "v6_1epoch", "v7_3epoch",
             "v8_rank8", "v9_rank32", "v10_lr1e4", "v11_bare_reason",
             "v14_qwen3b"],
}


def _built(path="data/eval.jsonl"):
    """Present AND non-empty. A crashed build can leave a 0-byte eval.jsonl;
    treating that as "done" would skip the probe forever, so emptiness = not
    built."""
    return os.path.exists(path) and os.path.getsize(path) > 0


class Ctx:
    """Shared session state: copies the repo into the writable dir, fixes cwd
    and sys.path, exposes config. `torch` is looked up lazily so importing this
    module needs no GPU stack."""

    def __init__(self, **overrides):
        self.cfg = {**DEFAULTS, **overrides}
        if os.path.isdir(self.cfg["repo_src"]) and not os.path.exists(self.cfg["repo"]):
            shutil.copytree(self.cfg["repo_src"], self.cfg["repo"])
        if os.path.isdir(self.cfg["repo"]):
            os.chdir(self.cfg["repo"])
        for p in (os.getcwd(), os.path.join(os.getcwd(), "data")):
            if p not in sys.path:
                sys.path.insert(0, p)
        stage.failed = []
        try:
            import torch
            print(torch.cuda.get_device_name(0))
        except Exception:
            print("no GPU (CPU-only stage)")

    def _torch(self):
        import torch
        return torch

    def free_gpu(self, *objs):
        torch = self._torch()
        for o in objs:
            del o
        torch.cuda.empty_cache()

    @property
    def ready(self):
        return _built() and "build" not in stage.failed


# --------------------------------------------------------------------------- #
def probe(ctx):
    """Sample the base model k times per question. Appends per batch, so a crash
    costs at most one batch. Skipped if the eval is already built."""
    with stage("probe"):
        out_path = "data/probe_raw.jsonl"
        if _built():
            print("eval already built — skipping probe"); return
        torch = ctx._torch()
        from refusal import SYSTEM_PROMPT
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = ctx.cfg
        tok = AutoTokenizer.from_pretrained(cfg["probe_base"])
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["probe_base"], torch_dtype=torch.float16, device_map="cuda")
        model.eval()

        rows = [json.loads(l) for l in open("data/questions.jsonl", encoding="utf-8")][:cfg["n_questions"]]
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(l)["question"] for l in open(out_path, encoding="utf-8")}
        rows = [r for r in rows if r["question"] not in done]
        print(len(rows), "to probe;", len(done), "done", flush=True)

        def prompt_for(q):
            return tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True)

        bs, t0 = cfg["probe_batch_size"], time.time()
        try:
            with open(out_path, "a", encoding="utf-8") as f, torch.no_grad():
                for start in range(0, len(rows), bs):
                    batch = rows[start:start + bs]
                    enc = tok([prompt_for(r["question"]) for r in batch],
                              return_tensors="pt", padding=True).to("cuda")
                    gen = model.generate(
                        **enc, do_sample=True, temperature=cfg["temperature"], top_p=0.95,
                        num_return_sequences=cfg["k_probe"], max_new_tokens=cfg["max_new_tokens"],
                        pad_token_id=tok.pad_token_id)
                    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    for i, r in enumerate(batch):
                        f.write(json.dumps(
                            {**r, "samples": texts[i * cfg["k_probe"]:(i + 1) * cfg["k_probe"]]}) + "\n")
                    f.flush()
                    if (start // bs) % 20 == 0:
                        mins = (time.time() - t0) / 60
                        rate = (start + len(batch)) / max(mins, 0.1)
                        print(f"{start + len(batch)}/{len(rows)}  {mins:.0f} min  "
                              f"eta {(len(rows) - start) / rate:.0f} min", flush=True)
        finally:
            ctx.free_gpu(model)
        print("sample:", repr(json.loads(open(out_path, encoding="utf-8").readline())["samples"][0]))


def build(ctx):
    """Label, freeze the eval, build the mixes, then preflight-gate. CPU only."""
    with stage("build"):
        if not _built():
            assert os.path.getsize("data/probe_raw.jsonl") > 0 if os.path.exists("data/probe_raw.jsonl") else False, \
                "data/probe_raw.jsonl is empty — the probe produced nothing (it OOM'd or was never run). " \
                "Rerun probe first; nothing to build labels from."
            subprocess.run([sys.executable, "-m", "data.probe"], check=True)
            subprocess.run([sys.executable, "-m", "data.build"], check=True)
        subprocess.run([sys.executable, "preflight.py"], check=True)
    print("data ready:", ctx.ready)


def train(ctx, runs):
    """One subprocess per run: an OOM or bad config kills only that run, and
    every adapter already on disk stays. Skips runs that already trained."""
    if not ctx.ready:
        print("no frozen eval — skipping train"); return [], []
    trained, broken = [], []
    for name in runs:
        if os.path.exists(f"runs/{name}/adapter/adapter_config.json"):
            print("skip (done):", name); trained.append(name); continue
        print("=" * 60, name, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "train.py", "--config", f"configs/{name}.yaml"])
        ok = r.returncode == 0 and os.path.exists(f"runs/{name}/adapter/adapter_config.json")
        (trained if ok else broken).append(name)
        print(f"{'done' if ok else 'FAILED'} {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\ntrained: {trained}")
    if broken:
        print(f"failed (see RUNBOOK.md): {broken}")
    return trained, broken


def _arm_table(ctx):
    """name -> (inference base, adapter dir or None, system prompt, base arm).
    A run trained on a different base gets its own base/prompt baselines, since
    over-refusal is only meaningful within a model family."""
    import yaml
    from refusal import SYSTEM_PROMPT, full_precision

    cfg = ctx.cfg
    arms, families = {}, {}
    for adapter in sorted(glob.glob("runs/*/adapter")):
        name = os.path.basename(os.path.dirname(adapter))
        base_id = full_precision(yaml.safe_load(open(f"configs/{name}.yaml"))["base_model"])
        families.setdefault(base_id, "base" if "1.5B" in base_id else f"base_{base_id.split('-')[-2].lower()}")
        arms[name] = (base_id, adapter, SYSTEM_PROMPT, families[base_id])
    families.setdefault(cfg["probe_base"], "base")
    for base_id, base_arm in families.items():
        arms[base_arm] = (base_id, None, SYSTEM_PROMPT, base_arm)
        arms[base_arm.replace("base", "prompt")] = (base_id, None, cfg["prompt_baseline"], base_arm)
    return arms


def generate(ctx):
    """Generate every arm on the frozen eval. Each arm resumes from the item it
    died on; only a complete arm is renamed into place, never scored partial."""
    if not ctx.ready:
        print("no frozen eval — skipping generate"); return
    torch = ctx._torch()
    from probe import consistency
    from refusal import SYSTEM_PROMPT  # noqa: F401 (kept for parity/import check)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = ctx.cfg
    items = [json.loads(l) for l in open("data/eval.jsonl", encoding="utf-8")]
    questions = [i["question"] for i in items]
    print(len(items), "eval items", Counter(i["klass"] for i in items))
    arms = _arm_table(ctx)

    def build_model(base_id, adapter):
        tk = AutoTokenizer.from_pretrained(adapter or base_id)
        tk.padding_side = "left"
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token
        m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="cuda")
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
        m.eval()
        return m, tk

    def gen(model, tk, system, batch, sample, k=1):
        prompts = [tk.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True) for q in batch]
        enc = tk(prompts, return_tensors="pt", padding=True).to("cuda")
        kw = dict(max_new_tokens=cfg["max_new_tokens"], pad_token_id=tk.pad_token_id)
        kw.update(dict(do_sample=True, temperature=cfg["temperature"], top_p=0.95,
                       num_return_sequences=k) if sample else dict(do_sample=False))
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        return tk.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_arm(name, base_id, adapter, system, base_arm):
        w = Resumable(f"runs/{name}/responses.jsonl")
        if w.complete:
            print("skip (done):", name); return
        if w.done:
            print(f"resuming {name} at {w.done}/{len(questions)}", flush=True)
        t0, bs, model = time.time(), cfg["eval_batch_size"], None
        try:
            model, tk = build_model(base_id, adapter)
            with w:
                for s in range(w.done, len(questions), bs):
                    batch = questions[s:s + bs]
                    greedy = gen(model, tk, system, batch, sample=False)
                    sampled = gen(model, tk, system, batch, sample=True, k=cfg["k_eval"])
                    for i, q in enumerate(batch):
                        w.write(json.dumps({
                            "question": q, "response": greedy[i],
                            "confidence": round(consistency(sampled[i * cfg["k_eval"]:(i + 1) * cfg["k_eval"]]), 3),
                        }))
                    w.flush()
                w.finish()
            json.dump({"base_arm": base_arm, "base_model": base_id, "adapter": adapter},
                      open(f"runs/{name}/meta.json", "w"))
            print(f"wrote {w.final}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        finally:
            if model is not None:
                ctx.free_gpu(model)

    for name in sorted(arms, key=lambda n: (arms[n][1] is not None, n)):  # baselines first
        with stage(f"generate:{name}"):
            run_arm(name, *arms[name])


def score(ctx):
    """Every metric, paired deltas vs the matching base, seed noise floor, curve.
    CPU only — reruns identically on a laptop from the saved generations."""
    from curve import base_arm_of
    from eval import load_eval, load_records
    from metrics import METRICS, paired_delta, reliability, summarize

    records, runs = {}, {}
    with stage("load results"):
        frozen = load_eval()
        runs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
                for p in sorted(glob.glob("runs/*/responses.jsonl"))}
        for n, d in runs.items():
            try:
                records[n] = load_records(f"{d}/responses.jsonl", frozen, base_arm_of(d))
            except Exception as e:
                print(f"  ! {n} unreadable ({type(e).__name__}: {e}) — delete it and rerun that arm")

    for name, recs in records.items():
        with stage(f"summarize:{name}"):
            print(summarize(recs, name)); print()

    for name, recs in records.items():
        meta = f"{runs[name]}/meta.json"
        base_name = json.load(open(meta))["base_arm"] if os.path.exists(meta) else "base"
        if name == base_name or base_name not in records:
            continue
        with stage(f"compare:{name}"):
            by_q = {r["question"]: r for r in records[base_name]}
            paired = [(by_q[r["question"]], r) for r in recs if r["question"] in by_q]
            a, b = [p[0] for p in paired], [p[1] for p in paired]
            print(f"== {name} - {base_name}  (n={len(paired)})")
            for label, metric in METRICS.items():
                d, lo, hi = paired_delta(a, b, metric)
                print(f"  Δ {label:22s} {d:+6.1%}  [{lo:+.1%}, {hi:+.1%}]{'' if lo <= 0 <= hi else '  *'}")
            print()

    with stage("noise floor"):
        seeds = [n for n in ("v3_mix50", "v12_seed1", "v13_seed2") if n in records]
        if len(seeds) > 1:
            for label, metric in METRICS.items():
                vals = [v for v in (metric(records[n])[0] for n in seeds) if v is not None]
                if vals:
                    print(f"  {label:22s} " + "  ".join(f"{v:.1%}" for v in vals) +
                          f"   spread={max(vals) - min(vals):.1%}")
        else:
            print("train v12_seed1 and v13_seed2 for a noise floor")

    with stage("curve"):
        subprocess.run([sys.executable, "curve.py"], check=True)
        if "base" in records:
            print("\nreliability (base):")
            for mid, acc, n in reliability(records["base"]):
                print(f"  conf~{mid:.2f}  acc={acc:.1%}  n={n}")
    return records
