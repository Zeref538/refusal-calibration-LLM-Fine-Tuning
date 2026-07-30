"""LoRA SFT, one config = one run (FR-5/6). Same script for all three mixes —
the mix ratio lives in the config, not in a copy of this file.

(Lean shipped four near-identical notebooks, one per run, and that's where the
"is this the same recipe?" doubt comes from. One script, three configs.)

Usage:
    python train.py --config configs/mix_50_50.yaml
"""
# Unsloth MUST be imported before trl/transformers/peft — it patches them.
# Importing trl first leaves an unresolved <EOS_TOKEN> sentinel (SFTTrainer then
# errors "eos_token not in vocabulary") and disables the speed/memory patches.
from unsloth import FastLanguageModel  # noqa: E402,F401  (import order is load-bearing)

import argparse

import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from refusal import SYSTEM_PROMPT


def format_row(row, tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": row["prompt"]},
         {"role": "assistant", "content": row["completion"]}],
        tokenize=False,
    )
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["base_model"], max_seq_length=cfg["train"]["max_seq_len"], load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        random_state=cfg["seed"],
    )

    ds = load_dataset("json", data_files=cfg["train"]["data_path"], split="train")
    ds = ds.map(lambda r: format_row(r, tokenizer))
    split = ds.train_test_split(test_size=cfg["train"]["val_fraction"], seed=cfg["seed"])

    # TRL's SFTConfig/SFTTrainer signatures drift between versions (Kaggle ships
    # a bleeding-edge build): `max_seq_length` became `max_length`, `tokenizer`
    # became `processing_class`, and `dataset_text_field` moved from the trainer
    # into the config. Filter every kwarg against the installed signature so one
    # image upgrade can't break all runs again.
    import inspect

    cfg_sig = inspect.signature(SFTConfig.__init__).parameters
    cfg_kwargs = {
        "output_dir": cfg["train"]["output_dir"],
        "num_train_epochs": cfg["train"]["epochs"],
        "per_device_train_batch_size": cfg["train"]["batch_size"],
        "gradient_accumulation_steps": cfg["train"]["grad_accum"],
        "learning_rate": cfg["train"]["lr"],
        "seed": cfg["seed"],
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "epoch",
        "report_to": "none",
        # -1 (default) trains the full epochs; a positive value caps total
        # steps so a run stops early regardless of dataset/epoch size.
        "max_steps": cfg["train"].get("max_steps", -1),
        "dataset_text_field": "text",
        # T4 is Turing (no bf16). Force fp16; newer TRL otherwise defaults to
        # bf16 and hard-fails validation on pre-Ampere GPUs.
        "fp16": True,
        "bf16": False,
        # whichever name this TRL uses for the sequence-length cap
        "max_length": cfg["train"]["max_seq_len"],
        "max_seq_length": cfg["train"]["max_seq_len"],
    }
    args = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_sig})

    tr_sig = inspect.signature(SFTTrainer.__init__).parameters
    tr_kwargs = {"model": model, "train_dataset": split["train"],
                 "eval_dataset": split["test"], "args": args}
    tr_kwargs["processing_class" if "processing_class" in tr_sig else "tokenizer"] = tokenizer
    if "dataset_text_field" in tr_sig and "dataset_text_field" not in cfg_sig:
        tr_kwargs["dataset_text_field"] = "text"  # older TRL took it on the trainer

    trainer = SFTTrainer(**tr_kwargs)
    trainer.train()
    model.save_pretrained(cfg["train"]["output_dir"] + "/adapter")
    tokenizer.save_pretrained(cfg["train"]["output_dir"] + "/adapter")


if __name__ == "__main__":
    main()
