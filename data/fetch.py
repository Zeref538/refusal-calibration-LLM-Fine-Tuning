"""Pull the question pool once, locally, so the GPU notebooks never have to.

Streaming so it doesn't pull the whole TriviaQA parquet set for 8k rows, and
committed as `data/questions.jsonl` — the probe run is then reproducible from
a file in the repo rather than "whatever the hub returned that day".

Usage:
    python -m data.fetch [n]
"""
import json
import sys

DATASET = ("mandarjoshi/trivia_qa", "rc.nocontext")


def main(n=8000, out="data/questions.jsonl"):
    from datasets import load_dataset

    ds = load_dataset(*DATASET, split="train", streaming=True)
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            aliases = list(dict.fromkeys(row["answer"]["aliases"] + [row["answer"]["value"]]))[:12]
            aliases = [a for a in aliases if a and len(a) < 60]
            if not aliases or len(row["question"]) < 15:
                continue
            f.write(json.dumps({"question": row["question"].strip(), "aliases": aliases}) + "\n")
            written += 1
            if written >= n:
                break
    print(f"wrote {written} questions -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
