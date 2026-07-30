"""Crash containment and resume, kept out of the notebook so they can be tested.

Two rules the whole pipeline relies on:

  1. **A failing stage must not destroy the finished ones.** Every stage writes
     to disk as it goes; `stage` swallows the exception, prints the traceback,
     and lets the rest of the notebook (including the zip-it-up cell) run.
  2. **A half-written output must never look finished.** `Resumable` appends to
     `<name>.partial` and only renames to the real path once the last row is in,
     so a killed session resumes instead of silently scoring fewer items.
"""
import os
import time
import traceback


class stage:
    """Context manager: contain a failure to its own stage.

        with stage("train"):
            ...                 # anything here can raise; the notebook goes on

    Records failures in `stage.failed`. KeyboardInterrupt is re-raised — a
    deliberate stop should stop.
    """

    failed = []

    def __init__(self, name, log=print):
        self.name = name
        self.log = log

    def __enter__(self):
        self.t0 = time.time()
        self.log(f"--- {self.name} ---")
        return self

    def __exit__(self, exc_type, exc, tb):
        mins = (time.time() - self.t0) / 60
        if exc is None:
            self.log(f"--- {self.name}: ok ({mins:.1f} min) ---")
            return False
        if exc_type is KeyboardInterrupt:
            return False
        stage.failed.append(self.name)
        self.log(traceback.format_exc())   # via log so tests can silence it
        self.log(f"\n!!! {self.name} FAILED after {mins:.1f} min — everything it finished "
                 f"is still on disk. Continuing.\n"
                 f"    See RUNBOOK.md; rerun the notebook and finished work is skipped.\n")
        return True


class Resumable:
    """Append-as-you-go writer with an atomic finish.

        w = Resumable("runs/base/responses.jsonl")
        if w.complete: ...                     # nothing to do
        for row in items[w.done:]:             # resume where the last run died
            w.write(json.dumps(row))
        w.finish()

    `done` is the number of rows already written by a previous attempt.
    """

    def __init__(self, final_path):
        self.final = final_path
        self.partial = os.path.splitext(final_path)[0] + ".partial"
        self.complete = os.path.exists(final_path)
        self.done = 0
        if not self.complete and os.path.exists(self.partial):
            with open(self.partial, encoding="utf-8") as f:
                self.done = sum(1 for _ in f)
        self._f = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.final) or ".", exist_ok=True)
        self._f = open(self.partial, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._f:
            self._f.close()
            self._f = None
        return False

    def write(self, line):
        self._f.write(line.rstrip("\n") + "\n")

    def flush(self):
        self._f.flush()
        os.fsync(self._f.fileno())   # a killed VM shouldn't take the page cache with it

    def finish(self):
        """Rename partial -> final. Complete or absent, never half."""
        if self._f:
            self.flush()
            self._f.close()
            self._f = None
        os.replace(self.partial, self.final)
        self.complete = True
