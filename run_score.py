"""Rescore the downloaded session-3 generations locally. CPU only.

`stages.score()` only prints, so run it with stdout redirected to a file:
    python -u run_score.py > s3out/score.txt 2>&1
"""
from stages import Ctx, score

score(Ctx())
