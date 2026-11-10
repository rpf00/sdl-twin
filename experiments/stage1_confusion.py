"""Stage 1: discrimination and order invariance.

All 12 ordered pairs of A-D x 3 replicates (36 mixes), split across two sessions
("days"). Reports the confusion matrix, the order effect d(X->Y) - d(Y->X) and
the day-1 vs day-2 shift.

    python experiments/stage1_confusion.py --seed 0
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import parse_args, write_csv
from sdl import Lab
from sdl.schema import pair_recipe

EXPECTED = {frozenset("AB"): "target", frozenset("AC"): "distractor"}
CLASSES = ["target", "distractor", "null", "other"]


def main():
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)
    lab = Lab(seed=args.seed, store=args.store)

    # Every ordered pair appears in both sessions (2 + 1 or 1 + 2 replicates); order is shuffled within a session.
    pairs = list(itertools.permutations("ABCD", 2))
    plan = {0: [], 1: []}
    for k, (a, b) in enumerate(pairs):
        for rep in range(3):
            plan[(k + rep) % 2].append((a, b, rep))
    rows = []
    for session in (0, 1):
        if session:
            lab.new_session()
        order = rng.permutation(len(plan[session]))
        for a, b, rep in (plan[session][j] for j in order):
            res = lab.dispatch(pair_recipe(a, b))
            rows.append({"first": a, "second": b, "rep": rep, "session": res.session,
                         "expected": EXPECTED.get(frozenset((a, b)), "null"), "predicted": res.predicted_class,
                         "d_blank": round(res.d_blank, 3), "d_pre": round(res.d_pre, 3),
                         **res.features[-1].to_dict()})

    print("Confusion matrix (rows = expected, columns = predicted)")
    print(f"{'':12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for exp in CLASSES[:3]:
        counts = [sum(r["expected"] == exp and r["predicted"] == c for r in rows) for c in CLASSES]
        print(f"{exp:12s}" + "".join(f"{n:12d}" for n in counts))
    acc = np.mean([r["expected"] == r["predicted"] for r in rows])
    print(f"accuracy: {acc:.3f} over {len(rows)} mixes")

    print("\nOrder effect, d_blank(X->Y) - d_blank(Y->X)   [mean over replicates]")
    d = defaultdict(list)
    for r in rows:
        d[(r["first"], r["second"])].append(r["d_blank"])
    for a, b in itertools.combinations("ABCD", 2):
        fwd, rev = np.mean(d[(a, b)]), np.mean(d[(b, a)])
        sd = np.sqrt((np.var(d[(a, b)], ddof=1) + np.var(d[(b, a)], ddof=1)) / 2)
        print(f"  {a}+{b}: {fwd:7.2f} vs {rev:7.2f}   difference {fwd - rev:+6.2f}   (replicate SD {sd:.2f})")

    print("\nDay-to-day shift in d_blank (session 1 minus session 0)")
    for pair in ("AB", "AC", "AD"):
        by = defaultdict(list)
        for r in rows:
            if frozenset((r["first"], r["second"])) == frozenset(pair):
                by[r["session"]].append(r["d_blank"])
        if by[0] and by[1]:
            print(f"  {pair[0]}+{pair[1]}: {np.mean(by[1]) - np.mean(by[0]):+6.2f}")

    write_csv(Path(args.out) / "stage1_mixes.csv", rows)


if __name__ == "__main__":
    main()
