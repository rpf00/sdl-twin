"""Stage 2: noise floor on the Fe dilution series (KSCN fixed).

Each level N = 5 on each of two sessions. Reports mean d, CV per level, a linear
fit of d against post-mix [Fe], the minimum detectable concentration difference
and the inter-day shift.

    python experiments/stage2_noise_floor.py --seed 0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import parse_args, write_csv
from sdl import Lab
from sdl.schema import pair_recipe

SERIES = ["A1", "A2", "A3", "A4"]
N = 5


def main():
    args = parse_args(__doc__)
    lab = Lab(seed=args.seed, store=args.store)
    conc = {rid: lab.config.reagents[rid].species["Fe3"] / 2 for rid in SERIES}   # post-mix mol/L

    rows = []
    for session in (0, 1):
        if session:
            lab.new_session()
        for rid in SERIES:
            for rep in range(N):
                res = lab.dispatch(pair_recipe(rid, "B"))
                rows.append({"reagent": rid, "fe_postmix_M": conc[rid], "session": session, "rep": rep,
                             "d_blank": round(res.d_blank, 3), **res.features[-1].to_dict()})

    print(f"{'level':6s}{'[Fe] post-mix':>15s}{'mean d':>9s}{'SD':>7s}{'CV %':>7s}{'day shift':>11s}")
    x, y, pooled_var = [], [], []
    for rid in SERIES:
        d = np.array([r["d_blank"] for r in rows if r["reagent"] == rid])
        d0 = np.array([r["d_blank"] for r in rows if r["reagent"] == rid and r["session"] == 0])
        d1 = np.array([r["d_blank"] for r in rows if r["reagent"] == rid and r["session"] == 1])
        within = np.sqrt((d0.var(ddof=1) + d1.var(ddof=1)) / 2)
        pooled_var.append(within ** 2)
        print(f"{rid:6s}{conc[rid]:15.2e}{d.mean():9.2f}{within:7.2f}{100 * within / d.mean():7.1f}{d1.mean() - d0.mean():+11.2f}")
        x += [conc[rid]] * len(d)
        y += list(d)

    x, y = np.array(x), np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    sd = float(np.sqrt(np.mean(pooled_var)))
    print(f"\nlinear fit: d = {slope:.3g} * [Fe] + {intercept:.2f}   (R^2 = {r2:.4f})")
    print(f"pooled within-day SD of d: {sd:.2f}")
    print(f"minimum detectable [Fe] difference (3 * SD / slope): {3 * sd / slope:.2e} M")
    print("note: d bends over at higher concentration; R^2 well below 1 means the range is too wide for a line")

    write_csv(Path(args.out) / "stage2_series.csv", rows)


if __name__ == "__main__":
    main()
