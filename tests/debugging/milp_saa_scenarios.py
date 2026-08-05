"""How many SAA scenarios does the "exact" MILP need to actually be an upper bound?

Context. `exact_saa_milp` is the denominator of the approximation-ratio panel,
so it is supposed to stand in for the true optimum -- every heuristic must
score at or below it. It did not: heuristics beat it, and the previous session
blamed the 300 s time limit truncating the branch-and-bound. Shrinking the
panel to N=10, M=5, k=2 removed truncation completely (0 solves stopped early,
the whole 9-seed panel finishes in ~2 min), and the violations SHRANK but did
not vanish: at epsilon >= 0.3 individual seeds still have SAM or offer-all
above the MILP.

That residue is not a bug, it is sample-average approximation doing what it
does. The MILP maximizes the average over its OWN S sampled scenarios; it is
then evaluated on 25 independently drawn trials. With S=25 it can fit menu
choices to the particular noise draws it saw, and the more noise there is the
more there is to overfit -- which is exactly where the violations live. The
fix is more scenarios, not more solver time.

This script measures the in-sample/out-of-sample gap directly as a function of
S, so DEFAULT_MILP_S can be set from a number rather than a guess. It reports,
per S: out-of-sample normalized utility, how many (seed, epsilon) cells a
heuristic beats the MILP in, and the solve time.

Run:  PYTHONPATH=. python tests/debugging/milp_saa_scenarios.py
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient import policies as P
from patient.simulator import run_trials
from scripts.run_experiments import build_instance, DEFAULT_INSTANCE

HERE = Path(__file__).resolve().parent

SMALL_N, SMALL_M, SMALL_K = 10, 5, 2
SEEDS = list(range(9))
NUM_TRIALS = 25
# Only the high-noise end: at epsilon <= 0.2 the MILP already dominates, and
# the overfitting being measured grows with the noise it is fitting.
EPSILONS = [0.3, 0.4, 0.5]
S_GRID = [25, 50, 100, 200]
# The heuristics that were observed above the MILP.
RIVALS = {"sam": (P.sam, dict(S=10)), "offer_all": (P.offer_all, {}),
          "offer_one": (P.offer_one, {}), "random": (P.random_menu, {})}


def run_cell(cell):
    epsilon, seed = cell["epsilon"], cell["seed"]
    theta_hat, capacities, _ = build_instance(SMALL_N, SMALL_M, seed=seed, **DEFAULT_INSTANCE)
    out = dict(cell, milp={}, rivals={})

    omni = None
    for name, (fn, kwargs) in RIVALS.items():
        res = run_trials(theta_hat, capacities, fn, SMALL_K, num_trials=NUM_TRIALS,
                         epsilon=epsilon, seed=seed, **kwargs)
        if omni is None:
            omni = MET.omniscient_utility(res["theta_realized"], capacities)
        out["rivals"][name] = MET.utility(res["rewards"]) / omni

    for S in S_GRID:
        res = run_trials(theta_hat, capacities, P.exact_saa_milp, SMALL_K,
                         num_trials=NUM_TRIALS, epsilon=epsilon, seed=seed,
                         S=S, time_limit=1800)
        out["milp"][S] = dict(
            normalized=MET.utility(res["rewards"]) / omni,
            runtime=float(res.get("runtime_sec", np.nan)),
            menu=float(res["menus"].sum(axis=1).mean()),
        )
    out["omniscient"] = float(omni)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    cells = [dict(epsilon=e, seed=s) for e in EPSILONS for s in SEEDS]
    jobs = args.jobs or min(len(cells), 27)
    print(f"N={SMALL_N} M={SMALL_M} k={SMALL_K} | {len(SEEDS)} seeds x {NUM_TRIALS} trials "
          f"| S in {S_GRID}\n{len(cells)} cells on {jobs} processes\n")
    with Pool(jobs) as pool:
        recs = pool.map(run_cell, cells)

    print(f"{'eps':>5s} {'S':>5s} {'MILP norm':>11s} {'SE':>7s} "
          f"{'best rival':>11s} {'ratio':>7s} {'seeds beating MILP':>20s} {'menu':>6s}")
    for e in EPSILONS:
        grp = [r for r in recs if r["epsilon"] == e]
        rival_best = {r["seed"]: max(r["rivals"].values()) for r in grp}
        rival_mean = np.mean(list(rival_best.values()))
        for S in S_GRID:
            vals = np.array([r["milp"][S]["normalized"] for r in grp])
            beat = sum(rival_best[r["seed"]] > r["milp"][S]["normalized"] + 1e-9 for r in grp)
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            print(f"{e:>5.2f} {S:>5d} {vals.mean():>11.4f} {se:>7.4f} "
                  f"{rival_mean:>11.4f} {rival_mean / vals.mean():>7.3f} "
                  f"{beat:>13d}/{len(grp):<6d} "
                  f"{np.mean([r['milp'][S]['menu'] for r in grp]):>6.2f}")
        print()

    print("A ratio at or below 1.000 with 0 seeds beating the MILP is what the "
          "approximation-ratio\npanel needs; read off the smallest S that gets there "
          "and set DEFAULT_MILP_S to it.")
    (HERE / "milp_saa_scenarios.json").write_text(json.dumps(recs, indent=2, default=float))
    print(f"\nwrote {HERE / 'milp_saa_scenarios.json'}")


if __name__ == "__main__":
    main()
