"""Offer literally everything: no menu-size cap (k = M), all 9 seeds.

With k = M every patient is shown every live provider, so the platform is doing
no menu design at all -- patients simply take their best still-available option
in arrival order. Compared against the k=25 policies already in
results/default/, which used the same seeds and the same theta realizations
(they depend only on the seed) and therefore the same omniscient normalizers.

No LPs are solved here: the normalizers are read from the stored results, which
makes the whole thing embarrassingly parallel over seeds.

SEs are clustered at the seed (the project convention): sd over per-seed values
/ sqrt(num_seeds). Because every policy sees identical seeds and identical
trials, the uncapped-vs-SAM comparison is paired and reported as such.

Run:  PYTHONPATH=. python tests/debugging/offer_everything_uncapped.py
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient import policies as P
from patient.simulator import run_trials
from scripts.run_experiments import (
    build_instance, DEFAULT_INSTANCE, DEFAULT_N, DEFAULT_M, DEFAULT_EPSILON,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(9))
NUM_TRIALS = 25          # matches results/default
JOBS = 9


def se(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v))


def run_seed(seed):
    """One seed of uncapped offer-everything. Returns a metrics dict."""
    K_MAX = DEFAULT_M
    theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                               **DEFAULT_INSTANCE)
    out = run_trials(theta_hat, capacities, P.offer_all, K_MAX,
                     num_trials=NUM_TRIALS, epsilon=DEFAULT_EPSILON, seed=seed)
    chosen, rewards, menus = out["chosen"], out["rewards"], out["menus"]
    filled = np.mean([np.bincount(chosen[t], minlength=DEFAULT_M + 1)[:DEFAULT_M].sum()
                      for t in range(NUM_TRIALS)])
    ccount = MET.choice_count(out["effective_menus"])
    return dict(
        seed=seed,
        utility=MET.utility(rewards),
        match_rate=MET.match_rate(chosen, DEFAULT_M),
        choice_count_mean=float(np.mean(ccount)),
        choice_count_median=float(np.median(ccount)),
        menu_planned=float(menus.sum(axis=1).mean()),
        live=int((capacities > 0).sum()),
        total_cap=int(capacities.sum()),
        filled=float(filled),
        exits=float(np.mean([(chosen[t] == DEFAULT_M).sum() for t in range(NUM_TRIALS)])),
    )


def stored_per_seed():
    """{policy: {seed: row}} from results/default, plus the omniscient by seed."""
    by_policy, omni = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/default/*.json"))):
        d = json.load(open(f))
        by_policy[d["policy"]] = {r["seed"]: r for r in d["per_seed"]}
        for r in d["per_seed"]:
            omni[r["seed"]] = r["omniscient_utility"]
    return by_policy, omni


def main():
    print(f"N={DEFAULT_N} M={DEFAULT_M} epsilon={DEFAULT_EPSILON} "
          f"trials={NUM_TRIALS} | k = M = {DEFAULT_M} (uncapped) | "
          f"seeds={SEEDS}\n")

    with Pool(JOBS) as pool:
        recs = pool.map(run_seed, SEEDS)
    recs.sort(key=lambda r: r["seed"])

    by_policy, omni = stored_per_seed()
    for r in recs:
        r["normalized_utility"] = r["utility"] / omni[r["seed"]]

    # ------------------------------------------------------------- per seed
    print("=" * 96)
    print("UNCAPPED (k=M), PER SEED")
    print("=" * 96)
    print(f"{'seed':>4s} {'live':>5s} {'cap':>5s} {'utility':>8s} {'norm.':>7s} "
          f"{'match':>7s} {'#choices':>9s} {'median':>7s} {'fill%':>6s} {'exit%':>6s}")
    for r in recs:
        print(f"{r['seed']:>4d} {r['live']:>5d} {r['total_cap']:>5d} "
              f"{r['utility']:8.4f} {r['normalized_utility']:7.4f} "
              f"{r['match_rate']:7.4f} {r['choice_count_mean']:9.2f} "
              f"{r['choice_count_median']:7.1f} "
              f"{100 * r['filled'] / r['total_cap']:5.1f}% "
              f"{100 * r['exits'] / DEFAULT_N:5.1f}%")

    for key in ("utility", "normalized_utility", "match_rate", "choice_count_mean"):
        v = [r[key] for r in recs]
        print(f"\n{key:22s} mean={np.mean(v):.4f} +/- {se(v):.4f} (SE, 9 seeds)")

    # --------------------------------------------------- vs the k=25 policies
    print("\n" + "=" * 96)
    print("VS THE k=25 POLICIES (same 9 seeds, same theta realizations)")
    print("=" * 96)
    print(f"{'policy':22s} {'k':>4s} {'utility':>17s} {'normalized':>19s} "
          f"{'match':>8s} {'#choices':>9s}")
    u = [r["utility"] for r in recs]
    n = [r["normalized_utility"] for r in recs]
    print(f"{'offer_all (uncapped)':22s} {DEFAULT_M:>4d} "
          f"{np.mean(u):8.4f} +/-{se(u):.4f} {np.mean(n):9.4f} +/-{se(n):.4f} "
          f"{np.mean([r['match_rate'] for r in recs]):8.4f} "
          f"{np.mean([r['choice_count_mean'] for r in recs]):9.2f}")
    order = sorted(by_policy, key=lambda p: -np.mean(
        [by_policy[p][s]["normalized_utility"] for s in SEEDS]))
    for p in order:
        rows = [by_policy[p][s] for s in SEEDS]
        pu = [r["utility"] for r in rows]
        pn = [r["normalized_utility"] for r in rows]
        print(f"{p:22s} {25:>4d} {np.mean(pu):8.4f} +/-{se(pu):.4f} "
              f"{np.mean(pn):9.4f} +/-{se(pn):.4f} "
              f"{np.mean([r['match_rate'] for r in rows]):8.4f} "
              f"{np.mean([r['choice_count_mean'] for r in rows]):9.2f}")

    # ------------------------------------------------------ paired differences
    print("\n" + "=" * 96)
    print("PAIRED DIFFERENCE: uncapped - policy  (per seed, then averaged)")
    print("=" * 96)
    print(f"{'policy':22s} {'d(norm.)':>10s} {'SE':>8s} {'t':>7s} "
          f"{'seeds>0':>8s} {'rel.':>8s}")
    for p in order:
        d = [recs[i]["normalized_utility"] - by_policy[p][s]["normalized_utility"]
             for i, s in enumerate(SEEDS)]
        rel = np.mean([recs[i]["utility"] / by_policy[p][s]["utility"] - 1
                       for i, s in enumerate(SEEDS)])
        print(f"{p:22s} {np.mean(d):10.4f} {se(d):8.4f} "
              f"{np.mean(d) / se(d):7.2f} {sum(x > 0 for x in d):>4d}/9 "
              f"{100 * rel:7.2f}%")

    # what the k=25 cap costs each policy, relative to no cap at all
    print("\ncost of the k=25 cap, as a share of the uncapped level:")
    for p in order:
        rel = np.mean([1 - by_policy[p][s]["utility"] / recs[i]["utility"]
                       for i, s in enumerate(SEEDS)])
        print(f"   {p:22s} {100 * rel:6.2f}%")

    json.dump(recs, open(Path(__file__).resolve().parent /
                          "offer_everything_uncapped.json", "w"), indent=1)
    print(f"\nsaved offer_everything_uncapped.json")


if __name__ == "__main__":
    main()
