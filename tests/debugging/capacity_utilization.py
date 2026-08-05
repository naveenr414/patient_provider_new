"""Capacity distribution, leftover capacity per policy, and a hard check that
no provider is ever matched beyond its capacity.

Capacities are Poisson(avg_capacity) draws (`run_experiments.build_instance`),
so the interesting facts are how much mass sits at 0 (providers that can never
be matched) and how much of the total capacity each policy actually fills.

Run:  PYTHONPATH=. python tests/debugging/capacity_utilization.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import policies as P
from patient.simulator import run_trials
from scripts.run_experiments import (
    build_instance, DEFAULT_INSTANCE, DEFAULT_N, DEFAULT_M, DEFAULT_K,
    DEFAULT_EPSILON, DEFAULT_S,
)

SEEDS = [0, 1, 2]
NUM_TRIALS = 5
POLICIES = {
    "random": (P.random_menu, {}),
    "offer_one": (P.offer_one, {}),
    "offer_all": (P.offer_all, {}),
    "sam": (P.sam, dict(S=DEFAULT_S)),
    "deferred_acceptance": (P.deferred_acceptance, {}),
    "capacity_greedy": (P.capacity_greedy, {}),
}


def main():
    print(f"N={DEFAULT_N} M={DEFAULT_M} k={DEFAULT_K} epsilon={DEFAULT_EPSILON} "
          f"avg_capacity={DEFAULT_INSTANCE['avg_capacity']} | seeds={SEEDS} "
          f"x {NUM_TRIALS} trials\n")

    all_caps, rows, violations = [], {p: [] for p in POLICIES}, []
    for seed in SEEDS:
        theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                                   **DEFAULT_INSTANCE)
        all_caps.append(capacities)

        for pname, (fn, kwargs) in POLICIES.items():
            out = run_trials(theta_hat, capacities, fn, DEFAULT_K,
                             num_trials=NUM_TRIALS, epsilon=DEFAULT_EPSILON,
                             seed=seed, **kwargs)
            chosen, menus = out["chosen"], out["menus"]
            for trial in range(NUM_TRIALS):
                # matches per provider this trial (index M == exit, dropped)
                matched = np.bincount(chosen[trial], minlength=DEFAULT_M + 1)[:DEFAULT_M]
                over = matched - capacities
                if over.max() > 0:
                    violations.append((pname, seed, trial, int(over.max()),
                                       int((over > 0).sum())))
                rows[pname].append(dict(
                    seed=seed, trial=trial,
                    total_cap=int(capacities.sum()),
                    filled=int(matched.sum()),
                    leftover_slots=int((capacities - matched).sum()),
                    # providers with capacity that ended with a free slot
                    prov_with_cap=int((capacities > 0).sum()),
                    prov_unused=int(((capacities > 0) & (matched == 0)).sum()),
                    prov_partly_free=int(((capacities - matched) > 0).sum()),
                    prov_full=int(((capacities > 0) & (matched == capacities)).sum()),
                    # was the provider even reachable, i.e. on anyone's menu?
                    prov_never_offered=int(((capacities > 0) &
                                            (menus.sum(axis=0) == 0)).sum()),
                    exits=int((chosen[trial] == DEFAULT_M).sum()),
                ))
        print(f"  seed {seed} done")

    # ------------------------------------------------------------- 1) capacities
    caps = np.concatenate(all_caps)
    print("\n" + "=" * 70)
    print(f"1. CAPACITY DISTRIBUTION  (Poisson({DEFAULT_INSTANCE['avg_capacity']}), "
          f"{len(SEEDS)} seeds x {DEFAULT_M} providers = {len(caps)} draws)")
    print("=" * 70)
    hist = Counter(caps.tolist())
    print(f"{'c_j':>4} {'count':>7} {'share':>8} {'slots':>7}")
    for c in sorted(hist):
        print(f"{c:>4} {hist[c]:>7} {100 * hist[c] / len(caps):>7.2f}% "
              f"{c * hist[c]:>7}")
    print(f"{'all':>4} {len(caps):>7} {100.0:>7.2f}% {caps.sum():>7}")
    print(f"\nmean c_j = {caps.mean():.4f}, total slots/seed = {caps.sum() / len(SEEDS):.1f}, "
          f"N/slots = {DEFAULT_N / (caps.sum() / len(SEEDS)):.3f}")
    print(f"providers with c_j = 0: {100 * (caps == 0).mean():.2f}% "
          f"-- they can never be matched, so the effective provider pool is "
          f"{(caps > 0).sum() / len(SEEDS):.0f} of {DEFAULT_M}")
    print(f"upper bound on match rate = total slots / N = "
          f"{(caps.sum() / len(SEEDS)) / DEFAULT_N:.4f}")

    # --------------------------------------------------------------- 2) leftover
    print("\n" + "=" * 70)
    print("2. LEFTOVER CAPACITY AFTER EACH POLICY  (mean over seeds x trials)")
    print("=" * 70)
    print(f"{'policy':20s} {'filled':>7s} {'left':>6s} {'fill%':>6s} | "
          f"{'full':>6s} {'part':>6s} {'unused':>7s} {'never off.':>10s} | {'exits':>6s}")
    for pname, recs in rows.items():
        a = {k: np.mean([r[k] for r in recs]) for k in recs[0] if k not in ("seed", "trial")}
        print(f"{pname:20s} {a['filled']:7.1f} {a['leftover_slots']:6.1f} "
              f"{100 * a['filled'] / a['total_cap']:5.1f}% | "
              f"{a['prov_full']:6.1f} {a['prov_partly_free']:6.1f} "
              f"{a['prov_unused']:7.1f} {a['prov_never_offered']:10.1f} | "
              f"{a['exits']:6.1f}")
    print(f"\n(of {np.mean([r['prov_with_cap'] for r in rows['sam']]):.1f} providers with "
          f"c_j > 0 and {np.mean([r['total_cap'] for r in rows['sam']]):.1f} total slots; "
          f"'full' = every slot used, 'part' = >=1 slot free, 'unused' = zero matches, "
          f"'never off.' = absent from every menu)")

    # ------------------------------------------------------------- 3) feasibility
    print("\n" + "=" * 70)
    print("3. CAPACITY FEASIBILITY CHECK")
    print("=" * 70)
    n_checks = len(SEEDS) * NUM_TRIALS * len(POLICIES)
    if violations:
        print(f"FAIL: {len(violations)} of {n_checks} (policy, seed, trial) runs "
              f"matched some provider beyond capacity")
        for v in violations[:20]:
            print(f"   {v[0]} seed={v[1]} trial={v[2]} max_overage={v[3]} "
                  f"providers_over={v[4]}")
    else:
        print(f"PASS: across all {n_checks} (policy, seed, trial) runs "
              f"({len(POLICIES)} policies x {len(SEEDS)} seeds x {NUM_TRIALS} trials), "
              f"no provider was matched more times than its capacity.")
        worst = max(max(r["filled"] - r["total_cap"] for r in recs)
                    for recs in rows.values())
        print(f"      max(total matches - total capacity) = {worst} (<= 0 required)")


if __name__ == "__main__":
    main()
