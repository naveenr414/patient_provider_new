"""Sanity check: one provider holds ALL the capacity, every other provider has
c_j = 0. Small instance so it runs in seconds.

In that world menu design is vacuous -- there is exactly one thing to offer --
so the checks are mechanical:

  A. every menu is a subset of {j*} (no phantom zero-capacity options)
  B. nobody is ever matched to a provider other than j*
  C. matches never exceed j*'s capacity
  D. with slack capacity, the match rate equals the fraction of patients who
     prefer j* to their exit option under the REALIZED theta -- i.e. the only
     unmatched patients are the ones who genuinely chose to exit
  E. with binding capacity, matches saturate at c_{j*}

Run:  PYTHONPATH=. python tests/debugging/single_provider_sanity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient import policies as P
from patient.simulator import run_trials
from scripts.run_experiments import build_instance, DEFAULT_INSTANCE

N, M, K = 60, 40, 5
EPSILON = 0.25
SEEDS = [0, 1, 2]
NUM_TRIALS = 10
POLICIES = {
    "random": (P.random_menu, {}),
    "offer_one": (P.offer_one, {}),
    "offer_all": (P.offer_all, {}),
    "sam": (P.sam, dict(S=10)),
    "deferred_acceptance": (P.deferred_acceptance, {}),
    "capacity_greedy": (P.capacity_greedy, {}),
}
# c_{j*} = N leaves capacity slack (only preferences bind); N//3 makes it bind
CAPACITY_SETTINGS = {"slack (c=N)": N, "binding (c=N/3)": N // 3}


def main():
    print(f"N={N} M={M} k={K} epsilon={EPSILON} seeds={SEEDS} x {NUM_TRIALS} trials")
    print("one provider holds all capacity; every other provider has c_j = 0\n")

    failures = []
    for label, cap in CAPACITY_SETTINGS.items():
        print("=" * 92)
        print(f"CAPACITY SETTING: {label} -> c_(j*) = {cap}")
        print("=" * 92)
        print(f"{'policy':22s} {'menu':>6s} {'offered':>8s} {'match':>7s} "
              f"{'pred.':>7s} {'matches':>8s} {'utility':>8s} {'norm.':>7s} "
              f"{'exit%':>6s}  checks")

        for seed in SEEDS:
            theta_hat, _caps, _ = build_instance(N, M, seed=seed, **DEFAULT_INSTANCE)
            # the most attractive provider on average gets everything
            jstar = int(theta_hat[:, :M].mean(axis=0).argmax())
            capacities = np.zeros(M, dtype=int)
            capacities[jstar] = cap

            for pname, (fn, kwargs) in POLICIES.items():
                out = run_trials(theta_hat, capacities, fn, K, num_trials=NUM_TRIALS,
                                 epsilon=EPSILON, seed=seed, **kwargs)
                menus, chosen = out["menus"], out["chosen"]
                theta_r = out["theta_realized"]

                # -- A: every menu is a subset of {j*}
                off_jstar = menus.sum() - menus[:, jstar].sum()
                ok_a = off_jstar == 0
                # -- B: nobody matched elsewhere
                ok_b = bool(((chosen != jstar) & (chosen != M)).sum() == 0)
                # -- C: capacity respected every trial
                per_trial = np.array([(chosen[t] == jstar).sum()
                                      for t in range(NUM_TRIALS)])
                ok_c = bool((per_trial <= cap).all())
                # -- D/E: predicted matches = min(cap, #offered patients who
                #         prefer j* to their own realized exit utility)
                offered = menus[:, jstar] == 1
                prefers = (theta_r[:, :, jstar] > theta_r[:, :, M])
                predicted = np.array([min(cap, int((prefers[t] & offered).sum()))
                                      for t in range(NUM_TRIALS)])
                ok_d = bool((per_trial == predicted).all())

                for name, ok in [("A", ok_a), ("B", ok_b), ("C", ok_c), ("D", ok_d)]:
                    if not ok:
                        failures.append((label, seed, pname, name))

                if seed == SEEDS[0]:
                    omni = MET.omniscient_utility(theta_r, capacities)
                    util = MET.utility(out["rewards"])
                    print(f"{pname:22s} {menus.sum(axis=1).mean():6.2f} "
                          f"{100 * offered.mean():7.1f}% "
                          f"{MET.match_rate(chosen, M):7.4f} "
                          f"{predicted.mean() / N:7.4f} "
                          f"{per_trial.mean():8.1f} {util:8.4f} "
                          f"{util / omni:7.4f} "
                          f"{100 * (chosen == M).mean():5.1f}%  "
                          f"{'A' if ok_a else 'a'}{'B' if ok_b else 'b'}"
                          f"{'C' if ok_c else 'c'}{'D' if ok_d else 'd'}")
        print(f"   (rows shown for seed {SEEDS[0]}; checks run on all "
              f"{len(SEEDS)} seeds. 'offered' = share of patients whose menu "
              f"contains j*; 'pred.' = predicted match rate)\n")

    print("=" * 92)
    if failures:
        print(f"FAIL: {len(failures)} check failures")
        for f in failures[:30]:
            print(f"   setting={f[0]} seed={f[1]} policy={f[2]} check={f[3]}")
    else:
        n = len(CAPACITY_SETTINGS) * len(SEEDS) * len(POLICIES)
        print(f"PASS: all 4 checks (A menus subset of {{j*}}, B no match elsewhere, "
              f"C capacity respected,\n      D matches = min(cap, #offered who "
              f"prefer j* under realized theta)) held for all {n} "
              f"(setting, seed, policy) cases.")


if __name__ == "__main__":
    main()
