"""Ground-truth check for `policies.exact_saa_milp`.

On instances small enough to enumerate every admissible menu, compute the SAA
objective exactly by simulating the S scenarios, and compare the brute-force
optimum against what the MILP returns. The MILP claims to be the exact
optimizer of that same objective, so the two must agree on the objective value
(the argmax menu may differ under ties).

Scenario generation is replicated bit-for-bit from `exact_saa_milp`, so both
sides score the identical (theta^(s), sigma^(s)) pairs.

Run:  PYTHONPATH=. python tests/debugging/milp_bruteforce_check.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import policies as P
from patient.utils import create_random_weights


def make_scenarios(theta_hat, epsilon, S, seed):
    """Exactly what exact_saa_milp draws internally."""
    rng = np.random.RandomState(seed)
    N = theta_hat.shape[0]
    out = []
    for _ in range(S):
        theta_s = create_random_weights(theta_hat, epsilon, rng)
        order = rng.permutation(N)
        out.append((theta_s, order))
    return out


def saa_objective(menu, capacities, scenarios):
    """Exact sum over scenarios of realized utility under sequential arrivals
    with rational choice -- the quantity Eq. 2 maximizes."""
    N, M = menu.shape
    total = 0.0
    for theta_s, order in scenarios:
        remaining = np.array(capacities, dtype=int).copy()
        for i in order:
            avail = (menu[i] == 1) & (remaining > 0)
            best_j, best_v = M, theta_s[i, M]          # exit
            for j in np.flatnonzero(avail):
                if theta_s[i, j] > best_v:
                    best_j, best_v = j, theta_s[i, j]
            if best_j < M:
                remaining[best_j] -= 1
            total += best_v
    return total


def all_menus(N, M, k, live):
    """Every N x M 0/1 menu with row sums <= k, restricted to live providers."""
    live_idx = np.flatnonzero(live)
    rows = []
    for r in range(min(k, len(live_idx)) + 1):
        for combo in itertools.combinations(live_idx, r):
            row = np.zeros(M, dtype=int)
            row[list(combo)] = 1
            rows.append(row)
    for choice in itertools.product(range(len(rows)), repeat=N):
        yield np.array([rows[c] for c in choice])


def check(name, theta_hat, capacities, k, epsilon, S, seed):
    N, M = theta_hat.shape[0], theta_hat.shape[1] - 1
    live = np.asarray(capacities) > 0
    scenarios = make_scenarios(theta_hat, epsilon, S, seed)

    best_val, best_menu, count = -np.inf, None, 0
    for menu in all_menus(N, M, k, live):
        v = saa_objective(menu, capacities, scenarios)
        count += 1
        if v > best_val:
            best_val, best_menu = v, menu.copy()

    milp_menu = P.exact_saa_milp(theta_hat, np.asarray(capacities), k,
                                 epsilon=epsilon, S=S, seed=seed)
    milp_val = saa_objective(milp_menu, capacities, scenarios)

    ok = np.isclose(milp_val, best_val, atol=1e-6)
    print(f"{name}: N={N} M={M} k={k} S={S} eps={epsilon} "
          f"({count} menus enumerated)")
    print(f"   brute-force optimum : {best_val / S:.6f}  (per scenario)")
    print(f"   MILP menu scores    : {milp_val / S:.6f}  "
          f"{'OK' if ok else '<-- MISMATCH'}")
    if not ok:
        print(f"   shortfall           : {(best_val - milp_val) / S:.6f} "
              f"({100 * (1 - milp_val / best_val):.2f}%)")
        print(f"   brute-force menu row sums: {best_menu.sum(axis=1)}")
        print(f"   MILP menu row sums       : {milp_menu.sum(axis=1)}")
        print(f"   brute-force menu:\n{best_menu}")
        print(f"   MILP menu:\n{milp_menu}")
    print()
    return ok


def main():
    cases = []

    # 1) tiny, unit capacity, k=1
    th = np.array([[0.8, 0.3, 0.2],
                   [0.7, 0.6, 0.2],
                   [0.4, 0.9, 0.2]])
    cases.append(("case1", th, [1, 1], 1, 0.15, 5, 0))

    # 2) same but k=2 (menus actually matter)
    cases.append(("case2", th, [1, 1], 2, 0.25, 5, 1))

    # 3) contention: 4 patients, 2 slots, an exit worth taking
    th2 = np.array([[0.9, 0.5, 0.3],
                    [0.85, 0.55, 0.3],
                    [0.8, 0.6, 0.3],
                    [0.75, 0.65, 0.3]])
    cases.append(("case3", th2, [1, 1], 2, 0.2, 5, 2))

    # 4) a provider with capacity 2, and one dead provider
    th3 = np.array([[0.9, 0.4, 0.6, 0.25],
                    [0.8, 0.5, 0.6, 0.25],
                    [0.7, 0.6, 0.6, 0.25],
                    [0.6, 0.7, 0.6, 0.25]])
    cases.append(("case4", th3, [2, 1, 0], 2, 0.2, 5, 3))

    results = [check(*c) for c in cases]
    print("=" * 70)
    if all(results):
        print(f"PASS: MILP matched the brute-force optimum on all {len(results)} cases.")
    else:
        print(f"FAIL: {results.count(False)}/{len(results)} cases mismatched -- "
              "the MILP is not solving Eq. 2's objective.")


if __name__ == "__main__":
    main()
