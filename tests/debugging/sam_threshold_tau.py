"""Does SAM's `m > 0` rule leave anything on the table at full scale?

Instance A (tests/debugging/intuition_instances.py) showed that replacing
Algorithm 1's knife-edge inclusion test `m_ij > 0` with a small magnitude
floor `m_ij > tau` recovered offer-one's performance there. tau = 0.01 was
tuned by hand on that instance; this script asks what QUANTILE of the m
distribution that corresponded to, then applies the same quantile at
N=1225 to see whether the effect generalizes.

Cheap by construction: SAM's scores and duals are computed once per seed
(10 LPs), then every tau reuses them -- only the top-k selection and the
simulation are redone. Omniscient normalizers are read from
results/default (policy-independent at a given seed).

Run:  PYTHONPATH=. python tests/debugging/sam_threshold_tau.py
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient.policies import _topk_mask
from patient.simulator import run_trials
from patient.utils import create_random_weights
from scripts.run_experiments import (
    build_instance, DEFAULT_INSTANCE, DEFAULT_N, DEFAULT_M, DEFAULT_K,
    DEFAULT_EPSILON, DEFAULT_S,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(9))
NUM_TRIALS = 25
QUANTILES = [0.0, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]


def scenario_duals(delta, capacities):
    N, M = delta.shape
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(N, M, lb=0.0, ub=1.0)
    m.setObjective(gp.quicksum(delta[i, j] * x[i, j] for i in range(N) for j in range(M)),
                   GRB.MAXIMIZE)
    for i in range(N):
        m.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= 1)
    cap = [m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= capacities[j])
           for j in range(M)]
    m.optimize()
    return np.array([c.Pi for c in cap])


def sam_scores(theta_hat, capacities, epsilon, S, seed):
    """m_ij and lambda_bar, exactly as policies.sam computes them (same seed
    derivation run_trials uses, so this reproduces the deployed menus)."""
    M = theta_hat.shape[1] - 1
    rng = np.random.RandomState(np.random.RandomState(seed).randint(2 ** 31))
    scen = [create_random_weights(theta_hat, epsilon, rng) for _ in range(S)]
    deltas = [t[:, :M] - t[:, M:M + 1] for t in scen]
    lam = np.zeros(M)
    for d in deltas:
        lam += scenario_duals(d, capacities) / S
    m = np.zeros((theta_hat.shape[0], M))
    for d in deltas:
        m += np.maximum(d - lam[None, :], 0) / S
    return m, lam


def instance_a_quantile():
    """What quantile of m was tau = 0.01 on instance A?"""
    N, M = 8, 2
    th = np.zeros((N, M + 1))
    th[:, 0] = 0.45
    th[0, 0] = 0.75
    th[:, 1] = 0.40
    caps = np.array([1, 1])
    m, _ = sam_scores(th, caps, 0.20, 10, 0)
    flat = m.ravel()
    tau = 0.01
    q_all = (flat <= tau).mean()
    pos = flat[flat > 0]
    q_pos = (pos <= tau).mean()
    print("instance A: where did the hand-tuned tau = 0.01 sit?")
    print(f"   m entries: {flat.size} total, {(flat == 0).mean() * 100:.1f}% exactly zero")
    print(f"   tau=0.01 is the {100 * q_all:.1f}th percentile of ALL m entries")
    print(f"   tau=0.01 is the {100 * q_pos:.1f}th percentile of POSITIVE m entries\n")
    return q_all, q_pos


def run_seed(seed):
    theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                               **DEFAULT_INSTANCE)
    live = capacities > 0
    m, lam = sam_scores(theta_hat, capacities, DEFAULT_EPSILON, DEFAULT_S, seed)

    # taus from quantiles of this seed's own positive-score distribution
    pos = m[:, live]
    pos = pos[pos > 0]
    out = {"seed": seed, "frac_zero": float((m[:, live] == 0).mean()), "taus": {}}
    for q in QUANTILES:
        tau = 0.0 if q == 0.0 else float(np.quantile(pos, q))
        menus = _topk_mask(m - tau, live, DEFAULT_K)
        res = run_trials(theta_hat, capacities,
                         lambda th, c, kk, _m=menus: _m.copy(),
                         DEFAULT_K, num_trials=NUM_TRIALS,
                         epsilon=DEFAULT_EPSILON, seed=seed)
        out["taus"][q] = dict(
            tau=tau,
            utility=MET.utility(res["rewards"]),
            match_rate=MET.match_rate(res["chosen"], DEFAULT_M),
            menu=float(menus.sum(axis=1).mean()),
            offered_frac=float((menus.sum(axis=1) > 0).mean()),
        )
    return out


def main():
    q_all, q_pos = instance_a_quantile()

    with Pool(len(SEEDS)) as pool:
        recs = pool.map(run_seed, SEEDS)
    recs.sort(key=lambda r: r["seed"])

    omni = {}
    for f in glob.glob(str(ROOT / "results/default/*.json")):
        for r in json.load(open(f))["per_seed"]:
            omni[r["seed"]] = r["omniscient_utility"]

    print(f"N={DEFAULT_N} M={DEFAULT_M} k={DEFAULT_K} eps={DEFAULT_EPSILON} "
          f"S={DEFAULT_S} | {len(SEEDS)} seeds x {NUM_TRIALS} trials")
    print(f"m is {100 * np.mean([r['frac_zero'] for r in recs]):.1f}% exactly zero "
          f"on live providers\n")
    print(f"{'quantile':>9s} {'tau':>9s} {'norm. util':>11s} {'SE':>8s} "
          f"{'menu':>6s} {'match':>7s} {'w/ menu':>8s}")

    base = None
    for q in QUANTILES:
        norm = [r["taus"][q]["utility"] / omni[r["seed"]] for r in recs]
        tau = np.mean([r["taus"][q]["tau"] for r in recs])
        menu = np.mean([r["taus"][q]["menu"] for r in recs])
        mr = np.mean([r["taus"][q]["match_rate"] for r in recs])
        off = np.mean([r["taus"][q]["offered_frac"] for r in recs])
        se = np.std(norm, ddof=1) / np.sqrt(len(norm))
        if base is None:
            base = norm
        tag = "   <- Algorithm 1 (m > 0)" if q == 0.0 else ""
        print(f"{q:>9.3f} {tau:>9.5f} {np.mean(norm):>11.4f} {se:>8.4f} "
              f"{menu:>6.2f} {mr:>7.4f} {100 * off:>7.1f}%{tag}")

    print("\npaired difference vs Algorithm 1 (same seeds, same trials):")
    for q in QUANTILES[1:]:
        norm = [r["taus"][q]["utility"] / omni[r["seed"]] for r in recs]
        d = np.array(norm) - np.array(base)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"   q={q:<6.3f} d={d.mean():+.4f} +/- {se:.4f}  "
              f"t={d.mean() / se if se > 0 else float('nan'):>6.2f}  "
              f"{sum(x > 0 for x in d)}/9 seeds better")

    print(f"\ninstance A's hand-tuned tau sat at the {100 * q_pos:.0f}th percentile of "
          f"positive m; read that row above.")


if __name__ == "__main__":
    main()
