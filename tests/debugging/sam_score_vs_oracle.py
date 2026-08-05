"""How close is SAM's score m_ij to the score it would compute with hindsight?

SAM (Algorithm 1) never sees the realized theta. It samples S scenarios around
theta_hat, solves each scenario's LP for provider duals, averages them into
lambda_bar, and scores

    m_ij = mean_s[ max( Delta^(s)_ij - lambda_bar_j , 0 ) ],
    Delta^(s)_ij = theta^(s)_ij - theta^(s)_i,exit

The hindsight ("oracle") counterpart uses the ONE realized theta and ITS OWN
dual prices from the same LP:

    z_ij = max( Delta*_ij - lambda*_j , 0 ),
    Delta*_ij = theta*_ij - theta*_i,exit,  lambda* = duals at theta*

This compares the two score distributions, the two dual vectors, and -- since
only the induced ranking matters -- the top-k menus they select.

One instance seed and one theta realization, as requested: the whole thing is
S+1 LPs on the full 1225 x 700 instance.

Run:  PYTHONPATH=. python tests/debugging/sam_score_vs_oracle.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient.policies import _topk_mask
from patient.simulator import Simulator, draw_realized_theta
from patient.utils import create_random_weights
from scripts.run_experiments import (
    build_instance, DEFAULT_INSTANCE, DEFAULT_N, DEFAULT_M, DEFAULT_K,
    DEFAULT_EPSILON, DEFAULT_S,
)

OUT = Path(__file__).resolve().parent
SEED = 0
TRIAL = 0          # which realized theta to treat as the truth


def solve_duals(delta, capacities):
    """SAM's inner LP on a Delta matrix; returns (duals, objective)."""
    N, M = delta.shape
    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addVars(N, M, lb=0.0, ub=1.0)
    model.setObjective(
        gp.quicksum(delta[i, j] * x[i, j] for i in range(N) for j in range(M)),
        GRB.MAXIMIZE)
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= 1)
    cap = [model.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= capacities[j])
           for j in range(M)]
    model.optimize()
    return np.array([c.Pi for c in cap]), model.ObjVal


def describe(name, v):
    q = np.percentile(v, [50, 75, 90, 95, 99, 100])
    print(f"{name:26s} mean={v.mean():.4f} sd={v.std():.4f} "
          f"zero={100 * np.isclose(v, 0).mean():5.2f}% | med={q[0]:.4f} "
          f"p75={q[1]:.4f} p90={q[2]:.4f} p95={q[3]:.4f} p99={q[4]:.4f} "
          f"max={q[5]:.4f}")


def simulate(theta, capacities, menus, sim_seed):
    N, M = theta.shape[0], theta.shape[1] - 1
    sim = Simulator(theta, capacities, gamma=None, seed=sim_seed)
    sim.reset_initial(); sim.reset_patient_order()
    total = 0.0
    for t in sim.patient_order:
        menu = np.concatenate([menus[t] * (sim.provider_capacities > 0).astype(int), [1]])
        total += sim.all_patients[t].theta_row[sim.step(int(t), menu)]
    return total / N


def main():
    t0 = time.time()
    print(f"seed={SEED} trial={TRIAL} N={DEFAULT_N} M={DEFAULT_M} k={DEFAULT_K} "
          f"epsilon={DEFAULT_EPSILON} S={DEFAULT_S}\n")
    theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=SEED,
                                               **DEFAULT_INSTANCE)
    M = DEFAULT_M
    live = capacities > 0

    # the realization SAM is judged against (same stream run_trials uses)
    theta_star = draw_realized_theta(theta_hat, DEFAULT_EPSILON, TRIAL + 1,
                                     SEED)[TRIAL].astype(float)

    # ---- SAM's own scenarios, duals, score (mirrors policies.sam exactly) ----
    rng = np.random.RandomState(np.random.RandomState(SEED).randint(2 ** 31))
    scenarios = [create_random_weights(theta_hat, DEFAULT_EPSILON, rng)
                 for _ in range(DEFAULT_S)]
    deltas = [t[:, :M] - t[:, M:M + 1] for t in scenarios]
    lam_bar = np.zeros(M)
    scenario_duals = []
    for s, d in enumerate(deltas):
        lam_s, _ = solve_duals(d, capacities)
        scenario_duals.append(lam_s)
        lam_bar += lam_s / DEFAULT_S
        print(f"  scenario {s + 1}/{DEFAULT_S} solved ({time.time() - t0:.0f}s)")
    scenario_duals = np.array(scenario_duals)
    m = np.zeros((DEFAULT_N, M))
    for d in deltas:
        m += np.maximum(d - lam_bar[None, :], 0) / DEFAULT_S

    # ---- the hindsight counterpart on the realized theta ----
    delta_star = theta_star[:, :M] - theta_star[:, M:M + 1]
    lam_star, _ = solve_duals(delta_star, capacities)
    z = np.maximum(delta_star - lam_star[None, :], 0)
    print(f"  oracle LP solved ({time.time() - t0:.0f}s)\n")

    # also: the score using the TRUE theta but SAM's averaged duals, to
    # attribute the m-vs-z gap between "wrong duals" and "wrong theta"
    z_lambar = np.maximum(delta_star - lam_bar[None, :], 0)

    # ------------------------------------------------------------ 1) scores
    print("=" * 92)
    print("1. SCORE DISTRIBUTIONS  (all N x M = %d entries)" % m.size)
    print("=" * 92)
    describe("m_ij  (SAM)", m.ravel())
    describe("z_ij  (oracle)", z.ravel())
    describe("z_ij w/ lambda_bar", z_lambar.ravel())
    print()
    describe("m_ij, live providers", m[:, live].ravel())
    describe("z_ij, live providers", z[:, live].ravel())
    print(f"\npositive-score entries per patient (the pool top-k draws from):")
    print(f"   m: mean {(m[:, live] > 0).sum(axis=1).mean():7.1f}  "
          f"median {np.median((m[:, live] > 0).sum(axis=1)):7.1f}  "
          f"min {(m[:, live] > 0).sum(axis=1).min()}  "
          f"max {(m[:, live] > 0).sum(axis=1).max()}")
    print(f"   z: mean {(z[:, live] > 0).sum(axis=1).mean():7.1f}  "
          f"median {np.median((z[:, live] > 0).sum(axis=1)):7.1f}  "
          f"min {(z[:, live] > 0).sum(axis=1).min()}  "
          f"max {(z[:, live] > 0).sum(axis=1).max()}")

    # -------------------------------------------------------- 2) m vs z gap
    print("\n" + "=" * 92)
    print("2. HOW DIFFERENT ARE m AND z?")
    print("=" * 92)
    for label, a, b in [("m vs z", m, z),
                        ("m vs z (same duals)", m, z_lambar),
                        ("z_lambar vs z", z_lambar, z)]:
        d = (a - b)[:, live].ravel()
        both_pos = ((a[:, live] > 0) & (b[:, live] > 0))
        agree = ((a[:, live] > 0) == (b[:, live] > 0)).mean()
        print(f"{label:22s} mean_diff={d.mean():+.4f}  MAE={np.abs(d).mean():.4f}  "
              f"RMSE={np.sqrt((d ** 2).mean()):.4f}  "
              f"corr={np.corrcoef(a[:, live].ravel(), b[:, live].ravel())[0, 1]:.4f}  "
              f"sign-agree={100 * agree:.2f}%  both>0={100 * both_pos.mean():.2f}%")

    # decision-relevant: do they pick the same menus?
    print("\ntop-k menu overlap (k=%d, the only thing the score is used for):" % DEFAULT_K)
    menu_m = _topk_mask(m, live, DEFAULT_K)
    menu_z = _topk_mask(z, live, DEFAULT_K)
    inter = (menu_m & menu_z).sum(axis=1)
    print(f"   |X_SAM| mean {menu_m.sum(axis=1).mean():.2f}, "
          f"|X_oracle| mean {menu_z.sum(axis=1).mean():.2f}, "
          f"overlap mean {inter.mean():.2f} "
          f"({100 * inter.sum() / max(menu_m.sum(), 1):.1f}% of SAM's picks)")
    print(f"   patients with identical menus: "
          f"{100 * (inter == menu_m.sum(axis=1)).mean():.2f}%")
    # rank correlation on a subsample of rows (Spearman is O(M log M) per row)
    from scipy.stats import spearmanr
    idx = np.random.RandomState(0).choice(DEFAULT_N, 200, replace=False)
    rhos = [spearmanr(m[i, live], z[i, live]).statistic for i in idx]
    print(f"   per-patient Spearman rho(m, z) over live providers: "
          f"mean {np.nanmean(rhos):.4f}, median {np.nanmedian(rhos):.4f}")

    # ------------------------------------------------------------- 3) duals
    print("\n" + "=" * 92)
    print("3. DUALS: lambda_bar (SAM's average) VS lambda* (realized theta)")
    print("=" * 92)
    describe("lambda_bar", lam_bar)
    describe("lambda*", lam_star)
    describe("per-scenario lambda", scenario_duals.ravel())
    d = lam_bar - lam_star
    print(f"\nmean_diff={d.mean():+.4f}  MAE={np.abs(d).mean():.4f}  "
          f"RMSE={np.sqrt((d ** 2).mean()):.4f}  "
          f"corr={np.corrcoef(lam_bar, lam_star)[0, 1]:.4f}")
    print(f"nonzero duals: lambda_bar {100 * (lam_bar > 1e-9).mean():.2f}%, "
          f"lambda* {100 * (lam_star > 1e-9).mean():.2f}%  "
          f"(providers with c_j>0: {100 * live.mean():.2f}%)")
    print(f"on live providers: MAE={np.abs(d[live]).mean():.4f}, "
          f"relative MAE={np.abs(d[live]).mean() / max(lam_star[live].mean(), 1e-9):.3f}")
    print(f"scenario-to-scenario sd of lambda_j (mean over j): "
          f"{scenario_duals.std(axis=0).mean():.4f}  -- i.e. how much averaging "
          f"S={DEFAULT_S} scenarios is smoothing")
    print(f"|lambda_bar - lambda*| vs that sd: ratio "
          f"{np.abs(d).mean() / max(scenario_duals.std(axis=0).mean(), 1e-9):.3f}")

    # ------------------------------------------ 4) does the difference matter?
    print("\n" + "=" * 92)
    print("4. VALUE OF THE TWO MENUS ON THE REALIZED theta (10 arrival orders)")
    print("=" * 92)
    sig = np.random.RandomState(999).randint(2 ** 31, size=10)
    v_m = np.array([simulate(theta_star, capacities, menu_m, int(s)) for s in sig])
    v_z = np.array([simulate(theta_star, capacities, menu_z, int(s)) for s in sig])
    print(f"   V(X from m) = {v_m.mean():.4f} +/- {v_m.std():.4f}")
    print(f"   V(X from z) = {v_z.mean():.4f} +/- {v_z.std():.4f}   "
          f"(hindsight advantage {100 * (v_z.mean() / v_m.mean() - 1):+.2f}%)")

    np.savez(OUT / "sam_score_vs_oracle.npz", m=m.astype(np.float32),
             z=z.astype(np.float32), lam_bar=lam_bar, lam_star=lam_star,
             scenario_duals=scenario_duals, capacities=capacities)
    print(f"\nsaved {OUT / 'sam_score_vs_oracle.npz'}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
