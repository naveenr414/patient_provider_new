"""SAM at k = M (no menu cap), and SAM with an inclusion floor tau = 0.1*eps.

Two questions, both raised by `offer_everything_noise_sweep.py`, which found
uncapped offer-everything beats SAM(k=25) above eps ~ 0.18:

1. Is that a fair fight? Offer-everything got k = M and SAM got k = 25. Give
   SAM the same budget -- what does Algorithm 1 do when the menu-size
   constraint it was designed around is removed?
2. With no k, the only thing left limiting a menu is Algorithm 1's inclusion
   test m_ij > 0. That test barely bites (`sam_score_vs_oracle.py`: m is
   positive for ~133 providers per patient because averaging max(Delta-lambda, 0)
   over S scenarios smears every near-tie positive). Replacing it with a
   magnitude floor m_ij > tau is the knob that decides menu size instead. tau
   is set to 0.1*eps so it scales with the noise the scores are averaged over.

Four menu rules, all derived from ONE score computation per (eps, seed) -- the
S=10 dual LPs are ~95% of SAM's cost, and every rule is a different selection
over the same m_ij:

    sam_k25            k=25,  m > 0          reproduces results/noise `sam`
    sam_uncapped       k=M,   m > 0          question 1
    sam_k25_tau        k=25,  m > tau        the floor alone, at the usual k
    sam_uncapped_tau   k=M,   m > tau        question 2

sam_k25 is recomputed rather than read from results/noise so that all four
share identical scores (the dual LP has multiple optimal bases, so a re-solve
need not return the same lambda). It doubles as a check: it should land on top
of the stored `sam` curve.

Config is the `noise` experiment's: N=1225, M=700, 9 seeds x 25 trials,
eps in {0.01, ..., 0.5}. Omniscient normalizers are read from results/noise
(policy-independent at a given (eps, seed)), so no normalizer LPs are solved.
Trials are driven entirely by seed-derived trial seeds, not by the policy, so
every number here is paired with the stored baselines patient-for-patient.

Results go to `results/noise_sam_uncapped/` in the standard sweep shape.

Run:  PYTHONPATH=. python tests/debugging/sam_uncapped_noise_sweep.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    DEFAULT_INSTANCE, DEFAULT_K, DEFAULT_M, DEFAULT_N, DEFAULT_S, EPS_GRID,
    aggregate_seeds, build_instance, save_result,
)

ROOT = Path(__file__).resolve().parents[2]
NOISE_DIR = ROOT / "results" / "noise"
OUT_DIR = ROOT / "results" / "noise_sam_uncapped"

SEEDS = list(range(9))
NUM_TRIALS = 25
EPSILONS = EPS_GRID["epsilon"]
TAU_SCALE = 0.1                 # tau = TAU_SCALE * epsilon
JOBS = len(SEEDS) * len(EPSILONS)
# Each of the 54 workers solves 10 LPs with N*M = 857,500 variables. Left at
# Gurobi's default every worker would try to use all 112 cores; 2 keeps the
# machine from thrashing. Pinned (rather than left to the concurrent
# optimizer) for a second reason: it makes the basis, and therefore the duals,
# reproducible run to run.
LP_THREADS = 2

VARIANTS = {
    "sam_k25":          dict(k=DEFAULT_K, tau_scale=0.0),
    "sam_uncapped":     dict(k=DEFAULT_M, tau_scale=0.0),
    "sam_k25_tau":      dict(k=DEFAULT_K, tau_scale=TAU_SCALE),
    "sam_uncapped_tau": dict(k=DEFAULT_M, tau_scale=TAU_SCALE),
}


def se(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v))


def scenario_duals(delta, capacities):
    """Provider dual prices of one scenario's LP relaxation -- the same model
    `policies.sam` builds, with the thread count pinned."""
    N, M = delta.shape
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = LP_THREADS
    x = model.addVars(N, M, lb=0.0, ub=1.0)
    model.setObjective(
        gp.quicksum(delta[i, j] * x[i, j] for i in range(N) for j in range(M)),
        GRB.MAXIMIZE)
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= 1)
    cap = [model.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= capacities[j])
           for j in range(M)]
    model.optimize()
    return np.array([c.Pi for c in cap])


def sam_scores(theta_hat, capacities, epsilon, S, seed):
    """m_ij and lambda_bar exactly as `policies.sam` computes them.

    The doubly-nested RandomState reproduces the seed `run_trials` actually
    hands the policy: it draws the policy's seed from a RandomState(seed)
    before the trial loop (`simulator.run_trials`, the `extra["seed"]` line),
    so these are the scores the deployed SAM menus were built from."""
    M = theta_hat.shape[1] - 1
    rng = np.random.RandomState(np.random.RandomState(seed).randint(2 ** 31))
    scenarios = [create_random_weights(theta_hat, epsilon, rng) for _ in range(S)]
    deltas = [t[:, :M] - t[:, M:M + 1] for t in scenarios]

    lam = np.zeros(M)
    for d in deltas:
        lam += scenario_duals(d, capacities) / S
    m = np.zeros((theta_hat.shape[0], M))
    for d in deltas:
        m += np.maximum(d - lam[None, :], 0) / S
    return m, lam


def stored_omniscient():
    """{epsilon: {seed: E_theta[OPT(theta)]}} from the existing noise sweep."""
    omni = {}
    for f in sorted(NOISE_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        assert d["N"] == DEFAULT_N and d["M"] == DEFAULT_M, f"{f.name}: unexpected scale"
        assert d["num_trials"] == NUM_TRIALS, f"{f.name}: unexpected trial count"
        per_eps = omni.setdefault(d["epsilon"], {})
        for r in d["per_seed"]:
            per_eps.setdefault(r["seed"], r["omniscient_utility"])
    return omni


def job(args):
    """One (epsilon, seed): score once, then simulate each of the four rules."""
    epsilon, seed, omniscient = args
    t0 = time.time()
    theta_hat, capacities, patients = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                                      **DEFAULT_INSTANCE)
    live = capacities > 0
    m, _lam = sam_scores(theta_hat, capacities, epsilon, DEFAULT_S, seed)

    out = {}
    for name, spec in VARIANTS.items():
        tau = spec["tau_scale"] * epsilon
        # `_topk_mask(m - tau, ...)` keeps entries with m - tau > 0, i.e.
        # m > tau: the same positivity test Algorithm 1 applies, shifted.
        menus = _topk_mask(m - tau, live, spec["k"])
        res = run_trials(theta_hat, capacities,
                         lambda th, c, kk, _m=menus: _m.copy(),
                         spec["k"], num_trials=NUM_TRIALS, epsilon=epsilon, seed=seed)
        theta_realized = res["theta_realized"]
        util = MET.utility(res["rewards"])
        zips = [p["zip"] for p in patients]
        fair = MET.zip_fairness(res["chosen"], DEFAULT_M, zips, rewards=res["rewards"])
        out[name] = {
            "seed": seed,
            "runtime_sec": time.time() - t0,
            "utility": util,
            "normalized_utility": util / omniscient,
            "omniscient_utility": omniscient,
            "match_rate": MET.match_rate(res["chosen"], DEFAULT_M),
            "choice_count_mean": float(MET.choice_count(res["effective_menus"]).mean()),
            "choice_utility_mean": float(np.nanmean(
                MET.choice_utility(res["effective_menus"], theta_realized))),
            "zip_fairness_p25": fair["p25"],
            "zip_gini_match_rate": fair["gini_match_rate"],
            "zip_gini_utility": fair["gini_utility"],
            # planned menu size, before the simulator intersects with live
            # capacity -- this is what the tau floor actually controls
            "menu_planned": float(menus.sum(axis=1).mean()),
            "offered_frac": float((menus.sum(axis=1) > 0).mean()),
            "tau": tau,
        }
    return epsilon, seed, out


def main():
    omni = stored_omniscient()
    missing = [e for e in EPSILONS if e not in omni]
    assert not missing, f"no stored omniscient for epsilon={missing}; run `noise` first"

    print(f"N={DEFAULT_N} M={DEFAULT_M} S={DEFAULT_S} | {len(SEEDS)} seeds x "
          f"{NUM_TRIALS} trials | epsilon={EPSILONS} | tau={TAU_SCALE}*epsilon\n"
          f"variants: {', '.join(VARIANTS)}\n"
          f"{len(EPSILONS) * len(SEEDS)} (eps, seed) jobs on {JOBS} workers, "
          f"{LP_THREADS} LP threads each; scores computed once per job\n", flush=True)

    specs = [(eps, seed, omni[eps][seed]) for eps in EPSILONS for seed in SEEDS]
    t0 = time.time()
    recs = {name: {eps: [] for eps in EPSILONS} for name in VARIANTS}
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futures = [ex.submit(job, s) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            eps, seed, out = fut.result()
            for name in VARIANTS:
                recs[name][eps].append(out[name])
            print(f"  [{i}/{len(specs)}] ({time.time() - t0:.0f}s) eps={eps} seed={seed} "
                  + " ".join(f"{n.replace('sam_', '')}={out[n]['normalized_utility']:.4f}"
                             f"/{out[n]['menu_planned']:.0f}" for n in VARIANTS),
                  flush=True)

    for name in VARIANTS:
        for eps in EPSILONS:
            per_seed = sorted(recs[name][eps], key=lambda r: r["seed"])
            result = aggregate_seeds(name, DEFAULT_N, DEFAULT_M, VARIANTS[name]["k"],
                                     eps, NUM_TRIALS, per_seed)
            save_result(OUT_DIR, "", name, {"epsilon": eps}, result)

    # ------------------------------------------------------------- reporting
    stored = {}
    for f in sorted(NOISE_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        stored.setdefault(d["policy"], {})[d["epsilon"]] = {
            r["seed"]: r for r in d["per_seed"]}
    for f in sorted((ROOT / "results" / "noise_offer_everything").glob("*.json")):
        d = json.loads(f.read_text())
        stored.setdefault(d["policy"], {})[d["epsilon"]] = {
            r["seed"]: r for r in d["per_seed"]}

    rows = {n: {e: [r["normalized_utility"] for r in sorted(recs[n][e],
                                                            key=lambda r: r["seed"])]
                for e in EPSILONS} for n in VARIANTS}
    for p in stored:
        rows[p] = {e: [stored[p][e][s]["normalized_utility"] for s in SEEDS]
                   for e in EPSILONS}

    def table(title, keys, cell):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        print(f"{'policy':22s}" + "".join(f"{('eps=' + str(e)):>11s}" for e in EPSILONS))
        for p in keys:
            print(f"{p:22s}" + "".join(f"{cell(p, e):>11s}" for e in EPSILONS))

    order = list(VARIANTS) + ["offer_everything", "sam", "offer_all", "offer_one", "random"]
    order = [p for p in order if p in rows]
    table("NORMALIZED UTILITY vs EPSILON (mean across 9 seeds; SE <= 0.003 throughout)",
          order, lambda p, e: f"{np.mean(rows[p][e]):.4f}")

    print("\n" + "=" * 96)
    print("PLANNED MENU SIZE (providers offered per patient, before capacity masking)")
    print("=" * 96)
    print(f"{'variant':22s}{'tau':>8s}" + "".join(f"{('eps=' + str(e)):>11s}"
                                                  for e in EPSILONS))
    for n in VARIANTS:
        taus = [np.mean([r["tau"] for r in recs[n][e]]) for e in EPSILONS]
        cells = "".join(f"{np.mean([r['menu_planned'] for r in recs[n][e]]):11.1f}"
                        for e in EPSILONS)
        print(f"{n:22s}{('0.1eps' if taus[-1] > 0 else '0'):>8s}{cells}")
    print(f"\n{'variant':22s}{'':8s}" + "".join(f"{('eps=' + str(e)):>11s}"
                                                for e in EPSILONS)
          + "   (% of patients offered a non-empty menu)")
    for n in VARIANTS:
        cells = "".join(f"{100 * np.mean([r['offered_frac'] for r in recs[n][e]]):10.1f}%"
                        for e in EPSILONS)
        print(f"{n:22s}{'':8s}{cells}")

    print("\n" + "=" * 96)
    print("PAIRED DIFFERENCES IN NORMALIZED UTILITY (per seed; t in parentheses)")
    print("=" * 96)
    pairs = [
        ("sam_k25", "sam", "reproduction check -- should be ~0"),
        ("sam_uncapped", "sam_k25", "removing the k=25 cap"),
        ("sam_uncapped", "offer_everything", "uncapped SAM vs uncapped offer-everything"),
        ("sam_k25_tau", "sam_k25", "the tau floor at k=25"),
        ("sam_uncapped_tau", "sam_uncapped", "the tau floor with no k"),
        ("sam_uncapped_tau", "sam_k25", "best variant vs Algorithm 1 as published"),
    ]
    for a, b, note in pairs:
        if a not in rows or b not in rows:
            continue
        print(f"\n{a} - {b}   ({note})")
        print(f"{'':4s}" + "".join(f"{('eps=' + str(e)):>18s}" for e in EPSILONS))
        cells = []
        for e in EPSILONS:
            d = np.array(rows[a][e]) - np.array(rows[b][e])
            s = se(d)
            t = d.mean() / s if s > 0 else float("nan")
            cells.append(f"{d.mean():+.4f} ({t:+6.1f})".rjust(18))
        print(f"{'':4s}" + "".join(cells))

    print(f"\nwrote {OUT_DIR}/ ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
