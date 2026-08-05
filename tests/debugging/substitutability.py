"""How substitutable are providers for a patient, and patients for a provider?

Motivated by H3 (Top-k redundancy), the one assumption among H2-H6 that is
purely about substitutability AND fully computable from what the codebase
already has. H3 says: with r_ij := Delta_ij - lambda*_j the true reduced
surplus, r+_ij its positive part, r+_{i,(k)} the k-th largest of those over
providers, and mu*_i := r+_{i,(1)},

    for every patient i with mu*_i > 0,   r+_{i,(k)} >= mu*_i - gamma * eps.

In words: a patient's k-th best option is worth nearly as much as their best,
so a menu of size k is full of near-equivalent substitutes. That is exactly a
statement about how fast a patient's reduced surplus decays down their ranked
list, which is what this script measures.

WHAT THIS DOES NOT DO. H4 (Def_{alpha eps}), H5 (the capacity-weighted spread
w_j) and H6 (fill shortfall F) are not evaluated: the repo's paper.pdf is an
older draft carrying the R1-R3 regularity conditions instead, so it has no
definition of Def, w_j, d_j or n_j. Panel 2 below is therefore the natural
PROVIDER-SIDE MIRROR of H3, defined here and named as such -- it is not H5.

Definitions used (all on the realized theta, not theta_hat):
  lambda*_j  the dual of provider j's capacity constraint in the omniscient LP
             solved on the REALIZED theta -- the "true" dual the assumptions
             are stated against, not SAM's lambda_bar estimate of it.
  Delta_ij   theta_ij - theta_i,exit.
  r_ij       Delta_ij - lambda*_j, and r+_ij = max(r_ij, 0).

Providers with c_j = 0 are dropped throughout. 36.8% of providers start at
zero capacity, their capacity constraint is sum_i x_ij <= 0, and its dual is a
degenerate artifact rather than a price -- including them would inject noise
into every r_ij at once.

The gaps are reported DIVIDED BY eps, because that is the quantity the
assumptions bound by a constant (gamma). Plotting gap/eps against k for each
eps therefore doubles as a test of the eps-scaling itself: if the curves
collapse onto one another, a single constant gamma covers every noise level,
which is what H3 asserts. If they fan out, gamma is not constant in eps.

Run:  PYTHONPATH=. python tests/debugging/substitutability.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient.plotting import color_schemes, create_axes, markers as plot_markers
from patient.simulator import draw_realized_theta
from scripts.run_experiments import (
    DEFAULT_INSTANCE, DEFAULT_K, DEFAULT_M, DEFAULT_N, EPS_GRID, build_instance,
)

HERE = Path(__file__).resolve().parent
SEEDS = list(range(9))
EPSILONS = EPS_GRID["epsilon"]
TRIAL = 0                 # one realized theta per (eps, seed) is plenty: the
                          # curves below are averages over 1225 patients
K_MAX = 100               # x-axis extent; k=25 is the paper's operating point
LP_THREADS = 2
JOBS = len(SEEDS) * len(EPSILONS)


def true_duals(delta, capacities):
    """lambda* from the omniscient LP on the realized theta."""
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


def job(args):
    """One (eps, seed): the two decay profiles plus summary statistics."""
    epsilon, seed = args
    theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                               **DEFAULT_INSTANCE)
    theta = draw_realized_theta(theta_hat, epsilon, TRIAL + 1, seed)[TRIAL]

    live = np.flatnonzero(capacities > 0)
    M_live = len(live)
    delta = theta[:, live] - theta[:, DEFAULT_M:DEFAULT_M + 1]
    lam = true_duals(delta, capacities[live])
    r_pos = np.maximum(delta - lam[None, :], 0.0)

    kmax = min(K_MAX, M_live)

    # ---- patient side: H3 exactly ------------------------------------------
    # Sort each patient's positive reduced surpluses descending; r_sorted[:, k-1]
    # is r+_{i,(k)}, and column 0 is mu*_i.
    r_sorted = -np.sort(-r_pos, axis=1)[:, :kmax]
    active = r_sorted[:, 0] > 0            # H3 quantifies over mu*_i > 0 only
    r_act = r_sorted[active]
    gap = (r_act[:, [0]] - r_act) / epsilon
    support = (r_pos > 0).sum(axis=1)      # how many options are candidates at all

    # ---- provider side: the mirror (NOT H5) --------------------------------
    # For each live provider, sort the patients by r+ descending: how much
    # surplus is lost replacing its best candidate patient with its m-th.
    rp_sorted = -np.sort(-r_pos, axis=0)[:kmax, :]
    p_active = rp_sorted[0, :] > 0
    rp_act = rp_sorted[:, p_active]
    p_gap = (rp_act[[0], :] - rp_act) / epsilon
    # Capacity-weighted, echoing H5's sum_j cbar_j w_j: a provider with more
    # slots reaches further down its own candidate list, so its spread matters
    # proportionally more.
    w = capacities[live][p_active].astype(float)

    return dict(
        epsilon=epsilon, seed=seed,
        n_active=int(active.sum()), n_prov_active=int(p_active.sum()),
        M_live=M_live,
        patient_gap_mean=gap.mean(axis=0).tolist(),
        patient_gap_p95=np.percentile(gap, 95, axis=0).tolist(),
        patient_gap_max=gap.max(axis=0).tolist(),
        provider_gap_wmean=((p_gap * w[None, :]).sum(axis=1) / w.sum()).tolist(),
        support_frac=[float((support >= k).mean()) for k in range(1, kmax + 1)],
        support_median=float(np.median(support)),
    )


def main():
    print(f"N={DEFAULT_N} M={DEFAULT_M} | {len(SEEDS)} seeds x 1 realized theta "
          f"| eps={EPSILONS} | k up to {K_MAX}\n"
          f"lambda* = duals of the omniscient LP on the REALIZED theta; "
          f"c_j=0 providers dropped\n", flush=True)
    t0 = time.time()
    specs = [(e, s) for e in EPSILONS for s in SEEDS]
    recs = {e: [] for e in EPSILONS}
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futures = [ex.submit(job, s) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            recs[r["epsilon"]].append(r)
            print(f"  [{i}/{len(specs)}] ({time.time() - t0:.0f}s) "
                  f"eps={r['epsilon']} seed={r['seed']} "
                  f"active={r['n_active']} median support={r['support_median']:.0f}",
                  flush=True)

    def avg(e, key):
        return np.mean([r[key] for r in recs[e]], axis=0)

    ks = np.arange(1, len(avg(EPSILONS[0], "patient_gap_mean")) + 1)
    kcol = list(ks).index(DEFAULT_K)

    print("\n" + "=" * 94)
    print(f"H3: gamma required so that r+_(i,k) >= mu*_i - gamma*eps, at k={DEFAULT_K}")
    print("=" * 94)
    print(f"{'eps':>6s} {'mean':>9s} {'p95':>9s} {'max':>9s} "
          f"{'patients w/ mu*>0':>18s} {'median #cands':>14s}")
    for e in EPSILONS:
        print(f"{e:>6g} {avg(e, 'patient_gap_mean')[kcol]:9.3f} "
              f"{avg(e, 'patient_gap_p95')[kcol]:9.3f} "
              f"{avg(e, 'patient_gap_max')[kcol]:9.3f} "
              f"{np.mean([r['n_active'] for r in recs[e]]):18.0f} "
              f"{np.mean([r['support_median'] for r in recs[e]]):14.0f}")

    print("\n" + "=" * 94)
    print("MEAN GAP / eps, patient side (columns are k)")
    print("=" * 94)
    show = [1, 2, 5, 10, 25, 50, 100]
    show = [k for k in show if k <= ks[-1]]
    print(f"{'eps':>6s}" + "".join(f"{('k=' + str(k)):>9s}" for k in show))
    for e in EPSILONS:
        v = avg(e, "patient_gap_mean")
        print(f"{e:>6g}" + "".join(f"{v[k - 1]:9.3f}" for k in show))

    print("\nMEAN GAP / eps, provider side (capacity-weighted; columns are m)")
    print(f"{'eps':>6s}" + "".join(f"{('m=' + str(k)):>9s}" for k in show))
    for e in EPSILONS:
        v = avg(e, "provider_gap_wmean")
        print(f"{e:>6g}" + "".join(f"{v[k - 1]:9.3f}" for k in show))

    # ------------------------------------------------------------------ plot
    colors = color_schemes["six_color"]
    dims = (1, 3)
    fig, ax = create_axes(
        dims,
        {"figsize": (13, 2.8), "style_size": "paper", "hide_spines": True,
         "has_grid": True},
        x_labels=[["Rank $k$ (providers)", "Rank $m$ (patients)", "Rank $k$"]],
        y_labels=[["$(\\mu^*_i - r^+_{i,(k)})\\,/\\,\\epsilon$",
                   "$(r^+_{j,(1)} - r^+_{j,(m)})\\,/\\,\\epsilon$",
                   "Fraction of patients"]],
        titles=[["Patient side (H3)", "Provider side (mirror)",
                 "Candidates per patient"]],
    )
    for i, e in enumerate(EPSILONS):
        style = dict(color=colors[i], linewidth=2, marker=plot_markers[i % 6],
                     markevery=max(1, len(ks) // 8), markersize=6,
                     label=f"$\\epsilon={e:g}$")
        ax[0][0].plot(ks, avg(e, "patient_gap_mean"), **style)
        ax[0][1].plot(ks, avg(e, "provider_gap_wmean"), **style)
        ax[0][2].plot(ks, avg(e, "support_frac"), **style)
    for col in (0, 1):
        # k = 25 is the budget every headline number in the project uses.
        ax[0][col].axvline(DEFAULT_K, color="0.35", linewidth=1.0, linestyle=":")
    ax[0][2].axvline(DEFAULT_K, color="0.35", linewidth=1.0, linestyle=":")

    fig.subplots_adjust(wspace=0.32)
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, -0.14), fontsize=12)
    for ext in ("pdf", "png"):
        path = HERE / f"substitutability.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)

    json.dump({str(e): recs[e] for e in EPSILONS},
              open(HERE / "substitutability.json", "w"))
    print(f"\ndone ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
