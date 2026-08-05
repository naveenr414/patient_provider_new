"""A one-parameter family of small instances that slides SAM past offer-one.

`intuition_instances.py` found two hand-built extremes -- one where SAM loses
badly to offer-one (A) and one where it wins big (C) -- but nothing in
between, and no statement of what actually separates them. This script turns
that into a continuum controlled by a single number.

THE SLIDER
----------
    kappa = (a patient's top-1 minus top-2 value) / epsilon

i.e. how many noise-widths separate a patient's favourite provider from their
next-best one. Everything else about the instance is held fixed.

Why this is the right quantity, and not e.g. "how much capacity is scarce" or
"how big the crowd is": both policies are choosing what to commit to under a
theta that will be re-drawn. offer-one commits to the LP's single best guess.
SAM instead prices capacity and then offers everything that could plausibly
beat that price, which is exactly the pairs within ~epsilon of the dual. So

  * kappa << 1 -- the favourite is inside the noise band of the runner-up.
    theta_hat cannot tell them apart; whichever one is free when the patient
    arrives is nearly as good, and the realized argmax is a coin flip.
    Committing (offer-one) throws away that option value; hedging (SAM) is
    free, because the alternatives it adds are worth almost the same.

  * kappa >> 1 -- the favourite is unambiguous. theta_hat is right about who
    should get whom, and offer-one's LP assignment is close to optimal. SAM
    still adds the runner-ups (they clear the dual price in some scenario),
    and now every one of them is a strictly worse match that can arrive first
    and take the slot. Hedging becomes dilution.

So the crossover is a property of kappa alone, which is the claim this script
tests. It tests it by reaching the same kappa two different ways:

    sweep "gap"   : epsilon fixed at 0.10, the top-1/top-2 gap `a` varies
    sweep "noise" : the gap fixed at 0.10, epsilon varies

If kappa is really the governing quantity then the two sweeps must trace the
SAME curve when plotted against kappa -- if instead the answer depended on the
gap and the noise separately, they would separate. That is the falsifiable
part; a single knob relabelled is not a finding.

THE INSTANCE
------------
M=5 providers, one slot each. n_core=8 "core" patients -- deliberately more
than there are slots, so the marginal claimant on any provider is another core
patient and the dual price lands ABOVE the crowd. (Instance C's note: with as
many providers as core patients the marginal claimant becomes a crowd patient,
lambda collapses, and SAM lets the crowd straight back in -- a different
failure mode that would confound this one.) Plus n_crowd=6 crowd patients who
value everything at 0.30 against an exit of 0.20 -- low enough to be excluded
by the price, high enough that they WILL consume a slot if handed one, which
is what stops offer-everything from being a free win.

Each core patient gets a uniformly random favourite. The also-rans are exactly
tied with each other (up to a 0.5%-of-epsilon jitter that only breaks argsort
ties, so it never moves value), which is what makes kappa exactly the top-1
minus top-2 gap rather than some summary of a preference ladder.

Levels are chosen so theta +/- epsilon never reaches 0 or 1 at any grid point:
the simulator's clip would otherwise truncate the upside of exactly the
near-ties being studied, which is the effect that dominated the semi-synthetic
environment (see the clip note in the project memory).

TASK 2: SAM WITH A THRESHOLD
----------------------------
Algorithm 1 offers (i,j) iff m_ij > 0 where m_ij = mean_s max(Delta^s_ij -
lambda_bar_j, 0). Because the max sits INSIDE the average, a pair that clears
the price in one scenario out of S still gets a positive score -- so the
positivity filter barely filters. `sam[tau]` replaces the test with m_ij > tau.
Since a uniform shift cannot reorder a row, this only ever shortens menus; it
never changes which providers rank above which. tau = 0.01 and 0.02 are the
requested absolute values; tau = 0.1*epsilon is included because an absolute
tau cannot be right across a sweep that varies epsilon by 5x (m itself scales
with epsilon), and the "noise" sweep is precisely where that shows.

Nothing outside this folder is modified: the tau variant is implemented here
by recomputing SAM's own scores with the same seed derivation `run_trials`
uses, so tau=0 reproduces `policies.sam` menu-for-menu.

Run:  PYTHONPATH=. python tests/debugging/instance_family.py
      PYTHONPATH=. python tests/debugging/instance_family.py --quick
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient import policies as P
from patient.policies import _topk_mask
from patient.simulator import run_trials
from patient.utils import create_random_weights

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- instance
M = 5                 # providers, one slot each
N_CORE = 8            # > M on purpose: the marginal claimant must be a core patient
N_CROWD = 6
K = 3                 # menu budget, < M so the size constraint actually binds
MU = 0.55             # core patients' mean value across providers
CROWD_VALUE = 0.30    # above the exit, so a crowd patient will take a slot
EXIT_UTILITY = 0.20
S_SCENARIOS = 10

SEEDS = list(range(8))
NUM_TRIALS = 300

# kappa = a / epsilon reached two ways (see module docstring).
GAP_SWEEP_EPSILON = 0.10
GAP_SWEEP_KAPPA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0]
# The fixed gap has to be small enough that low kappa is reachable at an
# epsilon that does not clip: kappa = 0.3 needs epsilon = gap/0.3, and epsilon
# much above 1/6 pushes exit utility 0.20 - epsilon into the [0,1] clip.
NOISE_SWEEP_GAP = 0.05
NOISE_SWEEP_EPSILON = [0.1667, 0.125, 0.10, 0.0833, 0.0667, 0.05, 0.0333, 0.0167]

TAUS = [0.0, 0.01, 0.02, "0.1eps"]


def build(kappa, epsilon, seed):
    """theta_hat for a family member. `a = kappa * epsilon` is the amount by
    which a core patient's favourite beats each of their also-rans; the mean
    value across providers is held at MU regardless of a, so sliding kappa
    redistributes value rather than adding it."""
    rng = np.random.RandomState(1000 + seed)
    a = kappa * epsilon
    n = N_CORE + N_CROWD
    theta = np.empty((n, M + 1))

    # Jitter is 0.5% of epsilon: enough to give argsort a strict order (without
    # it every also-ran ties and offer_all/capacity_greedy pile every patient
    # onto the same low-index providers, an artifact of the tie-break rather
    # than of the instance), small enough to leave kappa the top-1/top-2 gap.
    jitter = rng.uniform(-0.005 * epsilon, 0.005 * epsilon, size=(N_CORE, M))
    theta[:N_CORE, :M] = MU - a / M + jitter
    favourite = rng.randint(0, M, size=N_CORE)
    theta[np.arange(N_CORE), favourite] += a

    theta[N_CORE:, :M] = CROWD_VALUE
    theta[:, M] = EXIT_UTILITY
    return theta, np.ones(M, dtype=int)


def instance_diagnostics(theta_hat, capacities, epsilon):
    """Quantities computable from theta_hat alone, to check the family behaves
    as designed and to give kappa a measured counterpart."""
    delta = theta_hat[:, :M] - theta_hat[:, M:M + 1]
    srt = np.sort(delta[:N_CORE], axis=1)[:, ::-1]
    top_gap = float((srt[:, 0] - srt[:, 1]).mean())
    return dict(
        measured_top1_top2=top_gap,
        measured_kappa=top_gap / epsilon,
        core_delta=float(delta[:N_CORE].mean()),
        crowd_delta=float(delta[N_CORE:].mean()),
    )


# ------------------------------------------------------------------- SAM
def _scenario_duals(delta, capacities):
    n, m_ = delta.shape
    mdl = gp.Model()
    mdl.Params.OutputFlag = 0
    x = mdl.addVars(n, m_, lb=0.0, ub=1.0)
    mdl.setObjective(gp.quicksum(delta[i, j] * x[i, j] for i in range(n) for j in range(m_)),
                     GRB.MAXIMIZE)
    for i in range(n):
        mdl.addConstr(gp.quicksum(x[i, j] for j in range(m_)) <= 1)
    cap = [mdl.addConstr(gp.quicksum(x[i, j] for i in range(n)) <= capacities[j])
           for j in range(m_)]
    mdl.optimize()
    return np.array([c.Pi for c in cap])


def sam_scores(theta_hat, capacities, epsilon, seed):
    """m_ij and lambda_bar exactly as `policies.sam` computes them.

    The seed derivation mirrors `run_trials`: it builds RandomState(seed),
    draws the realized thetas through a separate path that does not touch that
    stream, then takes one randint for the policy's own seed. Reproducing it
    here means tau=0 below is `policies.sam`, not an approximation of it."""
    rng = np.random.RandomState(np.random.RandomState(seed).randint(2 ** 31))
    scen = [create_random_weights(theta_hat, epsilon, rng) for _ in range(S_SCENARIOS)]
    deltas = [t[:, :M] - t[:, M:M + 1] for t in scen]
    lam = np.zeros(M)
    for d in deltas:
        lam += _scenario_duals(d, capacities) / S_SCENARIOS
    m = np.zeros((theta_hat.shape[0], M))
    for d in deltas:
        m += np.maximum(d - lam[None, :], 0) / S_SCENARIOS
    return m, lam


# --------------------------------------------------------------- one config
def policy_table(theta_hat, capacities, epsilon, seed):
    """name -> callable returning a menu matrix. SAM's scores are computed
    once and shared by every tau, so the tau comparison is paired down to the
    scenario draw and the extra taus cost no LPs."""
    m, lam = sam_scores(theta_hat, capacities, epsilon, seed)
    live = capacities > 0
    tbl = {
        "offer_one": (P.offer_one, {}, K),
        "offer_all": (P.offer_all, {}, K),
        "offer_everything": (P.offer_all, {}, M),
        "random": (P.random_menu, {}, K),
        "deferred_acceptance": (P.deferred_acceptance, {}, K),
        "capacity_greedy": (P.capacity_greedy, {}, K),
    }
    for tau in TAUS:
        t = 0.1 * epsilon if tau == "0.1eps" else float(tau)
        menus = _topk_mask(m - t, live, K)
        name = "sam" if tau == 0.0 else f"sam[tau={tau}]"
        tbl[name] = (lambda th, c, kk, _m=menus: _m.copy(), {}, K)
    return tbl, m, lam


def run_config(cfg):
    kappa, epsilon, seed, sweep = cfg["kappa"], cfg["epsilon"], cfg["seed"], cfg["sweep"]
    theta_hat, capacities = build(kappa, epsilon, seed)
    tbl, m, lam = policy_table(theta_hat, capacities, epsilon, seed)

    out = dict(cfg)
    out.update(instance_diagnostics(theta_hat, capacities, epsilon))
    out["lambda_mean"] = float(lam.mean())
    out["m_positive_frac"] = float((m > 0).mean())
    out["policies"] = {}

    omni = None
    for name, (fn, kwargs, kk) in tbl.items():
        res = run_trials(theta_hat, capacities, fn, kk, num_trials=NUM_TRIALS,
                         epsilon=epsilon, seed=seed, **kwargs)
        if omni is None:
            # theta realizations depend only on (theta_hat, epsilon, seed), which
            # every policy here shares, so one solve serves the whole config.
            omni = MET.omniscient_utility(res["theta_realized"], capacities)
        menus = res["menus"]
        out["policies"][name] = dict(
            utility=float(MET.utility(res["rewards"])),
            match_rate=float(MET.match_rate(res["chosen"], M)),
            menu_size=float(menus.sum(axis=1).mean()),
            # the diagnostic that drives the whole story: how many slots the
            # crowd is offered, i.e. how much of the scarce capacity is put
            # within reach of patients who should never get it.
            crowd_offers=float(menus[N_CORE:].sum(axis=1).mean()),
        )
    out["omniscient"] = float(omni)
    for v in out["policies"].values():
        v["normalized"] = v["utility"] / omni
    return out


# ------------------------------------------------------------------ report
def se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def aggregate(recs):
    """(sweep, kappa) -> per-policy mean/SE over seeds, plus the paired
    SAM-minus-offer-one difference. Paired is the number that matters: every
    policy sees identical seeds and identical trials, so the difference has a
    far smaller SE than the levels do."""
    keys = sorted({(r["sweep"], round(r["kappa"], 4)) for r in recs},
                  key=lambda t: (t[0], t[1]))
    names = list(recs[0]["policies"].keys())
    rows = []
    for sweep, kappa in keys:
        grp = sorted([r for r in recs if r["sweep"] == sweep
                      and round(r["kappa"], 4) == kappa], key=lambda r: r["seed"])
        row = dict(sweep=sweep, kappa=kappa, epsilon=grp[0]["epsilon"],
                   measured_kappa=float(np.mean([g["measured_kappa"] for g in grp])),
                   lambda_mean=float(np.mean([g["lambda_mean"] for g in grp])),
                   n_seeds=len(grp), policies={})
        for nm in names:
            norm = [g["policies"][nm]["normalized"] for g in grp]
            row["policies"][nm] = dict(
                normalized=float(np.mean(norm)), se=float(se(norm)),
                menu_size=float(np.mean([g["policies"][nm]["menu_size"] for g in grp])),
                crowd_offers=float(np.mean([g["policies"][nm]["crowd_offers"] for g in grp])),
                match_rate=float(np.mean([g["policies"][nm]["match_rate"] for g in grp])),
            )
        for nm in names:
            if nm == "offer_one":
                continue
            d = np.array([g["policies"][nm]["normalized"] for g in grp]) - \
                np.array([g["policies"]["offer_one"]["normalized"] for g in grp])
            row["policies"][nm]["vs_offer_one"] = float(d.mean())
            row["policies"][nm]["vs_offer_one_se"] = float(se(d))
            # In noise units. The entire menu-vs-commitment question is about
            # utility that only exists because theta moves by +/- epsilon, so
            # the raw difference is mechanically proportional to epsilon and
            # the two sweeps cannot possibly coincide in raw units -- they run
            # at epsilons 10x apart. Dividing by epsilon is what makes the
            # collapse test a test of kappa rather than of scale.
            row["policies"][nm]["vs_offer_one_per_eps"] = float(d.mean() / row["epsilon"])
            row["policies"][nm]["vs_offer_one_per_eps_se"] = float(se(d) / row["epsilon"])
        rows.append(row)
    return rows


def print_report(rows):
    show = ["offer_one", "sam", "sam[tau=0.01]", "sam[tau=0.02]", "sam[tau=0.1eps]",
            "offer_all", "offer_everything", "deferred_acceptance", "random"]

    for sweep, title in (("gap", f"SWEEP 1 -- gap varies, epsilon fixed at {GAP_SWEEP_EPSILON}"),
                         ("noise", f"SWEEP 2 -- epsilon varies, gap fixed at {NOISE_SWEEP_GAP}")):
        sub = [r for r in rows if r["sweep"] == sweep]
        if not sub:
            continue
        print("=" * 108)
        print(title)
        print(f"  M={M} providers x 1 slot | {N_CORE} core + {N_CROWD} crowd patients | "
              f"k={K} | {len(SEEDS)} seeds x {NUM_TRIALS} trials")
        print("=" * 108)
        print("normalized utility (against the per-realization omniscient LP)")
        hdr = f"{'kappa':>6s} {'eps':>6s} " + "".join(f"{n[:14]:>15s}" for n in show)
        print(hdr)
        for r in sub:
            line = f"{r['kappa']:>6.2f} {r['epsilon']:>6.3f} "
            for n in show:
                p = r["policies"][n]
                line += f"{p['normalized']:>9.4f}+-{p['se']:<4.3f}"
            print(line)

        print("\npaired difference vs offer-one (same seeds, same trials), "
              "in units of epsilon")
        pol = ["sam", "sam[tau=0.01]", "sam[tau=0.02]", "sam[tau=0.1eps]", "offer_all",
               "offer_everything"]
        print(f"{'kappa':>6s} {'eps':>6s} " + "".join(f"{n[:14]:>17s}" for n in pol))
        for r in sub:
            line = f"{r['kappa']:>6.2f} {r['epsilon']:>6.3f} "
            for n in pol:
                p = r["policies"][n]
                line += (f"{p['vs_offer_one_per_eps']:>+10.3f}"
                         f"+-{p['vs_offer_one_per_eps_se']:<5.3f}")
            print(line)

        print("\nmenu size / slots offered to the crowd (the over-offering diagnostic)")
        pol = ["offer_one", "sam", "sam[tau=0.01]", "sam[tau=0.02]", "sam[tau=0.1eps]",
               "offer_everything"]
        print(f"{'kappa':>6s} {'eps':>6s} " + "".join(f"{n[:14]:>17s}" for n in pol))
        for r in sub:
            line = f"{r['kappa']:>6.2f} {r['epsilon']:>6.3f} "
            for n in pol:
                p = r["policies"][n]
                line += f"{p['menu_size']:>9.2f} /{p['crowd_offers']:<6.2f}"
            print(line)
        print()

    # the crossover, read off each sweep independently
    print("=" * 108)
    print("CROSSOVER: the kappa at which SAM stops beating offer-one")
    print("=" * 108)
    for sweep in ("gap", "noise"):
        sub = sorted([r for r in rows if r["sweep"] == sweep], key=lambda r: r["kappa"])
        if not sub:
            continue
        for nm in ("sam", "sam[tau=0.01]", "sam[tau=0.02]", "sam[tau=0.1eps]"):
            xs = [r["kappa"] for r in sub]
            ys = [r["policies"][nm]["vs_offer_one"] for r in sub]
            cross = None
            for i in range(len(xs) - 1):
                if ys[i] > 0 >= ys[i + 1]:
                    t = ys[i] / (ys[i] - ys[i + 1])
                    cross = xs[i] + t * (xs[i + 1] - xs[i])
                    break
            tag = f"kappa* = {cross:.2f}" if cross is not None else (
                "no crossover in range (" + ("SAM ahead throughout" if ys[-1] > 0
                                             else "offer-one ahead throughout") + ")")
            print(f"   sweep={sweep:<6s} {nm:<16s} {tag}")
    print("\nIf kappa is the governing quantity, the two sweeps' kappa* agree; the "
          "'gap' and\n'noise' sweeps reach the same kappa from opposite directions "
          "and share no primitives.\n")


def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    styles = {"gap": ("o-", "tab:blue"), "noise": ("s--", "tab:orange")}

    ax = axes[0]
    for sweep, (mk, _) in styles.items():
        sub = sorted([r for r in rows if r["sweep"] == sweep], key=lambda r: r["kappa"])
        if not sub:
            continue
        for nm, col in (("sam", "tab:red"), ("offer_one", "tab:blue"),
                        ("offer_everything", "tab:green")):
            ax.errorbar([r["kappa"] for r in sub],
                        [r["policies"][nm]["normalized"] for r in sub],
                        yerr=[r["policies"][nm]["se"] for r in sub],
                        fmt=mk, color=col, ms=4, lw=1.4, capsize=2,
                        label=f"{nm} ({sweep})")
    ax.set_xlabel(r"$\kappa$ = (top-1 $-$ top-2) / $\epsilon$")
    ax.set_ylabel("normalized utility")
    ax.set_title("Levels")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # The collapse test: if kappa governs, the two sweeps lie on one curve.
    ax = axes[1]
    for sweep, (mk, col) in styles.items():
        sub = sorted([r for r in rows if r["sweep"] == sweep], key=lambda r: r["kappa"])
        if not sub:
            continue
        ax.errorbar([r["kappa"] for r in sub],
                    [r["policies"]["sam"]["vs_offer_one_per_eps"] for r in sub],
                    yerr=[r["policies"]["sam"]["vs_offer_one_per_eps_se"] for r in sub],
                    fmt=mk, color=col, ms=5, lw=1.6, capsize=2,
                    label=f"{sweep} sweep")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"(SAM $-$ offer-one) / $\epsilon$")
    ax.set_title(r"Collapse test: do both sweeps trace one curve in $\kappa$?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    cols = {"sam": "tab:red", "sam[tau=0.01]": "tab:purple",
            "sam[tau=0.02]": "tab:brown", "sam[tau=0.1eps]": "tab:cyan"}
    for sweep, (mk, _) in styles.items():
        sub = sorted([r for r in rows if r["sweep"] == sweep], key=lambda r: r["kappa"])
        if not sub:
            continue
        for nm, col in cols.items():
            ax.plot([r["kappa"] for r in sub],
                    [r["policies"][nm]["vs_offer_one_per_eps"] for r in sub],
                    mk, color=col, ms=4, lw=1.3,
                    label=f"{nm} ({sweep})" if sweep == "gap" else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"(policy $-$ offer-one) / $\epsilon$")
    # An absolute tau collapses off the bottom of this axis on the noise
    # sweep's small-epsilon end (down to -12): m scales with epsilon, so a
    # fixed 0.01 eventually exceeds nearly every score and empties the menus.
    # Clipping keeps the region where the taus actually differ readable; the
    # printed table carries the full numbers.
    ax.set_ylim(-0.75, 0.35)
    ax.set_title(r"Effect of the inclusion threshold $\tau$"
                 "\n(solid = gap sweep, dashed = noise sweep; axis clipped)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds x 40 trials on a 4-point kappa grid, for a smoke check")
    ap.add_argument("--jobs", type=int, default=0, help="0 = one process per config")
    args = ap.parse_args()

    global SEEDS, NUM_TRIALS
    gap_grid, noise_grid = GAP_SWEEP_KAPPA, NOISE_SWEEP_EPSILON
    if args.quick:
        SEEDS, NUM_TRIALS = [0, 1], 40
        gap_grid = [0.0, 0.5, 1.0, 3.0]
        noise_grid = [0.1667, 0.05, 0.0167]

    cfgs = [dict(sweep="gap", kappa=k, epsilon=GAP_SWEEP_EPSILON, seed=s)
            for k in gap_grid for s in SEEDS]
    cfgs += [dict(sweep="noise", kappa=NOISE_SWEEP_GAP / e, epsilon=e, seed=s)
             for e in noise_grid for s in SEEDS]

    jobs = args.jobs or min(len(cfgs), 60)
    print(f"{len(cfgs)} (kappa, seed) configs on {jobs} processes; "
          f"{len(SEEDS)} seeds x {NUM_TRIALS} trials each\n")
    with Pool(jobs) as pool:
        recs = pool.map(run_config, cfgs)

    rows = aggregate(recs)
    print_report(rows)

    out = HERE / ("instance_family_quick.json" if args.quick else "instance_family.json")
    out.write_text(json.dumps(dict(
        config=dict(M=M, n_core=N_CORE, n_crowd=N_CROWD, k=K, mu=MU,
                    crowd_value=CROWD_VALUE, exit_utility=EXIT_UTILITY,
                    S=S_SCENARIOS, seeds=SEEDS, num_trials=NUM_TRIALS),
        rows=rows), indent=2))
    print(f"wrote {out}")
    make_plot(rows, HERE / ("instance_family_quick.png" if args.quick
                            else "instance_family.png"))


if __name__ == "__main__":
    main()
