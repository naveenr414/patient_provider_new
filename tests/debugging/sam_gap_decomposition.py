"""Two questions about SAM's gap to the omniscient benchmark.

1) DECOMPOSITION.  OPT - V splits into two economically distinct losses:

       OPT(theta)          omniscient: full information, no menu restriction
     - LP(X; theta)        best centralized matching RESTRICTED to the menus X
       ------------------  = MENU LOSS (the cost of committing to X before
                              seeing theta, and of the k-item budget)
       LP(X; theta)
     - V(X; theta, sigma)  what sequential self-interested arrivals actually get
       ------------------  = DECENTRALIZATION LOSS (arrival order + patients
                              choosing for themselves, not being assigned)

   LP(X; theta) is computed with the same omniscient LP, on theta masked to
   the menu (off-menu entries pushed below the exit utility so the LP never
   uses them, which is exactly "patient i may only be matched within X_i").

2) VARIANCE.  A crossed theta x sigma design (every theta paired with every
   sigma) so the variation in V attributable to the noise draw and to the
   arrival order can be separated by a two-way decomposition, instead of
   being confounded as they are in the production runs (where each trial
   redraws both at once).

Run:  PYTHONPATH=. python tests/debugging/sam_gap_decomposition.py
"""
import inspect
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import policies as P
from patient.simulator import Simulator, draw_realized_theta
from patient.utils import solve_omniscient_lp
from scripts.run_experiments import (
    build_instance, DEFAULT_INSTANCE, DEFAULT_N, DEFAULT_M, DEFAULT_K,
    DEFAULT_EPSILON, DEFAULT_S,
)

OUT = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]        # instance seeds (each has its own theta_hat + capacities)
N_THETA = 8              # noise draws per instance
N_SIGMA = 8              # arrival orders per instance, crossed with every theta
POLICIES = {
    "sam": (P.sam, dict(epsilon=DEFAULT_EPSILON, S=DEFAULT_S)),
    "offer_all": (P.offer_all, {}),
    "offer_one": (P.offer_one, {}),
}


def lp_value(theta, capacities):
    """Mean per-patient utility of the optimal centralized matching."""
    N, M = theta.shape[0], theta.shape[1] - 1
    assignment = solve_omniscient_lp(theta, capacities)
    matched = assignment.sum(axis=1) > 0.5
    # unmatched patients take their own exit utility (noise perturbs that column too)
    val = (assignment * theta[:, :M]).sum() + theta[~matched, M].sum()
    return val / N


def lp_value_restricted(theta, capacities, menus):
    """Same LP, but patient i may only be matched to providers in menus[i]."""
    masked = theta.copy()
    off_menu = menus == 0
    masked[:, :-1][off_menu] = -1.0     # strictly below the exit utility (>= 0)
    return lp_value(masked, capacities)


def simulate(theta, capacities, menus, sim_seed):
    """One (theta, sigma) evaluation of V: sequential arrivals against fixed
    menus. Mirrors `simulator.run_trials`' inner loop, but with the arrival
    order seeded independently of theta so the two can be crossed."""
    N, M = theta.shape[0], theta.shape[1] - 1
    sim = Simulator(theta, capacities, gamma=None, seed=sim_seed)
    sim.reset_initial()
    sim.reset_patient_order()
    total = 0.0
    for t in sim.patient_order:
        available = (sim.provider_capacities > 0).astype(int)
        menu = np.concatenate([menus[t] * available, [1]])
        chosen = sim.step(int(t), menu)
        total += sim.all_patients[t].theta_row[chosen]
    return total / N


def enforce_k(menus, k, rng):
    """The simulator's uniform menu-size budget (run_trials applies the same)."""
    menus = menus.copy()
    for i in np.flatnonzero(menus.sum(axis=1) > k):
        candidates = np.flatnonzero(menus[i])
        keep = rng.choice(candidates, size=k, replace=False)
        menus[i] = 0
        menus[i, keep] = 1
    return menus


def two_way_decomposition(V):
    """V: N_THETA x N_SIGMA. Returns the share of Var(V) from the theta main
    effect, the sigma main effect, and the interaction/residual."""
    grand = V.mean()
    row, col = V.mean(axis=1), V.mean(axis=0)
    ss_tot = ((V - grand) ** 2).sum()
    ss_theta = V.shape[1] * ((row - grand) ** 2).sum()
    ss_sigma = V.shape[0] * ((col - grand) ** 2).sum()
    # clip: when one factor has literally no effect (offer_one, where arrival
    # order cannot matter) the residual is a rounding-error-sized negative
    return (ss_theta / ss_tot, ss_sigma / ss_tot,
            max(1 - (ss_theta + ss_sigma) / ss_tot, 0.0), ss_tot)


def main():
    print(f"N={DEFAULT_N} M={DEFAULT_M} k={DEFAULT_K} epsilon={DEFAULT_EPSILON} "
          f"S={DEFAULT_S} | seeds={SEEDS} | {N_THETA} thetas x {N_SIGMA} sigmas "
          f"(crossed)\n")

    records = {p: [] for p in POLICIES}
    for seed in SEEDS:
        t0 = time.time()
        theta_hat, capacities, _ = build_instance(DEFAULT_N, DEFAULT_M, seed=seed,
                                                   **DEFAULT_INSTANCE)
        thetas = draw_realized_theta(theta_hat, DEFAULT_EPSILON, N_THETA, seed).astype(float)
        sigma_seeds = np.random.RandomState(10_000 + seed).randint(2 ** 31, size=N_SIGMA)

        # OPT depends only on theta, so it is shared across policies
        opt = np.array([lp_value(th, capacities) for th in thetas])

        for pname, (fn, kwargs) in POLICIES.items():
            rng = np.random.RandomState(seed)
            kw = dict(kwargs)
            if "seed" in inspect.signature(fn).parameters:
                kw.setdefault("seed", int(rng.randint(2 ** 31)))
            menus = enforce_k(fn(theta_hat, capacities, DEFAULT_K, **kw), DEFAULT_K, rng)

            lpx = np.array([lp_value_restricted(th, capacities, menus) for th in thetas])
            V = np.array([[simulate(th, capacities, menus, int(ss)) for ss in sigma_seeds]
                          for th in thetas])
            records[pname].append(dict(seed=seed, opt=opt, lpx=lpx, V=V,
                                        menu_size=menus.sum(axis=1).mean()))
            print(f"  [{pname:9s}] seed {seed}: OPT={opt.mean():.4f} "
                  f"LP(X)={lpx.mean():.4f} V={V.mean():.4f} "
                  f"menu={menus.sum(axis=1).mean():.1f}")
        print(f"  seed {seed} done in {time.time() - t0:.0f}s\n")

    # ------------------------------------------------------------ 1) decomposition
    print("=" * 78)
    print("1. DECOMPOSITION OF THE GAP TO OMNISCIENT  (mean per-patient utility)")
    print("=" * 78)
    print(f"{'policy':10s} {'OPT':>7s} {'LP(X)':>7s} {'V':>7s} | "
          f"{'OPT-V':>7s} {'menu':>7s} {'decent.':>8s} | {'menu%':>6s} {'dec%':>6s} "
          f"| {'V/OPT':>6s} {'LP(X)/OPT':>9s}")
    for pname, recs in records.items():
        opt = np.concatenate([r["opt"] for r in recs])
        lpx = np.concatenate([r["lpx"] for r in recs])
        V = np.concatenate([r["V"].mean(axis=1) for r in recs])   # avg over sigma
        gap, menu_loss, dec_loss = (opt - V).mean(), (opt - lpx).mean(), (lpx - V).mean()
        print(f"{pname:10s} {opt.mean():7.4f} {lpx.mean():7.4f} {V.mean():7.4f} | "
              f"{gap:7.4f} {menu_loss:7.4f} {dec_loss:8.4f} | "
              f"{100 * menu_loss / gap:5.1f}% {100 * dec_loss / gap:5.1f}% | "
              f"{V.mean() / opt.mean():6.4f} {lpx.mean() / opt.mean():9.4f}")

    # per-seed spread on the split, so the shares carry an error bar
    print("\nper-instance-seed menu-loss share of the total gap:")
    for pname, recs in records.items():
        shares = [((r["opt"] - r["lpx"]).mean()
                   / (r["opt"] - r["V"].mean(axis=1)).mean()) for r in recs]
        print(f"  {pname:10s} " + " ".join(f"{100 * s:5.1f}%" for s in shares))

    # --------------------------------------------------------------- 2) variance
    print("\n" + "=" * 78)
    print("2. VARIATION FROM theta VS FROM sigma  (crossed design, within instance)")
    print("=" * 78)
    print(f"{'policy':10s} {'sd(V)':>8s} {'sd_theta':>9s} {'sd_sigma':>9s} "
          f"{'sd_inter':>9s} | {'theta%':>7s} {'sigma%':>7s} {'inter%':>7s}")
    for pname, recs in records.items():
        rows = []
        for r in recs:
            V = r["V"]
            f_t, f_s, f_i, ss_tot = two_way_decomposition(V)
            sd = V.std()
            rows.append((sd, sd * np.sqrt(f_t), sd * np.sqrt(f_s), sd * np.sqrt(f_i),
                         f_t, f_s, f_i))
        m = np.array(rows).mean(axis=0)
        print(f"{pname:10s} {m[0]:8.5f} {m[1]:9.5f} {m[2]:9.5f} {m[3]:9.5f} | "
              f"{100 * m[4]:6.1f}% {100 * m[5]:6.1f}% {100 * m[6]:6.1f}%")

    # same question for the normalized ratio V/OPT, which is what the figures plot
    print("\nsame decomposition for V/OPT(theta) (the plotted quantity):")
    print(f"{'policy':10s} {'sd(V/OPT)':>10s} | {'theta%':>7s} {'sigma%':>7s} {'inter%':>7s}")
    for pname, recs in records.items():
        rows = []
        for r in recs:
            R = r["V"] / r["opt"][:, None]
            f_t, f_s, f_i, _ = two_way_decomposition(R)
            rows.append((R.std(), f_t, f_s, f_i))
        m = np.array(rows).mean(axis=0)
        print(f"{pname:10s} {m[0]:10.5f} | {100 * m[1]:6.1f}% {100 * m[2]:6.1f}% "
              f"{100 * m[3]:6.1f}%")

    # how much of the theta-driven movement in V is just OPT itself moving
    print("\nsd across theta of (averaged over sigma):")
    print(f"{'policy':10s} {'OPT':>8s} {'LP(X)':>8s} {'V':>8s} {'V/OPT':>8s} "
          f"{'corr(V,OPT)':>12s}")
    for pname, recs in records.items():
        rows = []
        for r in recs:
            Vt = r["V"].mean(axis=1)
            rows.append((r["opt"].std(), r["lpx"].std(), Vt.std(),
                         (Vt / r["opt"]).std(), np.corrcoef(Vt, r["opt"])[0, 1]))
        m = np.array(rows).mean(axis=0)
        print(f"{pname:10s} {m[0]:8.5f} {m[1]:8.5f} {m[2]:8.5f} {m[3]:8.5f} "
              f"{m[4]:12.3f}")

    np.savez(OUT / "sam_gap_decomposition.npz",
             **{f"{p}_{key}_{r['seed']}": r[key]
                for p, recs in records.items() for r in recs
                for key in ("opt", "lpx", "V")})
    print(f"\nsaved {OUT / 'sam_gap_decomposition.npz'}")


if __name__ == "__main__":
    main()
