"""Three tiny hand-built instances that isolate when each policy wins or loses.

Everything here is small enough (N <= 12, M <= 6) to reason about by hand, and
the LPs solve instantly, so we can afford many trials and get tight SEs.

The mechanism behind all three is the same knob: SAM offers provider j to
patient i iff the scenario-averaged surplus m_ij = mean_s max(Delta^(s)_ij -
lambda_bar_j, 0) is positive, and lambda_bar_j is set by the MARGINAL patient
for provider j (roughly the (c_j+1)-th highest Delta). So:

  A. If the also-rans sit AT that margin, they all get a sliver of positive
     score and SAM over-offers -- it degenerates toward offer-all and a scarce
     high-value pair gets stolen by whoever arrives first. SAM << offer-one.
  B. If the also-rans sit FAR BELOW the margin, SAM's threshold excludes them
     cleanly, and it is offer-everything that hands them the scarce slots.
  C. If a patient's own top options are near-tied relative to epsilon AND
     there is a crowd of low-value patients to keep out, SAM needs both of its
     halves at once -- hedging (which offer-one lacks) and capacity pricing
     (which offer-all lacks). SAM >> everything.

Run:  PYTHONPATH=. python tests/debugging/intuition_instances.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import metrics as MET
from patient import policies as P
from patient.simulator import run_trials

SEEDS = [0, 1, 2, 3, 4]
NUM_TRIALS = 200


def instance_a():
    """A) SAM << offer-one. 'One gem among near-ties.'

    Provider 0 has a single slot and is worth 0.75 to patient 0 but 0.45 to
    everyone else. Because capacity is 1, lambda_0 is set by the SECOND-best
    claimant -- who is one of the 0.45 crowd -- so every one of them sits within
    epsilon of the price and picks up a small positive m. SAM therefore offers
    the gem to all 8, and patient 0 only gets it if they arrive first.
    Offer-one's LP hands provider 0 to patient 0 outright.

    Levels are kept below 1 - epsilon so the simulator's [0,1] clip never
    bites; clipping compresses the gem and mutes the effect."""
    N, M = 8, 2
    theta = np.full((N, M + 1), 0.0)
    theta[:, 0] = 0.45
    theta[0, 0] = 0.75
    theta[:, 1] = 0.40
    theta[:, M] = 0.0                      # exit
    return dict(name="A: SAM << offer-one", theta_hat=theta,
                capacities=np.array([1, 1]), k=2, epsilon=0.20,
                note="gem worth 0.75 to patient 0, 0.45 to the other 7; c=[1,1]")


def instance_b():
    """B) Offer-everything is catastrophic (Proposition 4's failure mode).

    Two providers, two patients who value them enormously (0.95/0.90), and a
    crowd of 10 patients who value everything at 0.12 -- barely above their
    exit of 0.05, so they WILL take a slot if handed one. Here lambda is set by
    the second high-value patient, i.e. it is high (~0.9), so the crowd's
    Delta = 0.07 is nowhere near it and SAM's threshold excludes them cleanly.
    Offer-everything hands all 12 patients both providers, and the crowd
    almost always gets there first."""
    N, M = 12, 2
    theta = np.full((N, M + 1), 0.12)
    theta[0, 0], theta[0, 1] = 0.95, 0.90
    theta[1, 0], theta[1, 1] = 0.95, 0.90
    theta[:, M] = 0.05                     # exit
    return dict(name="B: offer-everything collapses", theta_hat=theta,
                capacities=np.array([1, 1]), k=2, epsilon=0.05,
                note="2 high-value patients (0.95/0.90) vs a crowd of 10 at 0.12; exit 0.05")


def instance_c():
    """C) SAM >> everything. Needs BOTH of SAM's halves.

    Eight unit-capacity providers, near-tied for twelve 'core' patients (0.65
    for a patient's own, 0.60 for the rest), with epsilon = 0.30 large enough
    to reorder them -- so committing to one provider (offer-one) throws away
    the option value. Plus six 'crowd' patients at 0.20 who would happily
    consume a slot -- so offering everything throws the slots away.

    The capacity has to be SCARCER than the core group (8 slots, 12 core
    patients), and that is the whole trick. Then the marginal claimant on any
    provider is the 9th core patient, so lambda_j sits up near 0.5 and the
    crowd's Delta = 0.10 stays far below it even after epsilon of noise. With
    as many providers as core patients the LP matches all of them, the
    marginal claimant becomes a CROWD patient, lambda_j collapses to ~0.10,
    and SAM lets the crowd straight back in -- which is failure mode A.

    Levels sit below 1 - epsilon on purpose: at theta_hat = 0.85 with
    epsilon = 0.35 the [0,1] clip truncates the upside of exactly the
    near-ties SAM is hedging over, and SAM's margin drops from ~13% to ~2%."""
    N, M = 18, 8
    n_core = 12
    theta = np.full((N, M + 1), 0.20)
    for i in range(n_core):
        theta[i, :M] = 0.60
        theta[i, i % M] = 0.65
    theta[:, M] = 0.10                     # exit
    return dict(name="C: SAM >> everything", theta_hat=theta,
                capacities=np.ones(M, dtype=int), k=7, epsilon=0.30,
                note="8 slots, 12 core patients w/ near-tied options (0.65/0.60), 6 crowd at 0.20")


INSTANCES = [instance_a, instance_b, instance_c]


def se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v))


def run(inst):
    theta_hat, caps = inst["theta_hat"], inst["capacities"]
    N, M = theta_hat.shape[0], theta_hat.shape[1] - 1
    k, eps = inst["k"], inst["epsilon"]

    policies = {
        "offer_one": (P.offer_one, {}, k),
        "sam": (P.sam, dict(S=10), k),
        f"offer_all (k={k})": (P.offer_all, {}, k),
        f"offer_everything (k=M={M})": (P.offer_all, {}, M),
        "random": (P.random_menu, {}, k),
        "deferred_acceptance": (P.deferred_acceptance, {}, k),
        "capacity_greedy": (P.capacity_greedy, {}, k),
    }

    print("=" * 88)
    print(f"{inst['name']}   |   N={N} M={M} k={k} eps={eps} c={list(caps)}")
    print(f"   {inst['note']}")
    print("=" * 88)
    print(f"{'policy':28s} {'norm. util':>12s} {'SE':>7s} {'menu':>6s} "
          f"{'match':>7s} {'utility':>8s}")

    rows = {}
    for pname, (fn, kwargs, kk) in policies.items():
        per_seed_norm, per_seed_util, menus_sz, mrate = [], [], [], []
        for seed in SEEDS:
            out = run_trials(theta_hat, caps, fn, kk, num_trials=NUM_TRIALS,
                             epsilon=eps, seed=seed, **kwargs)
            omni = MET.omniscient_utility(out["theta_realized"], caps)
            u = MET.utility(out["rewards"])
            per_seed_norm.append(u / omni)
            per_seed_util.append(u)
            menus_sz.append(out["menus"].sum(axis=1).mean())
            mrate.append(MET.match_rate(out["chosen"], M))
        rows[pname] = np.mean(per_seed_norm)
        print(f"{pname:28s} {np.mean(per_seed_norm):12.4f} "
              f"{se(per_seed_norm):7.4f} {np.mean(menus_sz):6.2f} "
              f"{np.mean(mrate):7.4f} {np.mean(per_seed_util):8.4f}")

    best_base = max((v, kk) for kk, v in rows.items() if kk != "sam")
    print(f"\n   SAM {rows['sam']:.4f} vs best non-SAM {best_base[1]} "
          f"{best_base[0]:.4f}  ->  {100 * (rows['sam'] / best_base[0] - 1):+.1f}%")
    ev = [v for kk, v in rows.items() if kk.startswith("offer_everything")][0]
    print(f"   offer-everything {ev:.4f} vs offer-one {rows['offer_one']:.4f}  ->  "
          f"{100 * (ev / rows['offer_one'] - 1):+.1f}%\n")
    return rows


def main():
    print(f"{len(SEEDS)} seeds x {NUM_TRIALS} trials each; "
          f"normalized by the per-realization omniscient LP\n")
    for builder in INSTANCES:
        run(builder())


if __name__ == "__main__":
    main()
