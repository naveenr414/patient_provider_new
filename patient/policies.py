"""Menu-construction policies (paper Section 5, main text + EC.2.5 appendix).

Every policy takes `(theta_hat, capacities, k, **kwargs)`:
    theta_hat: N x (M+1) estimated utilities, last column is the exit utility
        (same convention as `utils.solve_omniscient_lp`).
    capacities: length-M starting provider capacities.
    k: max providers to offer per patient (the menu-size budget).
and returns an N x M 0/1 menu matrix (exit itself is never in the matrix — a
patient always has the option to exit, handled by the simulator).

Every policy restricts to `live = capacities > 0` before any scoring/top-k
step, so a provider that starts at zero capacity is never a phantom option in
a menu (the capacity=0 fix — see module docstring of the rewrite plan)."""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from patient.utils import create_random_weights, solve_omniscient_lp


def _topk_mask(scores, live, k, require_positive=True):
    """N x M 0/1 mask: per row, the top-k live entries by score, ties broken
    by score's own ordering. Non-live entries are never selected. If
    `require_positive`, non-positive scores are excluded too (even if that
    leaves fewer than k offered) -- this is SAM's Algorithm 1 thresholding
    specifically; plain top-k policies (offer_one, offer_all) always fill up
    to k live slots regardless of score sign, matching the legacy
    `greedy_policy`'s `np.argpartition`-based selection, which applies no
    positivity threshold."""
    N, M = scores.shape
    valid = live[None, :]
    if require_positive:
        valid = valid & (scores > 0)
    masked = np.where(valid, scores, -np.inf)
    k_eff = min(k, M)
    mask = np.zeros((N, M), dtype=int)
    if k_eff == 0:
        return mask
    top_idx = np.argsort(-masked, axis=1)[:, :k_eff]
    rows = np.repeat(np.arange(N), k_eff)
    cols = top_idx.flatten()
    valid = masked[rows, cols] > -np.inf
    mask[rows[valid], cols[valid]] = 1
    return mask


def random_menu(theta_hat, capacities, k, seed=None):
    """Uniform random k-subset of live providers per patient."""
    rng = np.random.RandomState(seed)
    N = theta_hat.shape[0]
    M = theta_hat.shape[1] - 1
    live = np.flatnonzero(capacities > 0)
    mask = np.zeros((N, M), dtype=int)
    k_eff = min(k, len(live))
    if k_eff == 0:
        return mask
    for i in range(N):
        chosen = rng.choice(live, size=k_eff, replace=False)
        mask[i, chosen] = 1
    return mask


def offer_one(theta_hat, capacities, k=1):
    """Offer-One (Section 4.4.1): reduce the problem to offline matching by
    offering each patient at most one provider. The paper is explicit that
    this "corresponds to adopting the noiseless strategy from Section 4.1"
    -- i.e. solve the perfect-information bipartite-assignment LP (Eq. 1) on
    theta_hat and offer each patient exactly the provider it assigns them
    (an empty menu if the LP leaves them unmatched, which happens when
    theta_hat_ij <= theta_hat_i,exit for every provider with spare capacity).

    This is capacity-aware, NOT an independent per-patient argmax: an
    independent top-1 would let many patients collide on the same popular
    provider, which is a much weaker policy than the paper analyses (its
    regret would not match the Theorem-2 lower bound the way Proposition 3
    says offer-one's does, and Lemma 4 explicitly refers to "the dual
    variable of the offer-one LP"). Matches the reference implementation's
    `lp_policy` (`_legacy/patient/lp_policies.py`), which the paper's own
    plotting code labels "Offer-One".

    Capacity-0 providers need no explicit filtering here: their capacity
    constraint sum_i x_ij <= 0 already forbids any assignment."""
    return solve_omniscient_lp(theta_hat, capacities).astype(int)


def offer_all(theta_hat, capacities, k):
    """Deterministic top-k live providers by theta_hat -- despite the name,
    'Offer-All' in the paper is a greedy baseline that offers the patient's k
    best-looking options (Section 6.5: "in all cases, k=25, meaning that the
    initial menus are restricted to at most 25 providers"), not literally
    every live provider. Matches the legacy `greedy_policy`."""
    M = theta_hat.shape[1] - 1
    live = capacities > 0
    return _topk_mask(theta_hat[:, :M], live, k, require_positive=False)


def sam(theta_hat, capacities, k, epsilon=0.1, S=10, seed=None):
    """Algorithm 1 (Scenario-Averaged Marginals). Sample S noisy scenarios,
    solve each scenario's LP relaxation for provider dual prices lambda,
    average the duals, then score each (i,j) by
        m_ij = mean_s[ max(Delta^(s)_ij - lambda_bar_j, 0) ]
    where Delta^(s)_ij = theta^(s)_ij - theta^(s)_i,exit, and offer each
    patient their top-k live providers with positive score."""
    rng = np.random.RandomState(seed)
    N = theta_hat.shape[0]
    M = theta_hat.shape[1] - 1
    live = capacities > 0

    scenarios = [create_random_weights(theta_hat, epsilon, rng) for _ in range(S)]
    deltas = [theta_s[:, :M] - theta_s[:, M:M + 1] for theta_s in scenarios]

    duals_accum = np.zeros(M)
    for delta_s in deltas:
        model = gp.Model()
        model.Params.OutputFlag = 0
        x = model.addVars(N, M, lb=0.0, ub=1.0, name="x")
        model.setObjective(
            gp.quicksum(delta_s[i, j] * x[i, j] for i in range(N) for j in range(M)),
            GRB.MAXIMIZE,
        )
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= 1)
        cap_constrs = [
            model.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= capacities[j])
            for j in range(M)
        ]
        model.optimize()
        duals_accum += np.array([c.Pi for c in cap_constrs]) / S

    scores = np.zeros((N, M))
    for delta_s in deltas:
        scores += np.maximum(delta_s - duals_accum[None, :], 0) / S

    return _topk_mask(scores, live, k)


def deferred_acceptance(theta_hat, capacities, k=None):
    """Patient-proposing Deferred Acceptance (Gale-Shapley), both sides
    ranking by theta_hat. Offers each patient only their matched provider
    (an empty menu if unmatched)."""
    N = theta_hat.shape[0]
    M = theta_hat.shape[1] - 1
    live = capacities > 0
    live_providers = np.flatnonzero(live)

    patient_prefs = [
        [j for j in np.argsort(-theta_hat[i, :M]) if live[j]] for i in range(N)
    ]
    provider_rank = np.argsort(np.argsort(-theta_hat[:, :M], axis=0), axis=0)
    caps = {j: max(int(capacities[j]), 0) for j in live_providers}

    next_proposal = np.zeros(N, dtype=int)
    held = {j: [] for j in live_providers}
    free = list(range(N))

    while free:
        proposals = {j: [] for j in live_providers}
        still_free = []
        for i in free:
            if next_proposal[i] < len(patient_prefs[i]):
                j = patient_prefs[i][next_proposal[i]]
                next_proposal[i] += 1
                proposals[j].append(i)
            # patients who run out of live providers to propose to stay unmatched
        for j in live_providers:
            if not proposals[j]:
                continue
            candidates = held[j] + proposals[j]
            candidates.sort(key=lambda i: provider_rank[i, j])
            held[j] = candidates[:caps[j]]
            still_free.extend(candidates[caps[j]:])
        free = [i for i in still_free if next_proposal[i] < len(patient_prefs[i])]

    mask = np.zeros((N, M), dtype=int)
    for j in live_providers:
        for i in held[j]:
            mask[i, j] = 1
    return mask


def capacity_greedy(theta_hat, capacities, k):
    """EC.2.5 capacity-aware greedy: process patients in decreasing order of
    their best available utility; each offered their top-k live providers
    with remaining capacity, then tentatively assigned to their best
    available option to decrement remaining capacity."""
    N = theta_hat.shape[0]
    M = theta_hat.shape[1] - 1
    remaining = np.maximum(capacities.astype(int), 0).copy()
    order = np.argsort(-theta_hat[:, :M].max(axis=1))
    mask = np.zeros((N, M), dtype=int)
    k_eff = min(k, M)

    for i in order:
        live = remaining > 0
        scores = np.where(live, theta_hat[i, :M], -np.inf)
        top_idx = np.argsort(-scores)[:k_eff]
        top_idx = top_idx[scores[top_idx] > -np.inf]
        mask[i, top_idx] = 1
        if len(top_idx) > 0:
            remaining[top_idx[0]] -= 1

    return mask


def exact_saa_milp(theta_hat, capacities, k, epsilon=0.1, S=10, seed=None,
                    time_limit=None, mip_gap=1e-4, threads=0):
    """Ground-truth SAA-MILP (Eq. 2), small-N only (Figure 3-left). A single
    shared menu X is chosen to maximize expected realized utility across S
    noisy scenarios, each with its own random patient arrival order and exact
    sequential-capacity tracking, with each patient rationally choosing their
    best available (offered AND still-live) option in that scenario.

    Three things keep this tractable; without them S=25 at N=20 does not
    finish in reasonable time:

    1. The rational-choice constraint carries no index over the *chosen*
       provider. "Patient i does at least as well as any option they were
       offered and that was still available" is a statement about z alone --
       z >= theta_l whenever X_il = 1 and l is live at their turn -- so it
       needs one constraint per (patient, l, scenario), not one per
       (patient, chosen j, alternative l, scenario). That is a factor of M
       fewer big-M rows.
    2. The big-M is tightened from a blanket 1.0 to theta_il - theta_i,exit,
       which is exactly tight: z is always at least the exit utility (a
       patient can always walk away), so that is the largest slack the
       constraint ever needs. Pairs with theta_il <= theta_i,exit are dropped
       entirely -- z >= theta_i,exit already implies them.
    3. Remaining capacity is a linear expression over earlier choices rather
       than its own variable, removing N*M*S continuous variables and their
       defining equalities.

    The formulation is verified exact: `tests/debugging/milp_bruteforce_check.py`
    enumerates every admissible menu on four small instances and confirms the
    MILP's menu attains the brute-force optimum of the same SAA objective.

    time_limit / mip_gap bound the search. If the solver stops early the best
    incumbent is returned; a warning is printed with the gap actually
    achieved so a truncated solve is visible rather than silent. THAT WARNING
    MATTERS: a truncated solve is not the optimum, and since this policy is
    the denominator of the approximation-ratio figure, a truncated denominator
    shows up as heuristics scoring above 1.0. Check the log for
    "exact_saa_milp stopped" after any run that feeds that figure.

    threads defaults to 0 (Gurobi's "use every available core"). It was
    previously pinned at 1, which is what caused 41 of 54 solves in the
    9-seed sweep to hit the 300 s limit with a median gap of 6% -- the
    branch-and-bound, not the formulation, was the bottleneck."""
    rng = np.random.RandomState(seed)
    N = theta_hat.shape[0]
    M = theta_hat.shape[1] - 1
    live = capacities > 0
    caps = np.maximum(np.asarray(capacities, dtype=int), 0)

    scenarios = []
    for s in range(S):
        theta_s = create_random_weights(theta_hat, epsilon, rng)
        order = rng.permutation(N)
        scenarios.append((theta_s, order))

    model = gp.Model("exact_saa_milp")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = mip_gap
    model.Params.Threads = threads
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    X = model.addVars(N, M, vtype=GRB.BINARY, name="X")
    y = model.addVars(N, M + 1, S, vtype=GRB.BINARY, name="y")
    z = model.addVars(N, S, lb=0.0, ub=1.0, name="z")
    # b[t, j, s] = 1 iff provider j still has capacity when the patient in
    # position t of scenario s's arrival order takes their turn.
    b = model.addVars(N, M, S, vtype=GRB.BINARY, name="b")

    for j in range(M):
        if not live[j]:
            model.addConstr(gp.quicksum(X[i, j] for i in range(N)) == 0)
    for i in range(N):
        model.addConstr(gp.quicksum(X[i, j] for j in range(M)) <= k)

    for s, (theta_s, order) in enumerate(scenarios):
        for i in range(N):
            model.addConstr(gp.quicksum(y[i, j, s] for j in range(M + 1)) == 1)
            model.addConstr(
                z[i, s] == gp.quicksum(theta_s[i, j] * y[i, j, s] for j in range(M + 1))
            )
            # exit is always offered and never depletes, so this is the
            # l = M+1 case of rational choice -- and it is what makes the
            # tightened big-M below valid.
            model.addConstr(z[i, s] >= theta_s[i, M])

        for t in range(N):
            patient = order[t]
            for j in range(M):
                remaining = caps[j] - gp.quicksum(y[order[tp], j, s] for tp in range(t))
                model.addConstr(b[t, j, s] <= remaining)
                model.addConstr(remaining <= caps[j] * b[t, j, s])
                model.addConstr(y[patient, j, s] <= X[patient, j])
                model.addConstr(y[patient, j, s] <= b[t, j, s])

            for l in range(M):
                big_m = theta_s[patient, l] - theta_s[patient, M]
                if big_m <= 0:
                    continue  # implied by z >= theta_exit
                model.addConstr(
                    z[patient, s] >= theta_s[patient, l]
                    - big_m * (1 - X[patient, l]) - big_m * (1 - b[t, l, s])
                )

    model.setObjective(gp.quicksum(z[i, s] for i in range(N) for s in range(S)), GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(
            f"exact_saa_milp found no feasible solution (Gurobi status {model.status}). "
            "Returning an empty menu here would silently look like a policy that "
            "offers nobody anything, which scores near the exit utility and is easy "
            "to mistake for a real result."
        )
    if model.status != GRB.OPTIMAL:
        print(f"warning: exact_saa_milp stopped at status {model.status} with MIP gap "
              f"{model.MIPGap:.4f} after {model.Runtime:.0f}s -- returning best incumbent")

    mask = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            mask[i, j] = int(round(X[i, j].X))
    return mask
