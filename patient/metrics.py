"""Outcome metrics (Section 6 / Figures 2-7, Table 1).

Operate on the outputs of `simulator.run_trials`: `chosen` (num_trials x N,
M = exit) and `rewards` (num_trials x N), the single planned `menus`
(N x M), and theta/capacities for the omniscient-LP normalizer.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from patient.utils import solve_omniscient_lp


def utility(rewards):
    """Mean realized utility per patient, averaged over trials (paper's
    primary outcome measure)."""
    return float(np.mean(rewards))


def omniscient_utility(theta, capacities):
    """Mean per-patient utility under the omniscient (perfect-info) LP
    benchmark, E_theta[OPT(theta)] (Section 4.1) -- the normalizer used in
    every figure.

    theta: either a single N x (M+1) realization, or a num_trials x N x (M+1)
        stack of them, in which case OPT is solved separately per realization
        and averaged. The per-realization form is the one the paper's
        benchmark actually calls for: OPT(theta) = max_X V_theta(X) is a
        function of the drawn theta, and averaging OPT over draws is strictly
        larger than OPT at the average draw (the omniscient policy can chase
        a favourable noise realization; that headroom is the regret in
        Theorem 2)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 3:
        return float(np.mean([omniscient_utility(t, capacities) for t in theta]))
    N, m_plus_1 = theta.shape
    M = m_plus_1 - 1
    assignment = solve_omniscient_lp(theta, capacities)
    assigned = assignment.sum(axis=1) > 0
    reward = np.where(assigned, (theta[:, :M] * assignment).sum(axis=1), theta[:, M])
    return float(reward.mean())


def menu_restricted_lp_utility(theta, capacities, menus):
    """E_theta[LP(X, theta)]: the best assignment achievable if the realized
    theta were known but the platform were still confined to the menus it
    already committed to. Mean per-patient utility, same units and same
    exit-option convention as `omniscient_utility`.

    This is the middle term of the additive-error decomposition

        OPT(theta) - V  =  [OPT(theta) - LP(X, theta)]  +  [LP(X, theta) - V]

    and both brackets are non-negative by construction. LP(X) is a maximum
    over a subset of the assignments OPT maximises over (only pairs with
    X_ij = 1), so OPT >= LP(X); and the sequential process's own outcome is
    itself a feasible point of LP(X) -- it never exceeds capacity and never
    matches outside the menu -- so LP(X) >= V.

    The two brackets separate the two distinct things a menu policy can get
    wrong. The first is COVERAGE: X may simply not contain the pairs a good
    assignment needs (offer_one's single column per patient is the extreme
    case; an unbounded offer_all, whose X is everything, drives this to zero).
    The second is COORDINATION: even when X contains a good assignment,
    patients arrive in a random order and each takes their own best available
    option, with no mechanism making them collectively pick that assignment.
    It vanishes exactly when the menu leaves patients no discretion that
    matters -- offer_one, whose every patient has one option that its LP
    already reserved capacity for, has LP(X) - V = 0 identically.

    theta: N x (M+1), or num_trials x N x (M+1), in which case the LP is
        solved per realization and averaged (as in `omniscient_utility`).
    menus: N x M 0/1, the committed menu (`run_trials`' 'menus'), shared
        across trials.

    Only the allowed (i, j) pairs become variables, so this is far cheaper
    than the unrestricted LP -- a k=25 menu gives 25N columns rather than MN.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 3:
        return float(np.mean([menu_restricted_lp_utility(t, capacities, menus)
                              for t in theta]))
    N, m_plus_1 = theta.shape
    M = m_plus_1 - 1
    allowed = [(i, j) for i in range(N) for j in np.flatnonzero(menus[i])]

    model = gp.Model("menu_restricted_lp")
    model.setParam("OutputFlag", 0)
    x = model.addVars(allowed, lb=0.0, ub=1.0, name="x")
    # Same form as `utils.solve_omniscient_lp`: a patient's baseline is the
    # exit option, and assigning them to j is worth the improvement over it.
    # Pairs worse than exiting therefore carry a negative coefficient and are
    # simply left unassigned, which is what makes "assigned or exited" come
    # out right without a separate exit variable.
    model.setObjective(
        gp.quicksum(theta[i, M] for i in range(N))
        + gp.quicksum((theta[i, j] - theta[i, M]) * x[i, j] for i, j in allowed),
        GRB.MAXIMIZE)
    by_patient, by_provider = {}, {}
    for i, j in allowed:
        by_patient.setdefault(i, []).append((i, j))
        by_provider.setdefault(j, []).append((i, j))
    for i, pairs in by_patient.items():
        model.addConstr(gp.quicksum(x[p] for p in pairs) <= 1)
    for j, pairs in by_provider.items():
        model.addConstr(gp.quicksum(x[p] for p in pairs) <= capacities[j])
    model.optimize()
    return float(model.ObjVal / N)


def normalized_utility(rewards, theta, capacities):
    """Realized utility (mean over trials & patients) as a fraction of the
    omniscient benchmark's per-patient utility. <= 1 in expectation since the
    omniscient LP sees each realized theta with no menu-size limit."""
    return utility(rewards) / omniscient_utility(theta, capacities)


def match_rate(chosen, num_providers):
    """Fraction of patients matched to a provider (not exit), averaged over
    trials. `chosen`: num_trials x N array, M = exit encoded as
    `num_providers`."""
    return float(np.mean(chosen != num_providers))


def choice_count(effective_menus):
    """Effective number of choices available to each patient at decision
    time: `menus` intersected with live capacity, so it accounts for
    mid-simulation depletion (a patient's initial menu may have had more
    providers than were actually still available when their turn came) --
    Figure 6 / RQ4.

    The exit option counts as a choice: it is implicitly offered to every
    patient (Section 3.1) and never depletes, so every patient has at least
    one. This is the reference implementation's convention, and it is the
    only one under which the paper's own comparison holds -- "SAM improves
    the number of choices available for patients ... 277% over offer-one"
    is unachievable if offer-one is credited with at most 1 provider while
    SAM is credited with more than 5.

    `effective_menus`: num_trials x N x M bool, from `simulator.run_trials`.
    Returns a num_trials x N array."""
    return effective_menus.sum(axis=-1) + 1


def choice_utility(effective_menus, theta_realized):
    """Ratio of the total utility of the offered (effective) options to the
    sum of the top-r values of theta for that patient, where r is the number
    of options actually offered -- i.e. the best possible utility achievable
    with r options (Section 6, RQ4). Measured against the realized theta
    (what the patient actually prefers), not theta_hat -- this metric is
    about how well the offered menu matches real preferences. Patients
    offered zero options get NaN (use `np.nanmean` to aggregate).

    effective_menus: num_trials x N x M bool.
    theta_realized: num_trials x N x (M+1), or a single N x (M+1) array
        broadcast over trials. The exit column is dropped."""
    theta = np.asarray(theta_realized, dtype=float)
    if theta.ndim == 2:
        theta = theta[np.newaxis]
    # The exit option is part of every menu and part of the comparison set,
    # matching `choice_count` -- so both the offered set and the top-r
    # reference are taken over all M+1 columns.
    n_trials = effective_menus.shape[0]
    offered = np.concatenate(
        [effective_menus, np.ones(effective_menus.shape[:-1] + (1,), dtype=bool)], axis=-1
    )
    r = offered.sum(axis=-1)  # num_trials x N, always >= 1
    offered_utility = np.where(offered, theta, 0.0).sum(axis=-1)

    sorted_theta = -np.sort(-theta, axis=-1)  # descending
    cum_top = np.cumsum(sorted_theta, axis=-1)
    cum_top = np.broadcast_to(cum_top, (n_trials,) + cum_top.shape[1:])
    m_plus_1 = theta.shape[-1]
    r_idx = np.clip(r, 1, m_plus_1) - 1
    best_r_utility = np.take_along_axis(cum_top, r_idx[..., None], axis=-1)[..., 0]

    return np.divide(
        offered_utility, best_r_utility,
        out=np.full_like(offered_utility, np.nan), where=best_r_utility > 0,
    )


def gini(values):
    """Gini coefficient of a nonnegative array (0 = perfectly equal)."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cum = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _rolling_mean(x, window):
    """Centered rolling mean with shrinking windows at the edges (same
    convention as pandas' `rolling(window, min_periods=1, center=True)`:
    position i averages x[i-(w-1)//2 : i+w//2+1], clipped to the array)."""
    n = len(x)
    lo, hi = (window - 1) // 2, window // 2
    return np.array([x[max(0, i - lo):min(n, i + hi + 1)].mean() for i in range(n)])


def zip_fairness(chosen, num_providers, patient_zips, rewards=None, window=25):
    """ZIP-code geographic fairness (Section 6.6 / Table 1).

    Returns a dict with keys `p25`, `gini_match_rate`, and (when `rewards` is
    given) `gini_utility`.

    p25 -- "the match rate for the area (ZIP code) with the 25th
        percentile population size" (Section 6.1). Note this is the match
        rate *of a particular ZIP code selected by population*, NOT the 25th
        percentile of the match-rate distribution: ZIP codes are sorted
        ascending by population, and we read off the match rate at the 25th
        percentile position of that ordering. Population is proxied by the
        number of sampled patients in the ZIP (patients are drawn
        proportional to ZIP population in `data_gen.semi_synthetic_theta`).
        Because a single ZIP's rate is noisy at this sample size, the
        rates are smoothed with a centered rolling mean of `window` ZIP
        codes before reading off the percentile, matching the reference
        implementation (`_legacy/scripts/notebooks/Plotting.ipynb`, which
        uses window=25).

    gini_match_rate / gini_utility -- Gini coefficient across ZIP codes
        (lower = more even). Section 6.6's prose defines it over "the match
        rates for different ZIP codes", while the reference implementation
        that produced Table 1 computes it over per-ZIP mean *utility* and
        calls it the "geographic Gini". Both are reported; they answer
        slightly different questions and the paper is ambiguous about which
        the table shows.

    chosen: num_trials x N, exit encoded as `num_providers`.
    rewards: optional num_trials x N realized utilities (needed for
        gini_utility).
    """
    patient_zips = np.asarray(patient_zips)
    matched = chosen != num_providers  # num_trials x N
    per_patient_rate = matched.mean(axis=0)  # N

    zips, counts = np.unique(patient_zips, return_counts=True)
    group_rates = np.array([per_patient_rate[patient_zips == z].mean() for z in zips])

    order = np.argsort(counts, kind="stable")  # ascending "population"
    smoothed = _rolling_mean(group_rates[order], window)

    out = {
        "p25": float(smoothed[len(smoothed) // 4]),
        "gini_match_rate": gini(group_rates),
    }
    if rewards is not None:
        per_patient_util = np.asarray(rewards).mean(axis=0)
        out["gini_utility"] = gini(
            np.array([per_patient_util[patient_zips == z].mean() for z in zips])
        )
    return out
