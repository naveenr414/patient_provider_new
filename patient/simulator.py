"""Sequential-arrival simulator (Section 3 of the paper).

theta_hat is the platform's estimate: fixed, known, and the ONLY thing menus
are planned from. The true utilities are a random variable around it,
theta = theta_hat + delta with ||delta||_inf <= epsilon (Section 3.1), and
the objective is an expectation over BOTH that draw and the arrival order:

    R(X) := E_{theta ~ P_theta, sigma ~ D}[ V_theta(X; sigma) ]

so each trial draws its own theta realization as well as its own
permutation -- Section 6.1: "each seed averages across 25 random
permutations of sigma and noise values". Menus are computed once, in a
batch, from theta_hat and the providers' starting capacities; patients then
arrive one at a time and choose from their precomputed menu (intersected
with whichever providers still have live capacity) according to that
trial's realized theta.

This ordering matters and is not interchangeable with "fix theta, perturb
theta_hat": the omniscient benchmark is OPT(theta) = max_X V_theta(X) on the
*realized* theta, so it can capture the upside of a favourable delta draw
(top-c_j patients by realized noise) while an algorithm committed to a menu
before seeing delta cannot. That gap is precisely the epsilon * sum_j c_j
regret of Theorem 2, and it is what menus hedge against."""
import inspect

import numpy as np

from patient.utils import create_random_weights


class Patient:
    def __init__(self, theta_row, idx):
        self.theta_row = theta_row  # length M+1, last entry is the exit utility
        self.idx = idx

    def get_random_outcome(self, menu, rng, gamma=None):
        """menu: 0/1 vector of length M+1 (menu[-1] = 1, exit always available).

        gamma=None -> deterministic argmax over offered options (the paper's
            main model, Section 3.1).
        gamma=<float> -> MNL / multinomial-logit choice (Figure 5-left), with
            gamma as the logit temperature tau:
                P(j) = exp(theta_ij / tau) / sum_{j' offered or exit} exp(...)
            The exit option's utility is theta_i,M+1 as usual, so the only
            extra parameter is tau. tau -> 0 recovers the deterministic
            argmax; larger tau is noisier choice. The paper never states a
            value for tau (see `run_experiments.DEFAULT_MNL_TEMPERATURE`).

        Returns the chosen option's index (M = exit)."""
        M = len(menu) - 1
        masked = np.where(menu == 1, self.theta_row, -np.inf)
        if gamma is None:
            return int(np.argmax(masked))

        # softmax over offered options, shifted for numerical stability;
        # exit is always in the support so the denominator is never zero.
        logits = masked / gamma
        logits -= logits.max()
        probs = np.exp(logits)
        return int(rng.choice(M + 1, p=probs / probs.sum()))


class Simulator:
    """One trial's worth of sequential patient arrivals against fixed menus."""

    def __init__(self, theta, capacities, gamma=None, seed=None):
        """theta: N x (M+1) realized utilities (last column = exit).
        capacities: length-M starting provider capacities."""
        self.theta = np.asarray(theta)
        self.num_patients, m_plus_1 = self.theta.shape
        self.num_providers = m_plus_1 - 1
        self.provider_max_capacities = np.asarray(capacities, dtype=int)
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.all_patients = [Patient(self.theta[i], i) for i in range(self.num_patients)]

    def reset_initial(self):
        self.provider_capacities = self.provider_max_capacities.copy()

    def reset_patient_order(self):
        self.patient_order = self.rng.permutation(self.num_patients)

    def step(self, patient_idx, menu):
        """menu: 0/1 vector of length M+1. Decrements capacity on a match.
        Returns the chosen option's index (M = exit)."""
        chosen = self.all_patients[patient_idx].get_random_outcome(menu, self.rng, gamma=self.gamma)
        if chosen < self.num_providers:
            self.provider_capacities[chosen] -= 1
        return chosen


def draw_realized_theta(theta_hat, epsilon, num_trials, seed, with_trial_seeds=False):
    """The `num_trials` realized theta draws for one instance/seed, as a
    num_trials x N x (M+1) float32 array.

    Drawn from a stream seeded only by `seed`, and (in `run_trials`) consumed
    before any policy-dependent randomness, so two policies run at the same
    seed face bit-identical trials -- a paired comparison, as in the
    reference implementation, which reseeds on the trial index right before
    drawing each trial's theta and arrival order. It also means the
    omniscient normalizer is policy-independent and can be computed once per
    seed (`run_experiments.omniscient_for_seed`).

    with_trial_seeds also returns the per-trial simulator seeds (arrival
    order + any choice-model randomness) drawn from the same stream."""
    theta_hat = np.asarray(theta_hat)
    N, m_plus_1 = theta_hat.shape
    env_rng = np.random.RandomState(np.random.RandomState(seed).randint(2 ** 31))
    all_theta = np.zeros((num_trials, N, m_plus_1), dtype=np.float32)
    trial_seeds = np.zeros(num_trials, dtype=np.int64)
    for trial in range(num_trials):
        all_theta[trial] = (create_random_weights(theta_hat, epsilon, env_rng)
                            if epsilon > 0 else theta_hat)
        trial_seeds[trial] = env_rng.randint(2 ** 31)
    return (all_theta, trial_seeds) if with_trial_seeds else all_theta


def run_trials(theta_hat, capacities, policy_fn, k, num_trials=25,
                epsilon=0.0, gamma=None, seed=None, **policy_kwargs):
    """Plan a menu once from (theta_hat, capacities) via `policy_fn`, then run
    `num_trials` independent trials against it. Each trial draws its own
    realized theta = clip(theta_hat + Uniform(-epsilon, epsilon), 0, 1) AND
    its own patient arrival order, then replays patients sequentially against
    the fixed menu intersected with live availability (Section 6.1: 25
    permutations of sigma and noise values per seed).

    theta_hat: N x (M+1), the platform's estimate (last column = exit
        utility) -- used ONLY for menu planning, never as the realized
        utility a patient responds to.
    policy_fn: (theta_hat, capacities, k, **policy_kwargs) -> N x M 0/1 menu
        matrix; called once with the full theta_hat (exit column included, so
        policies that need Delta_ij = theta_ij - theta_i,exit can compute it,
        e.g. `sam`) and the starting capacities. The menu-size budget k is
        enforced here regardless of what the policy itself returns: any row
        offering more than k providers is randomly subsampled down to k
        (e.g. `offer_all` offers every live provider with no self-imposed
        cap; the k budget is a simulator-level constraint applied uniformly,
        not each policy's own responsibility). If `policy_fn` accepts a
        `seed` and/or `epsilon` parameter and `policy_kwargs` didn't already
        supply one, they default to this call's own `seed` (a value derived
        from it, kept separate from the trial loop's own randomness) and
        `epsilon` respectively -- e.g. SAM's scenario-noise level defaults to
        this call's `epsilon`, matching the "algorithm knows the true noise
        level" model assumption (the same epsilon that generated theta_hat
        from theta_true), unless the caller deliberately wants to study a
        mis-specified epsilon.

    Returns dict: 'menus' (N x M, the single planned menu matrix), 'chosen'
        (num_trials x N array, M = exit), 'rewards' (num_trials x N array of
        realized utility), 'effective_menus' (num_trials x N x M bool: the
        offered set actually available at each patient's decision time, i.e.
        `menus` intersected with live capacity -- accounts for
        mid-simulation depletion, used by `metrics.choice_count`/
        `choice_utility` per the paper's RQ4 defs), and 'theta_realized'
        (num_trials x N x (M+1) float32: each trial's drawn theta, needed by
        the omniscient normalizer and choice-utility metrics, both of which
        the paper defines against the realized theta)."""
    rng = np.random.RandomState(seed)
    theta_hat = np.asarray(theta_hat)
    N, M = theta_hat.shape[0], theta_hat.shape[1] - 1

    all_theta, trial_seeds = draw_realized_theta(theta_hat, epsilon, num_trials, seed,
                                                  with_trial_seeds=True)

    policy_params = inspect.signature(policy_fn).parameters
    extra = {}
    if "seed" not in policy_kwargs and "seed" in policy_params:
        extra["seed"] = int(rng.randint(2 ** 31))
    if "epsilon" not in policy_kwargs and "epsilon" in policy_params:
        extra["epsilon"] = epsilon
    menus = policy_fn(theta_hat, np.asarray(capacities, dtype=int), k, **policy_kwargs, **extra)

    # Enforce the menu-size budget k uniformly across every policy: if a
    # policy proposes more than k providers (e.g. offer_all, which offers
    # every live provider with no self-imposed cap), randomly subsample down
    # to exactly k. Matches the legacy simulator's max_shown truncation
    # (_legacy/patient/simulator.py:326-332) -- every policy is subject to
    # the same k budget, not just the ones that already self-limit.
    row_counts = menus.sum(axis=1)
    for i in np.flatnonzero(row_counts > k):
        candidates = np.flatnonzero(menus[i])
        keep = rng.choice(candidates, size=k, replace=False)
        menus[i] = 0
        menus[i, keep] = 1

    all_chosen = np.zeros((num_trials, N), dtype=int)
    all_rewards = np.zeros((num_trials, N))
    all_effective_menus = np.zeros((num_trials, N, M), dtype=bool)
    for trial in range(num_trials):
        sim = Simulator(all_theta[trial], capacities, gamma=gamma, seed=int(trial_seeds[trial]))
        sim.reset_initial()
        sim.reset_patient_order()

        for t in sim.patient_order:
            available = (sim.provider_capacities > 0).astype(int)
            effective = menus[t] * available
            all_effective_menus[trial, t] = effective.astype(bool)
            menu = np.concatenate([effective, [1]])
            chosen = sim.step(int(t), menu)
            all_chosen[trial, t] = chosen
            all_rewards[trial, t] = sim.all_patients[t].theta_row[chosen]

    return {
        "menus": menus,
        "chosen": all_chosen,
        "rewards": all_rewards,
        "effective_menus": all_effective_menus,
        "theta_realized": all_theta,
    }
