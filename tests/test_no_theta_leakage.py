"""Information-flow test: the platform's estimate theta_hat is the ONLY
utility information any policy sees, and the realized theta each patient
responds to is a fresh draw around it.

Three claims, checked mechanically rather than by reading the code:

  1. theta_hat is the EC.1 formula's output with the exit column appended --
     no noise has been applied to it.
  2. theta = clip(theta_hat + Uniform(-eps, eps), 0, 1), drawn independently
     per trial (Section 3.1 / Section 6.1).
  3. No policy -- SAM included -- ever receives a realized theta, and its
     menus are a deterministic function of theta_hat alone.

Claim 3 is the one worth testing rather than asserting: SAM draws its own S
scenarios internally, and a plausible-looking bug would be for those to be
seeded from, or replaced by, the realized draws.

Run with: python tests/test_no_theta_leakage.py
"""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patient import data_gen
from patient import policies as P
from patient.simulator import draw_realized_theta, run_trials
from scripts.run_experiments import (DEFAULT_INSTANCE, EC25_POLICIES, build_instance,
                                     _policy_kwargs)

N, M, K, EPSILON, SEED, TRIALS, S = 40, 25, 6, 0.25, 3, 4, 3


def test_theta_hat_is_the_formula():
    """Claim 1: build_instance returns the EC.1 formula's theta verbatim."""
    theta_hat, _caps, _pat = build_instance(N, M, seed=SEED, **DEFAULT_INSTANCE)
    raw, _, _ = data_gen.semi_synthetic_theta(
        N, M,
        average_distance=DEFAULT_INSTANCE["average_distance"],
        omega=DEFAULT_INSTANCE["omega"],
        alpha=DEFAULT_INSTANCE["alpha"],
        clip_distance_term=DEFAULT_INSTANCE["clip_distance_term"],
        seed=SEED,
    )
    assert np.array_equal(theta_hat[:, :M], raw), "theta_hat is not the raw formula output"
    assert np.all(theta_hat[:, M] == DEFAULT_INSTANCE["exit_utility"]), "exit column wrong"
    print("PASS: theta_hat is the EC.1 formula's output, unmodified, plus the exit column")


def test_realized_theta_is_theta_hat_plus_noise():
    """Claim 2: every realization sits within epsilon of theta_hat (up to the
    [0,1] clip), and the trials differ from one another."""
    theta_hat, _caps, _ = build_instance(N, M, seed=SEED, **DEFAULT_INSTANCE)
    theta = draw_realized_theta(theta_hat, EPSILON, TRIALS, SEED)

    assert theta.shape == (TRIALS, N, M + 1)
    for t in range(TRIALS):
        dev = theta[t] - theta_hat
        # the clip only ever pulls a value back toward [0,1], so |delta| <= eps
        assert np.abs(dev).max() <= EPSILON + 1e-6, f"trial {t} deviates by more than epsilon"
        assert np.all(theta[t] >= 0) and np.all(theta[t] <= 1), "theta escaped [0,1]"
        assert not np.array_equal(theta[t], theta_hat), f"trial {t} was not perturbed at all"
    for t in range(1, TRIALS):
        assert not np.array_equal(theta[t], theta[0]), "trials share a theta realization"

    # ... and the noise is centred, not one-sided
    mean_dev = float(np.mean(theta - theta_hat[None]))
    assert abs(mean_dev) < 0.05, f"noise looks biased (mean deviation {mean_dev:.4f})"
    print(f"PASS: theta = clip(theta_hat + U(-{EPSILON}, {EPSILON})), redrawn per trial, "
          f"mean deviation {mean_dev:+.4f}")


def test_policies_only_ever_see_theta_hat():
    """Claim 3: record the exact array handed to each policy and confirm it is
    theta_hat, not any realization."""
    theta_hat, caps, _ = build_instance(N, M, seed=SEED, **DEFAULT_INSTANCE)
    realized = draw_realized_theta(theta_hat, EPSILON, TRIALS, SEED)

    for name, (policy_fn, extra) in EC25_POLICIES.items():
        seen = []

        def spy(theta_arg, capacities, k, _fn=policy_fn, **kwargs):
            seen.append(np.array(theta_arg, copy=True))
            return _fn(theta_arg, capacities, k, **kwargs)

        # run_trials inspects the signature to decide what to inject, so the
        # spy has to advertise the same parameters as the policy it wraps.
        spy.__signature__ = inspect.signature(policy_fn)

        run_trials(theta_hat, caps, spy, K, num_trials=TRIALS, epsilon=EPSILON,
                   seed=SEED, **_policy_kwargs(name, extra, EPSILON, S))

        assert len(seen) == 1, f"{name}: policy called {len(seen)} times, expected once"
        assert np.array_equal(seen[0], theta_hat), f"{name}: was handed something other than theta_hat"
        for t in range(TRIALS):
            assert not np.array_equal(seen[0], realized[t]), f"{name}: was handed realization {t}"
    print(f"PASS: all {len(EC25_POLICIES)} policies receive theta_hat and nothing else")


def test_menus_do_not_depend_on_realized_theta():
    """Claim 3, structurally: the menu must be reproducible by calling the
    policy on theta_hat alone, outside the simulator entirely, and must not
    move when the number of realized draws changes. This catches leakage a
    spy would miss -- e.g. a policy reading the realized draws through shared
    RNG state rather than through its theta argument."""
    theta_hat, caps, _ = build_instance(N, M, seed=SEED, **DEFAULT_INSTANCE)
    for name, (policy_fn, extra) in EC25_POLICIES.items():
        pkwargs = _policy_kwargs(name, extra, EPSILON, S)
        a = run_trials(theta_hat, caps, policy_fn, K, num_trials=TRIALS,
                       epsilon=EPSILON, seed=SEED, **pkwargs)
        b = run_trials(theta_hat, caps, policy_fn, K, num_trials=TRIALS + 5,
                       epsilon=EPSILON, seed=SEED, **pkwargs)
        assert b["theta_realized"].shape[0] == TRIALS + 5, "extra realizations were not drawn"
        assert np.array_equal(a["menus"], b["menus"]), \
            f"{name}: menus changed when the realized-theta draws changed"

        # and the menu is reproducible from theta_hat alone, outside run_trials
        direct_kwargs = dict(pkwargs)
        params = inspect.signature(policy_fn).parameters
        if "epsilon" in params and "epsilon" not in direct_kwargs:
            direct_kwargs["epsilon"] = EPSILON
        if "seed" in params and "seed" not in direct_kwargs:
            direct_kwargs["seed"] = int(np.random.RandomState(SEED).randint(2 ** 31))
        direct = policy_fn(theta_hat, np.asarray(caps, dtype=int), K, **direct_kwargs)
        assert np.array_equal(a["menus"], direct), \
            f"{name}: run_trials' menu differs from calling the policy on theta_hat directly"
    print(f"PASS: menus are a function of theta_hat alone for all {len(EC25_POLICIES)} policies")


def test_sam_scenarios_are_not_the_realized_draws():
    """SAM samples S scenarios internally. Confirm they are its own draws
    around theta_hat, not the simulator's realizations."""
    theta_hat, caps, _ = build_instance(N, M, seed=SEED, **DEFAULT_INSTANCE)
    realized = draw_realized_theta(theta_hat, EPSILON, TRIALS, SEED)

    captured = []
    real_crw = P.create_random_weights

    def spy_crw(base, eps, rng=None):
        out = real_crw(base, eps, rng)
        captured.append((np.array(base, copy=True), np.array(out, copy=True)))
        return out

    P.create_random_weights = spy_crw
    try:
        run_trials(theta_hat, caps, P.sam, K, num_trials=TRIALS, epsilon=EPSILON,
                   seed=SEED, S=S)
    finally:
        P.create_random_weights = real_crw

    assert len(captured) == S, f"SAM drew {len(captured)} scenarios, expected S={S}"
    for base, scenario in captured:
        assert np.array_equal(base, theta_hat), "SAM perturbed something other than theta_hat"
        for t in range(TRIALS):
            assert not np.array_equal(scenario, realized[t]), \
                "a SAM scenario coincides with a simulator realization"
    print(f"PASS: SAM's {S} scenarios are its own draws around theta_hat, "
          f"disjoint from the {TRIALS} realized thetas")


if __name__ == "__main__":
    test_theta_hat_is_the_formula()
    test_realized_theta_is_theta_hat_plus_noise()
    test_policies_only_ever_see_theta_hat()
    test_menus_do_not_depend_on_realized_theta()
    test_sam_scenarios_are_not_the_realized_draws()
    print("\nALL TESTS PASSED")
