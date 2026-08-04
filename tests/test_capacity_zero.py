"""Regression test for the capacity=0 fix: no policy may ever offer a
provider that starts with zero capacity, since such a provider can never be
matched to anyone (a pure phantom option that wastes a menu slot).

No test framework dependency (none is installed in the `patient` conda env);
run directly with `python tests/test_capacity_zero.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from patient import policies as P

POLICIES = [
    ("random_menu", P.random_menu, dict(seed=0)),
    ("offer_one", P.offer_one, dict()),
    ("offer_all", P.offer_all, dict()),
    ("sam", P.sam, dict(epsilon=0.1, S=3, seed=0)),
    ("deferred_acceptance", P.deferred_acceptance, dict()),
    ("capacity_greedy", P.capacity_greedy, dict()),
    ("exact_saa_milp", P.exact_saa_milp, dict(epsilon=0.1, S=2, seed=0)),
]


def _random_instance(rng, N, M):
    theta_hat = rng.uniform(0.1, 0.9, size=(N, M + 1))
    theta_hat[:, -1] = 0.15
    return theta_hat


def test_zero_capacity_provider_never_offered():
    rng = np.random.RandomState(1)
    N, M = 10, 6
    theta_hat = _random_instance(rng, N, M)
    capacities = np.array([2, 0, 3, 0, 1, 4])
    zero_idx = np.flatnonzero(capacities == 0)
    assert len(zero_idx) > 0

    for name, fn, kwargs in POLICIES:
        menu = fn(theta_hat, capacities, k=2, **kwargs)
        assert menu.shape == (N, M), f"{name}: unexpected menu shape {menu.shape}"
        offered_zero_cap = menu[:, zero_idx].any()
        assert not offered_zero_cap, f"{name}: offered a zero-capacity provider"
    print("PASS: no policy ever offers a zero-capacity provider")


def test_all_zero_capacity_gives_empty_menu():
    rng = np.random.RandomState(2)
    N, M = 6, 3
    theta_hat = _random_instance(rng, N, M)
    capacities = np.zeros(M)

    for name, fn, kwargs in POLICIES:
        menu = fn(theta_hat, capacities, k=2, **kwargs)
        assert menu.sum() == 0, f"{name}: offered a provider when all capacities are 0"
    print("PASS: all-zero-capacity instance produces empty menus for every policy")


if __name__ == "__main__":
    test_zero_capacity_provider_never_offered()
    test_all_zero_capacity_gives_empty_menu()
    print("ALL TESTS PASSED")
