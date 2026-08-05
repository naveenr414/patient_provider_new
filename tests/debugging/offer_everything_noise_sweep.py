"""Uncapped offer-everything (k = M) swept over the noise level epsilon.

Follow-up to `offer_everything_uncapped.py`, which ran the no-menu-design
benchmark at the single default epsilon=0.25 and found it beats SAM(k=25) by
+1.95%. The question here: does that hold across the whole epsilon grid, or is
it an artifact of one noise level? Menu design exists to protect against noise,
so the gap between "no cap at all" and SAM should close as epsilon grows.

Configuration is the `noise` experiment's: N=1225, M=700, 9 seeds x 25 trials,
epsilon in {0.01, 0.1, 0.2, 0.3, 0.4, 0.5}, the only change being k = M = 700
instead of k = 25.

No LPs are solved. The omniscient normalizer depends only on (epsilon, seed) --
theta realizations are policy-independent (see `simulator.run_trials`) -- so it
is read out of the stored `results/noise/*.json`, which used the same seeds,
trials, and instance parameters. That makes the whole sweep embarrassingly
parallel over the 6 x 9 = 54 (epsilon, seed) units.

Results are written in the same shape `run_experiments.sweep` produces, into
`results/noise_offer_everything/`, so `plot_offer_everything_noise.py` (and
`make_figures.load_experiment`) can read them exactly like any other sweep.

Run:  PYTHONPATH=. python tests/debugging/offer_everything_noise_sweep.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import policies as P
from scripts.run_experiments import (
    DEFAULT_INSTANCE, DEFAULT_M, DEFAULT_N, EPS_GRID, aggregate_seeds,
    run_one_seed, save_result,
)

ROOT = Path(__file__).resolve().parents[2]
NOISE_DIR = ROOT / "results" / "noise"
OUT_DIR = ROOT / "results" / "noise_offer_everything"
POLICY = "offer_everything"

SEEDS = list(range(9))
NUM_TRIALS = 25                 # matches results/noise
EPSILONS = EPS_GRID["epsilon"]
JOBS = len(SEEDS) * len(EPSILONS)


def se(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v))


def stored_omniscient():
    """{epsilon: {seed: E_theta[OPT(theta)]}} from the existing noise sweep.

    Every policy at a given (epsilon, seed) recorded the same normalizer, so
    any one file per epsilon would do; reading them all and asserting they
    agree is the cheap check that we really are reusing the right numbers."""
    omni = {}
    for f in sorted(NOISE_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        assert d["N"] == DEFAULT_N and d["M"] == DEFAULT_M, f"{f.name}: unexpected scale"
        assert d["num_trials"] == NUM_TRIALS, f"{f.name}: unexpected trial count"
        per_eps = omni.setdefault(d["epsilon"], {})
        for r in d["per_seed"]:
            prev = per_eps.setdefault(r["seed"], r["omniscient_utility"])
            assert abs(prev - r["omniscient_utility"]) < 1e-12, (
                f"{f.name}: omniscient disagrees across policies at seed {r['seed']}")
    return omni


def job(args):
    """One (epsilon, seed) unit: uncapped offer-everything, k = M."""
    epsilon, seed, omniscient = args
    rec = run_one_seed(POLICY, P.offer_all, {}, DEFAULT_N, DEFAULT_M, DEFAULT_M,
                       epsilon, seed, NUM_TRIALS, instance_kwargs=DEFAULT_INSTANCE,
                       omniscient=omniscient)
    return epsilon, seed, rec


def main():
    omni = stored_omniscient()
    missing = [e for e in EPSILONS if e not in omni]
    assert not missing, (f"no stored omniscient for epsilon={missing}; run "
                          f"`run_experiments.py noise` first")

    print(f"N={DEFAULT_N} M={DEFAULT_M} k=M={DEFAULT_M} (uncapped) | "
          f"{len(SEEDS)} seeds x {NUM_TRIALS} trials | epsilon={EPSILONS}\n"
          f"{len(EPSILONS) * len(SEEDS)} jobs on {JOBS} workers; "
          f"omniscient normalizers reused from results/noise\n", flush=True)

    specs = [(eps, seed, omni[eps][seed]) for eps in EPSILONS for seed in SEEDS]
    t0 = time.time()
    by_eps = {eps: [] for eps in EPSILONS}
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        futures = [ex.submit(job, s) for s in specs]
        for i, fut in enumerate(as_completed(futures), 1):
            eps, seed, rec = fut.result()
            by_eps[eps].append(rec)
            print(f"  [{i}/{len(specs)}] ({time.time() - t0:.0f}s) "
                  f"eps={eps} seed={seed} norm={rec['normalized_utility']:.4f}",
                  flush=True)

    # ------------------------------------------------------------------ save
    for eps in EPSILONS:
        per_seed = sorted(by_eps[eps], key=lambda r: r["seed"])
        result = aggregate_seeds(POLICY, DEFAULT_N, DEFAULT_M, DEFAULT_M, eps,
                                 NUM_TRIALS, per_seed)
        save_result(OUT_DIR, "", POLICY, {"epsilon": eps}, result)

    # ------------------------------------------------------------- reporting
    stored = {}
    for f in sorted(NOISE_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        stored.setdefault(d["policy"], {})[d["epsilon"]] = {
            r["seed"]: r for r in d["per_seed"]}

    print("\n" + "=" * 92)
    print("NORMALIZED UTILITY vs EPSILON  (mean +/- SE across 9 seeds)")
    print("=" * 92)
    header = f"{'policy':22s}" + "".join(f"{('eps=' + str(e)):>11s}" for e in EPSILONS)
    print(header)
    rows = {POLICY: {e: [r["normalized_utility"] for r in by_eps[e]] for e in EPSILONS}}
    for p in stored:
        rows[p] = {e: [stored[p][e][s]["normalized_utility"] for s in SEEDS]
                   for e in EPSILONS}
    order = [POLICY] + sorted(stored, key=lambda p: -np.mean(rows[p][EPSILONS[-1]]))
    for p in order:
        print(f"{p:22s}" + "".join(f"{np.mean(rows[p][e]):11.4f}" for e in EPSILONS))
        print(f"{'':22s}" + "".join(f"{'+/-' + format(se(rows[p][e]), '.4f'):>11s}"
                                    for e in EPSILONS))

    print("\n" + "=" * 92)
    print("PAIRED DIFFERENCE: uncapped offer-everything - SAM(k=25), per seed")
    print("=" * 92)
    print(f"{'epsilon':>8s} {'d(norm.)':>10s} {'SE':>8s} {'t':>8s} {'seeds>0':>9s} "
          f"{'#choices(unc.)':>15s} {'#choices(SAM)':>14s}")
    for e in EPSILONS:
        d = [rows[POLICY][e][i] - rows["sam"][e][i] for i in range(len(SEEDS))]
        cc_u = np.mean([r["choice_count_mean"] for r in by_eps[e]])
        cc_s = np.mean([stored["sam"][e][s]["choice_count_mean"] for s in SEEDS])
        print(f"{e:>8g} {np.mean(d):10.4f} {se(d):8.4f} {np.mean(d) / se(d):8.2f} "
              f"{sum(x > 0 for x in d):>6d}/9 {cc_u:15.2f} {cc_s:14.2f}")

    print("\ncost of the k=25 cap at each epsilon, as a share of the uncapped utility:")
    print(f"{'policy':22s}" + "".join(f"{('eps=' + str(e)):>11s}" for e in EPSILONS))
    unc_u = {e: {r["seed"]: r["utility"] for r in by_eps[e]} for e in EPSILONS}
    for p in order[1:]:
        cells = []
        for e in EPSILONS:
            rel = np.mean([1 - stored[p][e][s]["utility"] / unc_u[e][s] for s in SEEDS])
            cells.append(f"{100 * rel:10.2f}%")
        print(f"{p:22s}" + "".join(cells))

    print(f"\nwrote {OUT_DIR}/{POLICY}__epsilon=*.json ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
