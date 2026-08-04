"""CLI experiment driver: reproduces every figure/table in the paper
(Figures 2-7, Table 1, EC.1-EC.2.6). Results are saved as one JSON per
(experiment, policy, config point) under `results/`, each aggregated across
`--num-seeds` independent environment draws x `--num-trials` patient
arrival-order permutations, per the paper's Section 6.1 methodology.

Per seed/instance: theta_hat (the platform's fixed estimate) is drawn once
from `data_gen`, and each of the `--num-trials` trials then draws its own
realized theta = theta_hat + Uniform(-eps, eps) together with its own arrival
order (Section 6.1: "each seed averages across 25 random permutations of
sigma and noise values"). Policies plan against theta_hat only; patients
choose, and every downstream metric (omniscient normalizer, choice_utility,
fairness) is measured against each trial's realized theta.

Default-config values (N=1225, M=700, c_j ~ Poisson(1), d_bar=20.2, exit
utility=0.25, epsilon=0.25, k=25, S=10) are taken verbatim from EC.1. Sweep
grids come from the axis ticks/values stated in Section 6 and EC.2. Values the
paper never states, and the defaults used instead, are flagged at their
definition: DEFAULT_ALPHA, DEFAULT_CLIP_DISTANCE_TERM, DEFAULT_MNL_TEMPERATURE,
and --small-M / --small-k.

Experiments are named for what they vary, not for the figure number they end
up in, and several figures share one run: the four default-config figures all
read `default`, and the main-text and EC.2.5 versions of the noise and
patient/provider-ratio figures are subsets of `noise` and
`patient_provider_ratio` rather than separate sweeps.

Usage:
    python scripts/run_experiments.py --experiment default
    python scripts/run_experiments.py --experiment all --num-seeds 2 --num-trials 3 --N 60 --M 40
    ./scripts/run_all.sh                     # everything, then the figures
"""
import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from patient import data_gen
from patient import metrics as MET
from patient import policies as P
from patient.simulator import draw_realized_theta, run_trials
from patient.utils import NumpyEncoder, load_json, save_json

# Sentinel: "do not solve the omniscient LPs in this job, the driver will
# supply the normalizer afterwards" (see `run_one_seed`).
DEFER_OMNISCIENT = "defer"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---------------------------------------------------------------------------
# Default-config values (EC.1)
# ---------------------------------------------------------------------------
DEFAULT_N = 1225
DEFAULT_M = 700
DEFAULT_EPSILON = 0.25
DEFAULT_K = 25
DEFAULT_S = 10
# Figure 3-left's exact SAA-MILP is meant to stand in for the true optimum, so
# it gets more scenarios than SAM does: with S=10 it visibly overfits its own
# sample and its out-of-sample utility drifts down as epsilon grows, which
# would show up as baselines spuriously "improving" with noise.
DEFAULT_MILP_S = 25
# Hard wall-clock cap on a single SAA-MILP solve. Without it one unlucky
# instance can hold up an entire sweep; with it the solver returns its best
# incumbent and prints the gap it reached (see `policies.exact_saa_milp`).
DEFAULT_MILP_TIME_LIMIT = 300
DEFAULT_AVERAGE_DISTANCE = 20.2
DEFAULT_OMEGA = 0.5
DEFAULT_EXIT_UTILITY = 0.25
DEFAULT_AVG_CAPACITY = 1.0
# EC.1's theta formula has an alpha the paper never assigns a number to.
# With DEFAULT_CLIP_DISTANCE_TERM below, theta lands in [alpha, 1] and alpha
# is the floor -- the match quality of a pair at or beyond d_bar with no
# comorbidity match. 0.5 is the constant the reference implementation used,
# and it puts a patient's viable options in a band about as wide as epsilon,
# which is the regime the paper's results live in.
DEFAULT_ALPHA = 0.5
# Figure 5-left's MNL choice model needs a logit temperature tau, which the
# paper never states. 0.1 sits between the two degenerate ends given
# theta in [0, 1]: choice is clearly stochastic (a 0.1 utility gap gives only
# ~2.7:1 odds) without collapsing to uniform.
DEFAULT_MNL_TEMPERATURE = 0.1
# Whether EC.1's distance term is clipped to [0,1] before mixing (reference
# implementation) or only the final theta is clipped (formula read
# literally). See `data_gen.semi_synthetic_theta` -- this decides whether a
# patient's viable options are compressed into a band narrower than epsilon.
DEFAULT_CLIP_DISTANCE_TERM = True

DEFAULT_INSTANCE = dict(
    average_distance=DEFAULT_AVERAGE_DISTANCE,
    omega=DEFAULT_OMEGA,
    exit_utility=DEFAULT_EXIT_UTILITY,
    avg_capacity=DEFAULT_AVG_CAPACITY,
    alpha=DEFAULT_ALPHA,
    clip_distance_term=DEFAULT_CLIP_DISTANCE_TERM,
    distribution="semi_synthetic",
)

MAIN_POLICIES = {
    "random": (P.random_menu, {}),
    "offer_one": (P.offer_one, {}),
    "offer_all": (P.offer_all, {}),
    "sam": (P.sam, {}),
}
EC25_POLICIES = {
    **MAIN_POLICIES,
    "deferred_acceptance": (P.deferred_acceptance, {}),
    "capacity_greedy": (P.capacity_greedy, {}),
}


# ---------------------------------------------------------------------------
# Instance construction / single-config runner
# ---------------------------------------------------------------------------
def build_instance(N, M, seed, average_distance, omega, exit_utility, avg_capacity,
                    alpha=DEFAULT_ALPHA, clip_distance_term=DEFAULT_CLIP_DISTANCE_TERM,
                    distribution="semi_synthetic"):
    """Returns (theta_hat: N x (M+1), capacities: length-M, patients-or-None)
    -- the platform's fixed utility estimate for this instance/seed. `patients` (list of
    dicts with a 'zip' key) is only available for the semi_synthetic
    distribution; other distributions return None (no ZIP metadata, so
    zip-fairness metrics are skipped for them)."""
    rng = np.random.RandomState(seed)
    if distribution == "semi_synthetic":
        theta, patients, _providers = data_gen.semi_synthetic_theta(
            N, M, average_distance=average_distance, omega=omega, alpha=alpha,
            clip_distance_term=clip_distance_term, seed=seed
        )
    elif distribution == "uniform":
        theta, patients = data_gen.uniform_theta(N, M, seed=seed), None
    elif distribution == "normal":
        theta, patients = data_gen.normal_theta(N, M, seed=seed), None
    elif distribution == "latent":
        theta, patients = data_gen.latent_theta(N, M, seed=seed), None
    else:
        raise ValueError(f"unknown distribution {distribution}")

    theta_hat = np.hstack([theta, np.full((N, 1), exit_utility)])
    capacities = rng.poisson(avg_capacity, M)
    return theta_hat, capacities, patients


def run_one_seed(policy_name, policy_fn, policy_kwargs, N, M, k, epsilon, seed,
                  num_trials, gamma=None, instance_kwargs=None, extra_metrics_fn=None,
                  omniscient=None):
    """Run a single seed/instance of one (policy, config): builds theta_hat
    (the platform's fixed estimate), plans a menu from it, replays
    `num_trials` trials each with its own realized theta and arrival order
    (Section 6.1), and returns that seed's metrics dict. Split out from
    `run_one` so seeds can be parallelized independently (each is a fully
    self-contained unit of work -- no shared state, safe to run in separate
    processes) -- see `run_one`, which is the sequential-seeds convenience
    wrapper around this.

    omniscient: precomputed E_theta[OPT(theta)] for this (config, seed). The
        trials' theta realizations depend only on the seed, not on the policy
        (see `simulator.run_trials`), so the normalizer is identical for
        every policy at a given seed and a driver can solve its `num_trials`
        LPs once and share the result rather than re-solving them per policy;
        `omniscient_for_seed` computes it. None (the default) solves them
        inline. DEFER_OMNISCIENT leaves `normalized_utility` as NaN for the
        caller to fill in later -- which lets a parallel driver schedule the
        omniscient solves as ordinary jobs alongside the policy jobs instead
        of as a blocking phase before them."""
    instance_kwargs = {**DEFAULT_INSTANCE, **(instance_kwargs or {})}
    seed_t0 = time.time()
    theta_hat, capacities, patients = build_instance(N, M, seed=seed, **instance_kwargs)

    out = run_trials(theta_hat, capacities, policy_fn, k, num_trials=num_trials,
                      epsilon=epsilon, gamma=gamma, seed=seed, **policy_kwargs)

    theta_realized = out["theta_realized"]
    choice_util = MET.choice_utility(out["effective_menus"], theta_realized)
    if omniscient is None:
        omniscient = MET.omniscient_utility(theta_realized, capacities)
    # compared by value, not identity: the sentinel is pickled on its way to a
    # worker process, so `is` would not survive the round trip.
    deferred = omniscient == DEFER_OMNISCIENT
    util = MET.utility(out["rewards"])
    rec = {
        "seed": seed,
        "runtime_sec": time.time() - seed_t0,
        "utility": util,
        "normalized_utility": float("nan") if deferred else util / omniscient,
        "omniscient_utility": float("nan") if deferred else omniscient,
        "match_rate": MET.match_rate(out["chosen"], M),
        "choice_count_mean": float(MET.choice_count(out["effective_menus"]).mean()),
        "choice_utility_mean": float(np.nanmean(choice_util)),
    }
    if patients is not None:
        zips = [p["zip"] for p in patients]
        fair = MET.zip_fairness(out["chosen"], M, zips, rewards=out["rewards"])
        rec["zip_fairness_p25"] = fair["p25"]
        rec["zip_gini_match_rate"] = fair["gini_match_rate"]
        rec["zip_gini_utility"] = fair["gini_utility"]

    if extra_metrics_fn is not None:
        rec["extra"] = extra_metrics_fn(out, theta_realized, capacities, patients)

    return rec


def omniscient_for_seed(N, M, epsilon, seed, num_trials, instance_kwargs=None,
                         trials=None):
    """E_theta[OPT(theta)] for one (config, seed): replays exactly the theta
    realizations `run_one_seed` will see (they depend only on the seed) and
    averages the per-realization omniscient LP. Policy-independent, so a
    driver can compute this once per seed and hand it to every policy via
    `run_one_seed(..., omniscient=...)`.

    trials: optional list of trial indices, if a caller wants only part of
    one seed's realizations."""
    instance_kwargs = {**DEFAULT_INSTANCE, **(instance_kwargs or {})}
    theta_hat, capacities, _ = build_instance(N, M, seed=seed, **instance_kwargs)
    theta_realized = draw_realized_theta(theta_hat, epsilon, num_trials, seed)
    if trials is not None:
        theta_realized = theta_realized[list(trials)]
    return MET.omniscient_utility(theta_realized, capacities)


def aggregate_seeds(policy_name, N, M, k, epsilon, num_trials, per_seed):
    """Combine a list of `run_one_seed` records (any order) into the same
    result-dict shape `run_one` returns: mean/std of every numeric field
    across seeds, plus per_seed detail."""
    num_seeds = len(per_seed)
    total_runtime = sum(r["runtime_sec"] for r in per_seed)
    numeric_keys = [
        key for key in per_seed[0]
        if key not in ("seed", "extra") and isinstance(per_seed[0][key], (int, float))
    ]
    # Seeds are the unit of independent replication, not trials: the 25 trials
    # inside a seed share that seed's theta_hat and capacities, so they shrink
    # the noise in that seed's mean but add no independent draw of the
    # environment. Pooling all (seed, trial) pairs -- or worse, all
    # (seed, trial, patient) triples, as the reference notebook does -- is
    # pseudo-replication and understates the error by ~sqrt(25) or ~sqrt(30000).
    # So: `_std` is the across-seed sample sd, `_se` is that / sqrt(num_seeds),
    # and `_se` is what figures should show.
    agg = {key: float(np.mean([r[key] for r in per_seed])) for key in numeric_keys}
    agg_std = {f"{key}_std": float(np.std([r[key] for r in per_seed], ddof=1))
               if num_seeds > 1 else 0.0 for key in numeric_keys}
    agg_se = {f"{key}_se": agg_std[f"{key}_std"] / np.sqrt(num_seeds) for key in numeric_keys}
    return {
        "policy": policy_name, "N": N, "M": M, "k": k, "epsilon": epsilon,
        "num_seeds": num_seeds, "num_trials": num_trials,
        # total_runtime_sec: wall time summed across all num_seeds -- useful for
        # driver progress tracking (NOT wall-clock time if seeds ran in parallel).
        # agg['runtime_sec']: mean per-seed runtime -- this is the metric EC.2.6's
        # runtime-vs-N/S figure actually wants (single-instance runtime), NOT
        # total_runtime_sec, which would be inflated by a factor of num_seeds.
        "total_runtime_sec": total_runtime,
        "per_seed": per_seed, "agg": {**agg, **agg_std, **agg_se},
    }


def _policy_seed_job(args):
    """One (policy, seed) unit of work. Module-level so it pickles."""
    key, kwargs = args
    return key, run_one_seed(**kwargs)


def _omniscient_seed_job(args):
    """One seed's omniscient normalizer. Module-level so it pickles."""
    key, kwargs = args
    return key, omniscient_for_seed(**kwargs)


def run_jobs(specs, jobs, progress_path=None, label=""):
    """Execute a flat list of independent units of work in parallel.

    specs: list of (kind, key, kwargs), kind in {"policy", "omniscient"}.
        Each spec is one (policy, seed) run or one seed's omniscient solve --
        the granularity the parallelism is meant to be at. Trials stay
        sequential inside a unit; they are cheap for policy runs, and for
        omniscient solves they are the LPs themselves, which is why those
        are scheduled here as peers of the policy jobs rather than as a
        blocking phase before them.
    jobs: worker processes; 1 runs in-process (no pool, easier to debug).
    progress_path: if given, every completed unit is appended there as a
        JSON line as soon as it finishes, so a long run's partial results
        survive an interruption.

    Returns {key: result} -- a per-seed metrics record for "policy" specs, a
    float for "omniscient" specs."""
    fns = {"policy": _policy_seed_job, "omniscient": _omniscient_seed_job}
    payload = [(fns[kind], (key, kwargs)) for kind, key, kwargs in specs]
    results, done, total, t0 = {}, 0, len(payload), time.time()

    if progress_path is not None:
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)

    def record(key, value):
        nonlocal done
        results[key] = value
        done += 1
        print(f"  [{label}] {done}/{total} done ({time.time() - t0:.0f}s) {key}", flush=True)
        if progress_path is not None:
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({"key": list(key), "result": value},
                                    cls=NumpyEncoder) + "\n")

    if jobs <= 1:
        for fn, args in payload:
            key, value = fn(args)
            record(key, value)
        return results

    with ProcessPoolExecutor(max_workers=min(jobs, total)) as ex:
        futures = [ex.submit(fn, args) for fn, args in payload]
        for fut in as_completed(futures):
            key, value = fut.result()
            record(key, value)
    return results


def run_one(policy_name, policy_fn, policy_kwargs, N, M, k, epsilon, num_seeds,
            num_trials, gamma=None, instance_kwargs=None, extra_metrics_fn=None):
    """Run one (policy, config) across `num_seeds` instance draws x
    `num_trials` arrival-order permutations each, sequentially, returning
    per-seed metrics plus their mean/std across seeds. `sweep` parallelizes
    across (policy, seed) instead; this is the simple path for one-off checks.

    extra_metrics_fn(out, theta_realized, capacities, patients) -> dict, optional: called
    once per seed to collect experiment-specific extras (e.g. fig6's
    histograms, fig7's per-ZIP match rates) merged into that seed's record
    under 'extra' (not included in the numeric agg/agg_std)."""
    per_seed = [
        run_one_seed(policy_name, policy_fn, policy_kwargs, N, M, k, epsilon, seed,
                     num_trials, gamma=gamma, instance_kwargs=instance_kwargs,
                     extra_metrics_fn=extra_metrics_fn)
        for seed in range(num_seeds)
    ]
    return aggregate_seeds(policy_name, N, M, k, epsilon, num_trials, per_seed)


def _slug(point):
    if not point:
        return "default"
    return "__".join(f"{k}={v}" for k, v in point.items())


def save_result(out_dir, experiment, policy_name, point, result):
    path = out_dir / experiment / f"{policy_name}__{_slug(point)}.json"
    save_json(str(path), result)
    return path


def _policy_kwargs(policy_name, extra_pkwargs, epsilon, S):
    """epsilon and seed are handled automatically by `run_trials` (defaulted
    to that call's own epsilon/seed unless overridden here); only S needs
    setting explicitly since it has no equivalent on `run_trials`."""
    pkwargs = dict(extra_pkwargs)
    if policy_name == "exact_saa_milp":
        pkwargs.setdefault("S", DEFAULT_MILP_S)
        pkwargs.setdefault("time_limit", DEFAULT_MILP_TIME_LIMIT)
    elif policy_name == "sam":
        pkwargs.setdefault("S", S)
    return pkwargs


# ---------------------------------------------------------------------------
# Generic sweep runner: for experiments that are "policies x grid of scalar
# params" with only the aggregate metrics needed downstream.
# ---------------------------------------------------------------------------
def sweep(experiment, policies, param_grid, base, num_seeds, num_trials, out_dir,
          jobs=1, extra_metrics_fn=None):
    """param_grid: dict of {param_name: [values]} swept via a full cartesian
    product; each grid point overrides `base`'s N/M/k/epsilon/S/gamma or an
    instance-construction param (average_distance/omega/exit_utility/
    avg_capacity/alpha/clip_distance_term/distribution).

    Every (grid point, policy, seed) is one independent job, and so is every
    (grid point, seed) omniscient solve; they all go into a single pool, so a
    sweep saturates `jobs` workers rather than running one config at a time.
    Trials run sequentially within a job."""
    keys = list(param_grid.keys())
    grid_points = list(itertools.product(*param_grid.values())) if keys else [()]
    instance_keys = ("average_distance", "omega", "exit_utility", "avg_capacity",
                      "alpha", "clip_distance_term", "distribution")

    configs, specs = {}, []
    for values in grid_points:
        point = dict(zip(keys, values))
        cfg = {**base, **point}
        pt = tuple(sorted(point.items()))
        configs[pt] = (point, cfg)
        ikw = {kk: cfg[kk] for kk in instance_keys if kk in cfg}
        common = dict(N=cfg["N"], M=cfg["M"], epsilon=cfg["epsilon"],
                      num_trials=num_trials, instance_kwargs=ikw)
        for seed in range(num_seeds):
            specs.append(("omniscient", ("omni", pt, seed), dict(seed=seed, **common)))
            for policy_name, (policy_fn, extra_pkwargs) in policies.items():
                specs.append(("policy", ("policy", pt, policy_name, seed), dict(
                    policy_name=policy_name, policy_fn=policy_fn,
                    policy_kwargs=_policy_kwargs(policy_name, extra_pkwargs,
                                                 cfg["epsilon"], cfg.get("S", DEFAULT_S)),
                    k=cfg["k"], seed=seed, gamma=cfg.get("gamma"),
                    extra_metrics_fn=extra_metrics_fn,
                    omniscient=DEFER_OMNISCIENT, **common)))

    out = run_jobs(specs, jobs, progress_path=out_dir / experiment / "_progress.jsonl",
                   label=experiment)

    for pt, (point, cfg) in configs.items():
        for policy_name in policies:
            per_seed = []
            for seed in range(num_seeds):
                rec = out[("policy", pt, policy_name, seed)]
                omni = out[("omni", pt, seed)]
                rec["omniscient_utility"] = omni
                rec["normalized_utility"] = rec["utility"] / omni
                per_seed.append(rec)
            per_seed.sort(key=lambda r: r["seed"])
            result = aggregate_seeds(policy_name, cfg["N"], cfg["M"], cfg["k"],
                                     cfg["epsilon"], num_trials, per_seed)
            save_result(out_dir, experiment, policy_name, point, result)
            print(f"[{experiment}] {policy_name} {point or '(default)'} -> "
                  f"norm_util={result['agg']['normalized_utility']:.3f} "
                  f"({result['total_runtime_sec']:.1f}s cpu)", flush=True)


# ---------------------------------------------------------------------------
# Per-experiment entry points
# ---------------------------------------------------------------------------
EPS_GRID = {"epsilon": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}


def _default_extra_metrics(out, theta_realized, capacities, patients):
    """Everything the default-config figures need beyond the scalar metrics:
    the per-patient choice-count and choice-utility histograms, and the
    per-ZIP match rate. Collected together because all of those figures read
    the same run."""
    cc = MET.choice_count(out["effective_menus"]).flatten()
    cu = MET.choice_utility(out["effective_menus"], theta_realized).flatten()
    cu = cu[~np.isnan(cu)]
    # Bin edges must not depend on the data, or seeds produce histograms of
    # different lengths that cannot be summed. 0..M+1 is the full possible
    # range of choice_count (at most every provider, plus the exit option).
    n_providers = out["menus"].shape[1]
    cc_hist, cc_edges = np.histogram(cc, bins=np.arange(0, n_providers + 3))
    cu_hist, cu_edges = np.histogram(cu, bins=20, range=(0, 1))
    extra = {
        "choice_count_hist": cc_hist.tolist(), "choice_count_edges": cc_edges.tolist(),
        "choice_utility_hist": cu_hist.tolist(), "choice_utility_edges": cu_edges.tolist(),
    }
    if patients is not None:
        zips = np.array([p["zip"] for p in patients])
        matched = out["chosen"] != (theta_realized.shape[-1] - 1)
        rate = matched.mean(axis=0)
        extra["zip_match_rate"] = {z: float(rate[zips == z].mean()) for z in np.unique(zips)}
    return extra


def experiment_default(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """The default configuration, all six policies. One run feeds four
    figures -- the headline comparison, the geographic Gini, the choice
    distributions, and the Connecticut map -- because they are all the same
    config and differ only in what is plotted."""
    sweep("default", EC25_POLICIES, {},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs,
          extra_metrics_fn=_default_extra_metrics)


def experiment_noise(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Noise level epsilon, all six policies. Feeds both the main-text noise
    figure (four policies) and the EC.2.5 stronger-baselines one (all six):
    same sweep, different subsets plotted.

    The grid starts at 0.01, not 0. At exactly epsilon=0 all S SAM scenarios
    coincide, so every LP-optimal pair sits at exactly Delta_ij = lambda_j --
    a tie that Algorithm 1's "largest *positive* m_ij" excludes, leaving SAM
    offering almost nothing. The reference implementation's grid also starts
    at 0.01."""
    sweep("noise", EC25_POLICIES, EPS_GRID,
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_noise_exact_small(small_N, small_M, small_k, num_seeds, num_trials,
                                  out_dir, jobs=1):
    """Small-scale comparison against the exact SAA-MILP, swept over epsilon
    -- the approximation-ratio figure. Kept separate from `noise` because it
    runs at a different (much smaller) scale.

    small_k must be strictly less than small_M: at k >= M "offer a subset"
    and "offer everyone" are the same menu, and every policy collapses to the
    same outcome."""
    if small_k >= small_M:
        raise ValueError(f"small_k ({small_k}) must be < small_M ({small_M}) or the menu-size "
                          "constraint this panel is supposed to test becomes a no-op")
    policies = {**MAIN_POLICIES, "exact_saa_milp": (P.exact_saa_milp, {})}
    sweep("noise_exact_small", policies, EPS_GRID,
          dict(N=small_N, M=small_M, k=small_k, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_patient_provider_ratio(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Patient/provider ratio N/M with M fixed, all six policies -- feeds both
    the main-text ratio figure and the EC.2.5 stronger-baselines one."""
    sweep("patient_provider_ratio", EC25_POLICIES,
          {"N": [round(r * M) for r in (1.5, 2.0, 2.5)]},
          dict(M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_provider_capacity(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Average provider capacity c_bar, with N and M fixed."""
    sweep("provider_capacity", MAIN_POLICIES, {"avg_capacity": [1.0, 2.0, 3.0, 4.0, 5.0]},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_mnl_choice(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Patients responding by MNL/logit rather than deterministic argmax."""
    sweep("mnl_choice", MAIN_POLICIES, {},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S,
               gamma=DEFAULT_MNL_TEMPERATURE),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_exit_option(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Value of the outside option theta_{i,M+1}."""
    sweep("exit_option", MAIN_POLICIES, {"exit_utility": [0.10, 0.25, 0.50]},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_distance_threshold(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Distance threshold d_bar behind theta (EC.2.1).

    Grid starts at 10: with the distance term clipped, d_bar also sets the
    plateau (d <= d_bar/2) and the floor (d >= d_bar), so at d_bar = 1 every
    Connecticut pair sits at the floor theta = alpha, only comorbidity varies,
    and every policy collapses to the same outcome."""
    sweep("distance_threshold", MAIN_POLICIES, {"average_distance": [10, 15, 20, 25, 30]},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_comorbidity_weight(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Comorbidity weight omega behind theta (EC.2.1)."""
    sweep("comorbidity_weight", MAIN_POLICIES, {"omega": [0.0, 0.25, 0.5, 0.75, 1.0]},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_sam_samples(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Number of SAM scenarios S. One run feeds both the utility-vs-S figure
    and the runtime-vs-S figure. The three baselines take no S argument, so
    they are run once at the default and their single result is written under
    every S point, keeping the file layout uniform for plotting."""
    s_values = [1, 2, 5, 10, 25]
    sweep("sam_samples", {"sam": MAIN_POLICIES["sam"]}, {"S": s_values},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON),
          num_seeds, num_trials, out_dir, jobs=jobs)

    baselines = {kk: v for kk, v in MAIN_POLICIES.items() if kk != "sam"}
    sweep("sam_samples_baselines", baselines, {},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)
    for policy_name in baselines:
        result = load_json(str(out_dir / "sam_samples_baselines" / f"{policy_name}__default.json"))
        for s_val in s_values:
            save_result(out_dir, "sam_samples", policy_name, {"S": s_val}, result)


def experiment_theta_distribution(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Alternative utility distributions: uniform, normal, latent (EC.2.3)."""
    sweep("theta_distribution", MAIN_POLICIES,
          {"distribution": ["uniform", "normal", "latent"]},
          dict(N=N, M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_menu_size(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """Menu budget k crossed with noise epsilon (EC.2.4)."""
    sweep("menu_size", MAIN_POLICIES, {"k": [5, 10, 25], "epsilon": [0.01, 0.20, 0.40]},
          dict(N=N, M=M, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


def experiment_scale(N, M, num_seeds, num_trials, out_dir, jobs=1):
    """SAM's runtime as the patient count grows (EC.2.6). The metric this
    wants is agg['runtime_sec'] (mean per-seed wall time for one planning
    call), not total_runtime_sec."""
    sweep("scale", {"sam": MAIN_POLICIES["sam"]}, {"N": [800, 1200, 1600, 2000]},
          dict(M=M, k=DEFAULT_K, epsilon=DEFAULT_EPSILON, S=DEFAULT_S),
          num_seeds, num_trials, out_dir, jobs=jobs)


# Every experiment, in the order run_all.sh runs them. `needs_small` marks the
# ones parameterised by --small-N/--small-M/--small-k rather than --N/--M.
EXPERIMENT_FNS = {
    "default": experiment_default,
    "noise": experiment_noise,
    "noise_exact_small": experiment_noise_exact_small,
    "patient_provider_ratio": experiment_patient_provider_ratio,
    "provider_capacity": experiment_provider_capacity,
    "mnl_choice": experiment_mnl_choice,
    "exit_option": experiment_exit_option,
    "distance_threshold": experiment_distance_threshold,
    "comorbidity_weight": experiment_comorbidity_weight,
    "sam_samples": experiment_sam_samples,
    "theta_distribution": experiment_theta_distribution,
    "menu_size": experiment_menu_size,
    "scale": experiment_scale,
}
SMALL_SCALE_EXPERIMENTS = {"noise_exact_small"}
EXPERIMENTS = list(EXPERIMENT_FNS)





def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS + ["all"])
    parser.add_argument("--num-seeds", type=int, default=9)
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="base patient count (ignored by sweeps that vary N themselves)")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="base provider count (ignored by sweeps that vary M themselves)")
    parser.add_argument("--small-N", type=int, default=20, help="exact-MILP patient count (paper-stated)")
    parser.add_argument("--small-M", type=int, default=10, help="exact-MILP provider count (not stated in the paper)")
    parser.add_argument("--small-k", type=int, default=5, help="exact-MILP menu size (not stated in the paper; must be < --small-M or the menu-size constraint it tests becomes a no-op)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes. Parallelism is over (policy, seed) units "
                              "plus one omniscient-normalizer unit per seed; trials stay "
                              "sequential inside a unit. 1 runs everything in-process.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        fn = EXPERIMENT_FNS[exp]
        if exp in SMALL_SCALE_EXPERIMENTS:
            fn(args.small_N, args.small_M, args.small_k,
               args.num_seeds, args.num_trials, args.out_dir, args.jobs)
        else:
            fn(args.N, args.M, args.num_seeds, args.num_trials, args.out_dir, args.jobs)


if __name__ == "__main__":
    main()
