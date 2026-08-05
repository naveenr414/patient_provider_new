"""Descriptive analysis of the semi-synthetic theta matrix at default settings.

Answers: what does the marginal distribution of theta_ij look like, what do the
per-patient (theta_i.) and per-provider (theta_.j) averages look like, and is
there cluster/atom structure or is it diffuse?

Run:  PYTHONPATH=. python tests/debugging/theta_distribution_analysis.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient import data_gen
from patient.utils import create_random_weights
from scripts.run_experiments import (
    DEFAULT_N, DEFAULT_M, DEFAULT_EPSILON, DEFAULT_K, DEFAULT_ALPHA,
    DEFAULT_OMEGA, DEFAULT_AVERAGE_DISTANCE, DEFAULT_EXIT_UTILITY,
    DEFAULT_CLIP_DISTANCE_TERM,
)

OUT = Path(__file__).resolve().parent
SEED = 0


def pct(x):
    return f"{100 * x:5.2f}%"


def describe(name, v):
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"{name:22s} mean={v.mean():.4f} sd={v.std():.4f}  "
          f"min={q[0]:.4f} p1={q[1]:.4f} p5={q[2]:.4f} p25={q[3]:.4f} "
          f"med={q[4]:.4f} p75={q[5]:.4f} p95={q[6]:.4f} p99={q[7]:.4f} max={q[8]:.4f}")


def main():
    print(f"instance: N={DEFAULT_N} M={DEFAULT_M} seed={SEED} alpha={DEFAULT_ALPHA} "
          f"omega={DEFAULT_OMEGA} dbar={DEFAULT_AVERAGE_DISTANCE} "
          f"clip_distance_term={DEFAULT_CLIP_DISTANCE_TERM} epsilon={DEFAULT_EPSILON} "
          f"exit_utility={DEFAULT_EXIT_UTILITY}\n")

    # cache theta so re-running the analysis doesn't reload the 629 MB Medicare csv
    cache = OUT / f"theta_cache_seed{SEED}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        theta, patients, providers = z["theta"], list(z["patients"]), list(z["providers"])
    else:
        theta, patients, providers = data_gen.semi_synthetic_theta(
            DEFAULT_N, DEFAULT_M, average_distance=DEFAULT_AVERAGE_DISTANCE,
            omega=DEFAULT_OMEGA, alpha=DEFAULT_ALPHA,
            clip_distance_term=DEFAULT_CLIP_DISTANCE_TERM, seed=SEED)
        np.savez(cache, theta=theta, patients=np.array(patients, dtype=object),
                 providers=np.array(providers, dtype=object))

    # one noise realization, exactly as the simulator draws it (exit column included
    # in the noise, then dropped again here since we only care about providers)
    theta_hat_full = np.hstack([theta, np.full((DEFAULT_N, 1), DEFAULT_EXIT_UTILITY)])
    rng = np.random.RandomState(SEED)
    realized_full = create_random_weights(theta_hat_full, DEFAULT_EPSILON, rng)
    realized = realized_full[:, :-1]

    # ---------------------------------------------------------------- marginals
    print("=" * 78)
    print("1. MARGINAL DISTRIBUTION OF theta_ij  (%d x %d = %d pairs)"
          % (theta.shape[0], theta.shape[1], theta.size))
    print("=" * 78)
    flat = theta.ravel()
    describe("theta_hat (all pairs)", flat)
    nz = flat[flat > 0]
    describe("theta_hat (nonzero)", nz)
    describe("realized theta", realized.ravel())
    print(f"\nmissing-distance pairs (theta=0): {pct((flat == 0).mean())} "
          f"of all pairs; {(flat == 0).sum()} pairs")
    print(f"exit utility = {DEFAULT_EXIT_UTILITY} sits BELOW the entire "
          f"[{DEFAULT_ALPHA}, 1] support of nonzero theta")

    # ------------------------------------------------------------------- atoms
    print("\n" + "=" * 78)
    print("2. ATOMS / CLUSTERS IN theta_ij")
    print("=" * 78)
    vals = Counter(np.round(flat, 10))
    top = vals.most_common(8)
    print(f"{len(vals)} distinct values overall; top atoms:")
    for v, c in top:
        print(f"   theta = {v:.4f}   {c:9d} pairs   {pct(c / flat.size)}")
    atom_mass = sum(c for _, c in top) / flat.size
    print(f"   -> top-8 values alone cover {pct(atom_mass)} of all pairs")

    # decompose: theta = alpha + (1-alpha)*(omega*beta + (1-omega)*dterm)
    # so beta=1 <=> theta >= alpha + (1-alpha)*omega
    band_lo = DEFAULT_ALPHA + (1 - DEFAULT_ALPHA) * DEFAULT_OMEGA
    beta1 = (flat >= band_lo - 1e-9) & (flat > 0)
    beta0 = (flat < band_lo - 1e-9) & (flat > 0)
    print(f"\ncomorbidity match beta_ij=1: {pct(beta1.mean())} of pairs "
          f"-> theta in [{band_lo:.2f}, 1.00]")
    print(f"comorbidity match beta_ij=0: {pct(beta0.mean())} of pairs "
          f"-> theta in [{DEFAULT_ALPHA:.2f}, {band_lo:.2f})")
    plateau = np.isclose(flat, 1.0) | np.isclose(flat, band_lo)
    floor = np.isclose(flat, band_lo) | np.isclose(flat, DEFAULT_ALPHA)
    print(f"at distance plateau (theta = 1.00 or {band_lo:.2f}): {pct(plateau.mean())}")
    print(f"at distance floor   (theta = {band_lo:.2f} or {DEFAULT_ALPHA:.2f}): {pct(floor.mean())}")
    interior = (flat > 0) & ~plateau & ~floor
    print(f"strictly interior (distance actually varying): {pct(interior.mean())}")

    print("\nnoise smears the atoms: distinct values in realized theta = "
          f"{len(np.unique(np.round(realized.ravel(), 10)))}")

    # ------------------------------------------------------- per-patient / provider
    print("\n" + "=" * 78)
    print("3. PER-PATIENT (theta_i.) AND PER-PROVIDER (theta_.j) PROFILES")
    print("=" * 78)
    pat_mean, prov_mean = theta.mean(axis=1), theta.mean(axis=0)
    describe("per-patient mean", pat_mean)
    describe("per-provider mean", prov_mean)
    print()
    describe("per-patient sd", theta.std(axis=1))
    describe("per-provider sd", theta.std(axis=0))

    # how much of the total variance is patient effect vs provider effect vs interaction
    grand = theta.mean()
    ss_tot = ((theta - grand) ** 2).sum()
    ss_pat = theta.shape[1] * ((pat_mean - grand) ** 2).sum()
    ss_prov = theta.shape[0] * ((prov_mean - grand) ** 2).sum()
    print(f"\nvariance decomposition of theta_ij:")
    print(f"   patient main effect  {pct(ss_pat / ss_tot)}")
    print(f"   provider main effect {pct(ss_prov / ss_tot)}")
    print(f"   interaction/residual {pct(1 - (ss_pat + ss_prov) / ss_tot)}")

    # ------------------------------------------------- what a patient actually sees
    print("\n" + "=" * 78)
    print(f"4. WITHIN-PATIENT SPREAD vs epsilon={DEFAULT_EPSILON} (k={DEFAULT_K})")
    print("=" * 78)
    srt = -np.sort(-theta, axis=1)
    top1, topk = srt[:, 0], srt[:, DEFAULT_K - 1]
    describe("best option", top1)
    describe(f"{DEFAULT_K}th best option", topk)
    describe("top1 - topk gap", top1 - topk)
    print(f"\npatients whose top-1 .. top-{DEFAULT_K} gap < epsilon ({DEFAULT_EPSILON}): "
          f"{pct(((top1 - topk) < DEFAULT_EPSILON).mean())}")
    print(f"patients whose top-1 .. top-{DEFAULT_K} gap < 2*epsilon: "
          f"{pct(((top1 - topk) < 2 * DEFAULT_EPSILON).mean())}")
    n_within = ((theta >= (top1[:, None] - DEFAULT_EPSILON)) & (theta > 0)).sum(axis=1)
    describe("# providers within eps of best", n_within.astype(float))
    n_above_exit = (theta > DEFAULT_EXIT_UTILITY).sum(axis=1)
    describe("# providers above exit", n_above_exit.astype(float))

    # how often does noise reorder the top of a patient's list?
    argmax_hat, argmax_real = theta.argmax(axis=1), realized.argmax(axis=1)
    print(f"\nnoise changes the argmax provider for "
          f"{pct((argmax_hat != argmax_real).mean())} of patients")

    # -------------------------------------------------------------- type structure
    print("\n" + "=" * 78)
    print("5. TYPE / CLUSTER STRUCTURE")
    print("=" * 78)
    COM = data_gen.COMORBIDITIES
    pat_types = Counter((p["zip"], tuple(p[c] for c in COM)) for p in patients)
    prov_types = Counter((p["zip"], tuple(p[c] for c in COM)) for p in providers)
    print(f"distinct patient types (zip x comorbidity profile): "
          f"{len(pat_types)} across {DEFAULT_N} patients")
    print(f"distinct provider types (zip x specialty profile):  "
          f"{len(prov_types)} across {DEFAULT_M} providers")
    print(f"distinct patient ZIPs: {len({p['zip'] for p in patients})}, "
          f"provider ZIPs: {len({p['zip'] for p in providers})}")
    print(f"distinct theta rows: {len(np.unique(theta, axis=0))}, "
          f"distinct theta cols: {len(np.unique(theta, axis=1).T)}")

    # effective rank via singular values. Raw theta is trivially rank-1-dominated
    # (every entry >= alpha, so the constant floor is the leading component);
    # the informative version double-centers first, i.e. looks at the interaction.
    for label, mat in [("raw theta", theta),
                       ("double-centered", theta - pat_mean[:, None]
                        - prov_mean[None, :] + grand)]:
        s = np.linalg.svd(mat, compute_uv=False)
        energy = np.cumsum(s ** 2) / (s ** 2).sum()
        r90 = int(np.searchsorted(energy, 0.90) + 1)
        r99 = int(np.searchsorted(energy, 0.99) + 1)
        print(f"\nSVD, {label}: rank for 90% energy = {r90}, 99% = {r99} "
              f"(of {min(theta.shape)})")
        print("   top-10 singular value share: " +
              " ".join(f"{x:.3f}" for x in (s[:10] ** 2 / (s ** 2).sum())))

    prov_com = np.array([[p[c] for c in COM] for p in providers])
    print(f"\nproviders with >=1 specialty flag: "
          f"{pct((prov_com.sum(axis=1) > 0).mean())}; "
          f"mean flags/provider = {prov_com.sum(axis=1).mean():.2f}")
    pat_com = np.array([[p[c] for c in COM] for p in patients])
    print(f"patients with >=1 comorbidity: {pct((pat_com.sum(axis=1) > 0).mean())}; "
          f"mean comorbidities/patient = {pat_com.sum(axis=1).mean():.2f}")

    # ------------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(flat, bins=100)
    ax[0, 0].set_yscale("log")
    ax[0, 0].axvline(DEFAULT_EXIT_UTILITY, color="r", ls="--", label="exit")
    ax[0, 0].set_title("theta_hat, all pairs (log count)")
    ax[0, 0].set_xlabel("theta"); ax[0, 0].legend()

    ax[0, 1].hist(nz, bins=100)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("theta_hat, nonzero pairs")
    ax[0, 1].set_xlabel("theta")

    ax[0, 2].hist(realized.ravel(), bins=100)
    ax[0, 2].set_yscale("log")
    ax[0, 2].axvline(DEFAULT_EXIT_UTILITY, color="r", ls="--")
    ax[0, 2].set_title(f"realized theta (one draw, eps={DEFAULT_EPSILON})")
    ax[0, 2].set_xlabel("theta")

    ax[1, 0].hist(pat_mean, bins=60)
    ax[1, 0].set_title("per-patient mean theta_i.")
    ax[1, 0].set_xlabel("mean theta")

    ax[1, 1].hist(prov_mean, bins=60)
    ax[1, 1].set_title("per-provider mean theta_.j")
    ax[1, 1].set_xlabel("mean theta")

    ax[1, 2].hist(top1 - topk, bins=60)
    ax[1, 2].axvline(DEFAULT_EPSILON, color="r", ls="--", label="epsilon")
    ax[1, 2].set_title(f"within-patient top1 - top{DEFAULT_K} gap")
    ax[1, 2].set_xlabel("gap"); ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig(OUT / "theta_distribution.png", dpi=120)
    print(f"\nsaved {OUT / 'theta_distribution.png'}")

    # sorted-profile heatmap: patients x providers, both sorted by mean, to see blocks
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    pi, pj = np.argsort(-pat_mean), np.argsort(-prov_mean)
    im = ax2[0].imshow(theta[np.ix_(pi, pj)], aspect="auto", cmap="viridis",
                       interpolation="nearest")
    ax2[0].set_title("theta sorted by patient/provider mean")
    ax2[0].set_xlabel("provider (sorted)"); ax2[0].set_ylabel("patient (sorted)")
    fig2.colorbar(im, ax=ax2[0])

    ax2[1].plot(np.sort(prov_mean)[::-1])
    ax2[1].set_title("provider mean theta, sorted")
    ax2[1].set_xlabel("provider rank"); ax2[1].set_ylabel("mean theta_.j")
    fig2.tight_layout()
    fig2.savefig(OUT / "theta_structure.png", dpi=120)
    print(f"saved {OUT / 'theta_structure.png'}")


if __name__ == "__main__":
    main()
