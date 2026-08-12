"""Spread vs performance, at FIXED epsilon -- the `gap` half of instance_family.

`instance_family.py`'s own three-panel figure is built around the collapse
test (do the gap and noise sweeps trace one curve in kappa?), so it shows both
sweeps overlaid and only three policies. This script answers the narrower
question instead:

    with epsilon held at 0.10, how does each policy's utility move as a
    patient's options spread apart?

so it reads ONLY the gap-sweep rows, and shows every policy. The x axis is

    kappa = (top-1 minus top-2) / epsilon

with a secondary axis giving the same thing in raw utility units (just
kappa * epsilon, since epsilon is constant across these rows). Recall from
`instance_family.build` that sliding kappa redistributes value within a
patient's row -- the row mean is pinned at MU -- so this is a pure spread
knob, not a "make everyone better off" knob.

Panels:
  left   levels: normalized utility (against the per-realization omniscient LP)
  right  the same thing paired against offer-one and divided by epsilon.
         Every policy sees identical seeds and identical trials, so the
         difference has a much smaller SE than the levels do, and the
         crossover is only legible here. offer-one appears as the identically
         zero line, which keeps the palette positions -- and therefore the
         colours -- identical across the two panels.

Styling is `patient.plotting` via `make_figures`, the project's standard.
Note `plot_line` asserts a palette is long enough for its line count and
`six_color` has six entries, which is exactly POLICIES below; adding a
seventh policy needs a different palette.

Run:  PYTHONPATH=. python tests/debugging/plot_instance_family_spread.py
      (needs instance_family.json -- run instance_family.py first)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient.plotting import create_axes, plot_line, plot_scatter
from scripts import make_figures as MF

HERE = Path(__file__).resolve().parent

POLICIES = ["random", "offer_one", "offer_all", "sam",
            "deferred_acceptance", "capacity_greedy"]
XLABEL = r"Option Spread $\kappa$ = (top-1 $-$ top-2) / $\epsilon$"


def crossover(rows, policy):
    """The kappa at which `policy` stops beating offer-one, linearly
    interpolated between the two bracketing grid points. None if the sign
    never flips inside the grid."""
    xs = [r["kappa"] for r in rows]
    ys = [r["policies"][policy].get("vs_offer_one", 0.0) for r in rows]
    for i in range(len(xs) - 1):
        if ys[i] > 0 >= ys[i + 1]:
            t = ys[i] / (ys[i] - ys[i + 1])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def main():
    src = HERE / "instance_family.json"
    if not src.exists():
        raise SystemExit(f"no {src} -- run instance_family.py first")
    blob = json.loads(src.read_text())
    cfg = blob["config"]
    rows = sorted([r for r in blob["rows"] if r["sweep"] == "gap"],
                  key=lambda r: r["kappa"])
    if not rows:
        raise SystemExit("no gap-sweep rows in instance_family.json")

    epsilon = rows[0]["epsilon"]
    assert all(r["epsilon"] == epsilon for r in rows), "gap sweep must fix epsilon"
    kstar = crossover(rows, "sam")

    dims = (1, 2)
    fig, ax = create_axes(
        dims,
        {"figsize": (11, 3.1), "style_size": "paper",
         "hide_spines": True, "has_grid": True},
        x_labels=[[XLABEL, XLABEL]],
        y_labels=[["Norm. Utility", r"(policy $-$ Offer-One) / $\epsilon$"]],
        titles=[[f"Levels ($\\epsilon$ fixed at {epsilon:g})",
                 "Paired difference vs Offer-One"]],
    )

    xs = [[r["kappa"] for r in rows] for _ in POLICIES]
    labels = [MF.POLICY_LABELS.get(p, p) for p in POLICIES]

    levels = [[r["policies"][p]["normalized"] for r in rows] for p in POLICIES]
    level_se = [[r["policies"][p]["se"] for r in rows] for p in POLICIES]
    plot_line(ax[0][0], xs, levels, level_se, labels, MF.LINE_FORMAT)
    plot_scatter(ax[0][0], xs, levels, [], MF.SCATTER_FORMAT)

    # offer-one is its own baseline, so it contributes the zero line rather
    # than being dropped -- that is what keeps colours aligned across panels.
    diffs = [[r["policies"][p].get("vs_offer_one_per_eps", 0.0) for r in rows]
             for p in POLICIES]
    diff_se = [[r["policies"][p].get("vs_offer_one_per_eps_se", 0.0) for r in rows]
               for p in POLICIES]
    plot_line(ax[0][1], xs, diffs, diff_se, labels, MF.LINE_FORMAT)
    plot_scatter(ax[0][1], xs, diffs, [], MF.SCATTER_FORMAT)
    ax[0][1].axhline(0, color="k", lw=0.8, zorder=0)

    for col in (0, 1):
        # the same axis in raw utility units; epsilon is constant here, so
        # this is a pure rescale rather than a second variable
        sec = ax[0][col].secondary_xaxis(
            "top", functions=(lambda k: k * epsilon, lambda g: g / epsilon))
        sec.set_xlabel("top-1 $-$ top-2 (utility units)", fontsize=10)
        sec.tick_params(labelsize=9)
        if kstar is not None:
            ax[0][col].axvline(kstar, color="0.35", lw=1.0, ls=":", zorder=0)
    if kstar is not None:
        # parked in the empty band between the SAM and Capacity-Greedy curves;
        # anywhere above the zero line is crossed by four of the six series
        ax[0][1].annotate(f"SAM $=$ Offer-One\nat $\\kappa$ = {kstar:.2f}",
                          xy=(kstar, 0), xytext=(1.15, -0.55),
                          fontsize=9, color="0.25",
                          arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))

    fig.subplots_adjust(wspace=0.28)   # see make_figures.metric_panels
    MF._legend(fig, ax, dims, ncol=3, anchor=(0.48, -0.22))

    sub = (f"M={cfg['M']} providers x 1 slot | {cfg['n_core']} core + "
           f"{cfg['n_crowd']} crowd patients | k={cfg['k']} | "
           f"{len(cfg['seeds'])} seeds x {cfg['num_trials']} trials")
    fig.text(0.5, 1.10, sub, ha="center", fontsize=9, color="0.35")

    # `MF._save` closes the figure, so it can only be the last write; the
    # figure is built once here rather than once per extension.
    for ext in ("png",):
        fig.savefig(HERE / f"instance_family_spread.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"saved {HERE / f'instance_family_spread.{ext}'}")
    MF._save(fig, HERE / "instance_family_spread.pdf")

    print(f"\nspread sweep at epsilon = {epsilon:g}, kappa "
          f"{rows[0]['kappa']:g} -> {rows[-1]['kappa']:g}")
    if kstar is not None:
        print(f"SAM overtakes/loses to offer-one at kappa* = {kstar:.2f}")


if __name__ == "__main__":
    main()
