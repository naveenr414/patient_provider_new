"""utility_vs_noise, extended with the uncapped variants.

Two figures, both reading the standard sweep JSONs:

  utility_vs_noise_offer_everything.{pdf,png}
      the main-text figure (four baselines) + uncapped offer-everything,
      from `results/noise` and `results/noise_offer_everything`.

  utility_vs_noise_sam_variants.{pdf,png}
      SAM under the four (k, tau) rules from `results/noise_sam_uncapped`,
      with uncapped offer-everything as the reference. Two panels: normalized
      utility, and the planned menu size that explains it. Skipped with a
      message if that sweep hasn't been run.

Styling is `patient.plotting` via `make_figures`, so these are the project's
standard figures with extra curves. Two caveats on colour, both from
`plot_line` assigning colour and marker by a line's POSITION in the list it is
handed, out of a six-entry palette:

  - Each figure is coloured independently. SAM and offer-everything are not
    the same colour in both; read each legend separately.
  - Within the SAM figure, offer-everything is listed LAST precisely so the
    four SAM variants hold positions 0-3 in both panels and keep their colours
    across them, even though offer-everything appears in only one.

Run:  PYTHONPATH=. python tests/debugging/plot_offer_everything_noise.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patient.plotting import create_axes, plot_line, plot_scatter
from scripts import make_figures as MF

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MF.POLICY_LABELS.update({
    "offer_everything": "Offer-Everything ($k{=}M$)",
    "sam_k25": "SAM ($k{=}25$)",
    "sam_uncapped": "SAM ($k{=}M$)",
    "sam_k25_tau": "SAM ($k{=}25$, $\\tau{=}0.1\\epsilon$)",
    "sam_uncapped_tau": "SAM ($k{=}M$, $\\tau{=}0.1\\epsilon$)",
})

XLABEL = "Noise Level ($\\epsilon$)"
SAM_VARIANTS = ["sam_k25", "sam_uncapped", "sam_k25_tau", "sam_uncapped_tau"]


def multi_metric_panels(records, policies, y_keys, ylabels, save_path,
                        figsize=None, log_y=None):
    """`make_figures.line_panels`, but each panel shows a different metric of
    the same records rather than the same metric of different records. A
    policy with no value for a panel's metric is simply absent from it."""
    n = len(y_keys)
    dims = (1, n)
    fig, ax = create_axes(
        dims,
        {"figsize": figsize or (5 * n, 2), "style_size": "paper",
         "hide_spines": True, "has_grid": True},
        x_labels=[[XLABEL] * n],
        y_labels=[ylabels],
    )
    for col, y_key in enumerate(y_keys):
        xs, ys, errs, labels = [], [], [], []
        for policy in policies:
            pts = sorted((r for r in records if r["policy"] == policy),
                         key=lambda r: r["epsilon"])
            pts = [r for r in pts if y_key in r["agg"]]
            if not pts:
                continue
            xs.append([r["epsilon"] for r in pts])
            ys.append([r["agg"][y_key] for r in pts])
            errs.append([r["agg"].get(f"{y_key}_se", 0.0) for r in pts])
            labels.append(MF.POLICY_LABELS.get(policy, policy))
        plot_line(ax[0][col], xs, ys, errs, labels, MF.LINE_FORMAT)
        plot_scatter(ax[0][col], xs, ys, [], MF.SCATTER_FORMAT)
        if log_y and log_y[col]:
            ax[0][col].set_yscale("log")
    fig.subplots_adjust(wspace=0.35)   # see make_figures.metric_panels
    MF._legend(fig, ax, dims, ncol=3, anchor=(0.48, -0.28))
    MF._save(fig, save_path)


def main():
    noise = MF.load_experiment(ROOT / "results" / "noise")
    uncapped = MF.load_experiment(ROOT / "results" / "noise_offer_everything")
    variants = MF.load_experiment(ROOT / "results" / "noise_sam_uncapped")

    if not uncapped:
        raise SystemExit("no results/noise_offer_everything -- run "
                          "offer_everything_noise_sweep.py first")

    for ext in ("pdf", "png"):
        MF.line_panels(
            [[r for r in noise if r["policy"] in MF.MAIN_FOUR] + uncapped],
            "epsilon", "normalized_utility", [XLABEL], "Norm. Utility",
            HERE / f"utility_vs_noise_offer_everything.{ext}",
            figsize=(5.5, 2), policies=MF.MAIN_FOUR + ["offer_everything"])

    if not variants:
        print("skip utility_vs_noise_sam_variants: run "
              "sam_uncapped_noise_sweep.py first")
        return

    # offer-everything has no planned-menu-size field (its menu is, by
    # definition, every live provider), so it drops out of the right panel.
    for ext in ("pdf", "png"):
        multi_metric_panels(
            variants + uncapped, SAM_VARIANTS + ["offer_everything"],
            ["normalized_utility", "menu_planned"],
            ["Norm. Utility", "Menu Size"],
            HERE / f"utility_vs_noise_sam_variants.{ext}",
            figsize=(9, 2.2), log_y=[False, True])

    # How big SAM's menus get as the noise grows, at both budgets, and how
    # much of that survives to the patient. The two panels answer different
    # questions: menu_planned is what Algorithm 1 selected, choice_count_mean
    # is what was still uncommitted when the patient arrived (plus the exit
    # option). The gap between them is capacity depletion, not menu design.
    for ext in ("pdf", "png"):
        multi_metric_panels(
            variants, ["sam_k25", "sam_uncapped"],
            ["menu_planned", "choice_count_mean"],
            ["Planned Menu Size", "Options on Arrival"],
            HERE / f"sam_menu_size_vs_noise.{ext}",
            figsize=(9, 2.2), log_y=[True, False])


if __name__ == "__main__":
    main()
