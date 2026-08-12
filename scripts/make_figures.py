"""Figure generation from run_experiments.py's saved JSON results.

Loads `results/<experiment>/<policy>__<slug>.json` files (one per
(policy, swept-param point), written by `run_experiments.py::save_result`)
and renders each figure as a PDF under `results/figures/`. A figure is
silently skipped (with a message) if its experiment directory hasn't been
run yet.

Styling comes entirely from `patient.plotting`, which is a verbatim copy of
the reference implementation's plotting module, driven here with the same
formatting dicts its notebook used -- same colour scheme ('six_color'),
marker sequence, figure proportions, hidden top/right spines, global legend
below the axes, and `dpi=300, bbox_inches='tight'` on save.

Error bars are the across-seed standard error (`<metric>_se` = sd across
seeds / sqrt(num_seeds)), not the across-seed sd and not a pooled
across-trial or across-patient error -- see `run_experiments.aggregate_seeds`
for why seeds are the unit of replication.

Output names describe the content rather than the paper's figure numbers;
`FIGURES` maps each short selector to its output file(s).

Usage:
    python scripts/make_figures.py --figure main_comparison
    python scripts/make_figures.py --figure all --results-dir results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from patient.plotting import (color_schemes, create_axes, create_legend,
                              markers as plot_markers, plot_bar, plot_kde,
                              plot_line, plot_scatter)

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = REPO_DIR / "data"

POLICY_ORDER = ["random", "offer_one", "offer_all", "sam", "deferred_acceptance",
                "capacity_greedy", "exact_saa_milp", "offer_everything", "sam_uncapped"]
POLICY_LABELS = {
    "random": "Random", "offer_one": "Offer-One", "offer_all": "Offer-All", "sam": "SAM",
    "deferred_acceptance": "Deferred Acceptance", "capacity_greedy": "Capacity-Greedy",
    "exact_saa_milp": "SAA-MILP",
    "offer_everything": "Offer-Everything ($k{=}M$)", "sam_uncapped": "SAM ($k{=}M$)",
}
# The reference implementation colours by position in its method list, not by
# policy identity, so a figure's colours depend on which policies it shows.
# Keeping 'six_color' and the same policy ordering reproduces that.
PALETTE = "six_color"
LINE_FORMAT = {"color_palette": PALETTE, "linewidth": 2}
SCATTER_FORMAT = {"color_palette": PALETTE, "size": 75}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _parse_slug(stem):
    """'policy__key1=val1__key2=val2' -> (policy, {key1: val1, ...}); values
    are cast to int/float where possible (matches `run_experiments._slug`)."""
    parts = stem.split("__")
    policy = parts[0]
    point = {}
    for part in parts[1:]:
        if part == "default":
            continue
        key, _, val = part.partition("=")
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        point[key] = val
    return policy, point


def load_experiment(exp_dir):
    """Returns a list of dicts: {"policy": ..., **point-from-filename,
    **saved-json-content} for every result file in `exp_dir`. Empty list
    (not an error) if the directory doesn't exist -- lets callers skip
    figures for experiments that haven't been run yet."""
    exp_dir = Path(exp_dir)
    records = []
    if not exp_dir.exists():
        return records
    for f in sorted(exp_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue  # _progress.jsonl and friends
        policy, point = _parse_slug(f.stem)
        content = json.loads(f.read_text())
        records.append({"policy": policy, **point, **content})
    return records


def _sorted_policies(records, order=POLICY_ORDER):
    present = {r["policy"] for r in records}
    return [p for p in order if p in present] + sorted(present - set(order))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _legend(fig, ax, dims, ncol, anchor=(0.48, -0.2), show_point=True,
            loc="upper center"):
    create_legend(fig, ax, dims, {
        "style_size": "paper", "type": "is_global", "loc": loc,
        "ncol": ncol, "bbox_to_anchor": anchor, "show_point": show_point,
    })


# ---------------------------------------------------------------------------
# Shared builders: most figures are "one metric swept over one param, one line
# per policy", optionally as several side-by-side panels.
# ---------------------------------------------------------------------------
def _series(records, policy, x_key, y_key):
    pts = sorted((r for r in records if r["policy"] == policy), key=lambda r: r[x_key])
    xs = [r[x_key] for r in pts]
    ys = [r["agg"][y_key] for r in pts]
    err = [r["agg"].get(f"{y_key}_se", 0.0) for r in pts]
    return xs, ys, err


def line_panels(panels, x_key, y_key, xlabels, ylabel, save_path,
                titles=None, policies=None, figsize=None, log_x=False):
    """One panel per entry in `panels` (a list of record lists), one line per
    policy within each panel."""
    panels = [p for p in panels]
    if not any(panels):
        print(f"skip {save_path.name}: no data")
        return
    policies = policies or _sorted_policies([r for p in panels for r in p])
    n = len(panels)
    dims = (1, n)
    fig, ax = create_axes(
        dims,
        {"figsize": figsize or (5 * n, 2), "style_size": "paper",
         "hide_spines": True, "has_grid": True},
        x_labels=[xlabels],
        y_labels=[[ylabel] + [""] * (n - 1)],
        titles=[titles] if titles else None,
    )
    for col, records in enumerate(panels):
        if not records:
            continue
        xs, ys, errs, labels = [], [], [], []
        for policy in policies:
            x, y, e = _series(records, policy, x_key, y_key)
            if not x:
                continue
            xs.append(x)
            ys.append(y)
            errs.append(e)
            labels.append(POLICY_LABELS.get(policy, policy))
        plot_line(ax[0][col], xs, ys, errs, labels, LINE_FORMAT)
        plot_scatter(ax[0][col], xs, ys, [], SCATTER_FORMAT)
        if log_x:
            ax[0][col].set_xscale("log")
    _legend(fig, ax, dims, ncol=len(policies))
    _save(fig, save_path)


def metric_panels(records, x_key, y_keys, xlabel, ylabels, save_path,
                  policies=None, figsize=None, log_x=False, log_y=None,
                  legend_ncol=None):
    """`line_panels`' sibling: one panel per METRIC of a single record set,
    rather than one panel per record set of a single metric. Used when two
    quantities swept over the same x belong side by side (utility and the menu
    size that produced it).

    A policy with no value for a panel's metric is simply absent from that
    panel -- which is why `policies` order matters: `patient.plotting` assigns
    colour and marker by position in the list it receives, so policies present
    in every panel should be listed FIRST, and the partial ones last, or a
    policy changes colour from panel to panel."""
    if not records:
        print(f"skip {save_path.name}: no data")
        return
    policies = policies or _sorted_policies(records)
    n = len(y_keys)
    dims = (1, n)
    fig, ax = create_axes(
        dims,
        {"figsize": figsize or (5 * n, 2), "style_size": "paper",
         "hide_spines": True, "has_grid": True},
        x_labels=[[xlabel] * n],
        y_labels=[list(ylabels)],
    )
    for col, y_key in enumerate(y_keys):
        xs, ys, errs, labels = [], [], [], []
        for policy in policies:
            pts = sorted((r for r in records
                          if r["policy"] == policy and y_key in r["agg"]),
                         key=lambda r: r[x_key])
            if not pts:
                continue
            xs.append([r[x_key] for r in pts])
            ys.append([r["agg"][y_key] for r in pts])
            errs.append([r["agg"].get(f"{y_key}_se", 0.0) for r in pts])
            labels.append(POLICY_LABELS.get(policy, policy))
        plot_line(ax[0][col], xs, ys, errs, labels, LINE_FORMAT)
        plot_scatter(ax[0][col], xs, ys, [], SCATTER_FORMAT)
        if log_x:
            ax[0][col].set_xscale("log")
        if log_y and log_y[col]:
            ax[0][col].set_yscale("log")
    # Every panel carries its own y label (unlike `line_panels`, where only
    # the first does), so the default spacing puts the right panel's label on
    # top of the left panel's ticks. Widen the gap explicitly rather than with
    # tight_layout(): the global legend is an out-of-axes figure artist, and
    # tight_layout lays out as though it were not there, which then clips the
    # y labels once `_save` applies bbox_inches='tight'.
    fig.subplots_adjust(wspace=0.35)
    _legend(fig, ax, dims, ncol=legend_ncol or len(policies), anchor=(0.48, -0.25))
    _save(fig, save_path)


def bar_panels(panels, y_keys, titles, save_path, policies=None, figsize=(9, 1.5)):
    """One panel per (y_key, title); one bar per policy within each panel.
    Bars are grouped the reference way: each policy is its own bar 'group', so
    it picks up its own palette colour and is identified by the legend rather
    than by an x tick."""
    records_all = [r for p in panels for r in p]
    if not records_all:
        print(f"skip {save_path.name}: no data")
        return
    policies = policies or _sorted_policies(records_all)
    n = len(y_keys)
    dims = (1, n)
    fig, ax = create_axes(
        dims,
        {"figsize": figsize, "style_size": "paper", "hide_spines": True,
         "x_ticks": [[[[], []] for _ in range(n)]]},
        x_labels=[["" for _ in range(n)]],
        titles=[titles],
    )
    bar_format = {"style_size": "paper", "color_palette": PALETTE,
                  "label_rotation": 0, "bar_width": 0.5}
    labels = {i: POLICY_LABELS.get(p, p) for i, p in enumerate(policies)}
    for col, (records, y_key) in enumerate(zip(panels, y_keys)):
        by_policy = {r["policy"]: r for r in records}
        values = [by_policy[p]["agg"].get(y_key, np.nan) if p in by_policy else np.nan
                  for p in policies]
        errors = [by_policy[p]["agg"].get(f"{y_key}_se", 0.0) if p in by_policy else 0.0
                  for p in policies]
        plot_bar(ax[0][col], list(range(len(policies))), values, errors, labels, bar_format)
    # Reference notebook's own sequence for its bar figure: tight_layout
    # first, then a global legend anchored below-right (line figures anchor
    # it below-centre instead). tight_layout must come first or the legend
    # is placed against the pre-layout axes and lands on top of them.
    plt.tight_layout()
    _legend(fig, ax, dims, ncol=len(policies) + 1, anchor=(0.85, 0.15),
            show_point=False, loc="upper right")
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
MAIN_FOUR = ["random", "offer_one", "offer_all", "sam"]


def make_main_comparison(results_dir, out_dir):
    """Default config, the three headline metrics side by side: normalized
    utility, choice utility, and the 25th-percentile-by-population ZIP match
    rate."""
    records = load_experiment(results_dir / "default")
    present = [p for p in MAIN_FOUR if p in {r["policy"] for r in records}]
    bar_panels(
        [records, records, records],
        ["normalized_utility", "choice_utility_mean", "zip_fairness_p25"],
        ["Norm. Utility", "Choice Utility", "Fairness (25th pct.)"],
        out_dir / "main_comparison.pdf",
        policies=present,
    )


def make_fairness_gini(results_dir, out_dir):
    """Geographic Gini across ZIP codes, under both readings of the metric
    (see `metrics.zip_fairness`): spread of per-ZIP match rates, and spread of
    per-ZIP mean utility."""
    records = load_experiment(results_dir / "default")
    present = [p for p in MAIN_FOUR if p in {r["policy"] for r in records}]
    bar_panels(
        [records, records],
        ["zip_gini_match_rate", "zip_gini_utility"],
        ["Geographic Gini (match rate)", "Geographic Gini (utility)"],
        out_dir / "fairness_gini.pdf",
        policies=present,
        figsize=(7, 1.5),
    )


def make_noise_sweep(results_dir, out_dir):
    """Effect of the noise level epsilon: approximation ratio against the
    exact SAA-MILP at small scale, and normalized utility at full scale."""
    left = load_experiment(results_dir / "noise_exact_small")
    if left:
        milp = {r["epsilon"]: r["agg"]["utility"]
                for r in left if r["policy"] == "exact_saa_milp"}
        for r in left:
            if r["epsilon"] in milp and milp[r["epsilon"]]:
                r["agg"]["approx_ratio"] = r["agg"]["utility"] / milp[r["epsilon"]]
                r["agg"]["approx_ratio_std"] = (r["agg"].get("utility_std", 0.0)
                                                / milp[r["epsilon"]])
        left = [r for r in left if r["policy"] != "exact_saa_milp"
                and "approx_ratio" in r["agg"]]
    line_panels([left], "epsilon", "approx_ratio",
                ["Noise Level ($\\epsilon$)"], "Approx. Ratio",
                out_dir / "approximation_ratio_vs_noise.pdf",
                figsize=(5.5, 2))

    right = load_experiment(results_dir / "noise")
    line_panels([right], "epsilon", "normalized_utility",
                ["Noise Level ($\\epsilon$)"], "Norm. Utility",
                out_dir / "utility_vs_noise.pdf", figsize=(5.5, 2),
                policies=MAIN_FOUR)


def make_noise_uncapped(results_dir, out_dir):
    """The noise sweep with the k=25 policies and the two uncapped (k=M) ones
    on the same axes: what the menu-size budget is actually costing.

    The second panel is how big the menus actually got. It is the reason the
    first one looks the way it does, and it is the only place SAM's menu size
    is plotted: SAM is the one policy whose menu is not simply min(k, live),
    because Algorithm 1's `m_ij > 0` test can stop short of the budget.

    Six lines is the palette's limit -- `patient.plotting` asserts on a
    seventh -- so this stays at the four main policies plus the two uncapped
    ones."""
    records = load_experiment(results_dir / "noise")
    uncapped = load_experiment(results_dir / "noise_uncapped")
    if not uncapped:
        print("skip utility_vs_noise_uncapped.pdf: no data")
        return
    metric_panels([r for r in records if r["policy"] in MAIN_FOUR] + uncapped,
                  "epsilon", ["normalized_utility", "menu_planned"],
                  "Noise Level ($\\epsilon$)", ["Norm. Utility", "Menu Size"],
                  out_dir / "utility_vs_noise_uncapped.pdf",
                  policies=MAIN_FOUR + ["offer_everything", "sam_uncapped"],
                  figsize=(9, 2.2), log_y=[False, True], legend_ncol=3)


def make_menu_budget(results_dir, out_dir):
    """Menu budget k from 1 to M at the default epsilon: utility against k,
    and the menu size each policy actually plans against k.

    The second panel is the point of the figure. k is only an upper bound --
    offer_one ignores it, and SAM's `m_ij > 0` test can stop well short of it
    -- so "menu size" and "k" are the same number only for the policies that
    always fill their budget. Both axes are log: k spans 1 to 700."""
    records = load_experiment(results_dir / "menu_budget")
    metric_panels(records, "k", ["normalized_utility", "menu_planned"],
                  "Menu Budget ($k$)", ["Norm. Utility", "Menu Size"],
                  out_dir / "utility_vs_menu_budget.pdf",
                  policies=MAIN_FOUR, figsize=(9, 2.2),
                  log_x=True, log_y=[False, True])


def _decomposition(records, policy):
    """Per-epsilon mean and SE of the three additive-error terms for one
    policy, computed from PER-SEED values and only then averaged.

    Every term is a difference of quantities the same seed produced, so
    differencing first is what makes them paired -- the seed-to-seed spread of
    OPT alone (SE ~0.004) dwarfs the spread of the differences, and averaging
    first would drown the effect in it."""
    eps = sorted({r["epsilon"] for r in records if r["policy"] == policy})
    cols = {key: ([], []) for key in ("total", "coverage", "coordination")}
    for e in eps:
        rec = next(r for r in records
                   if r["policy"] == policy and r["epsilon"] == e)
        terms = {key: [] for key in cols}
        for row in rec["per_seed"]:
            opt, lp, v = (row["omniscient_utility"], row["lp_menu_utility"],
                          row["utility"])
            terms["total"].append(opt - v)           # whole shortfall
            terms["coverage"].append(opt - lp)       # the menu lacks the pairs
            terms["coordination"].append(lp - v)     # patients don't coordinate
        for key, vals in terms.items():
            vals = np.asarray(vals)
            cols[key][0].append(vals.mean())
            cols[key][1].append(vals.std(ddof=1) / np.sqrt(len(vals)))
    return eps, cols


def make_error_decomposition(results_dir, out_dir):
    """Each policy's additive shortfall against the omniscient, split into the
    two things a menu policy can get wrong.

        OPT(theta) - V  =  [OPT(theta) - LP(X, theta)]  +  [LP(X, theta) - V]
                                 coverage                    coordination

    LP(X, theta) is the best assignment reachable inside the menu X the policy
    committed to, if the realized theta were known. Both brackets are
    non-negative (see `metrics.menu_restricted_lp_utility`), so the split is a
    true partition of the error and stacks.

    COVERAGE is what the menu fails to contain: offer_one's one column per
    patient is the extreme, and a menu large enough to hold the omniscient
    assignment drives it to zero. COORDINATION is what uncoordinated choice
    costs even when the menu does contain a good assignment: patients arrive
    in random order and each takes their own best available option. The two
    trade off directly -- widening a menu can only reduce coverage loss and
    can only increase the scope for patients to collide.

    Four panels ON A COMMON Y SCALE, which is the whole point of the layout:
    the three line panels (total, then each term) are only comparable by eye
    if a centimetre means the same error everywhere, and the shared scale is
    what shows at a glance that SAM's coverage panel is empty while its
    coordination panel carries its whole error. The fourth panel stacks the
    two terms so the sum back to the total is visible.

    Two policies are load-bearing checks: offer_one's coordination term must
    be 0 to numerical precision, and offer_all's coverage term must be small.
    """
    records = load_experiment(results_dir / "error_decomposition")
    if not records:
        print("skip error_decomposition.pdf: no data")
        return
    if "lp_menu_utility" not in records[0]["per_seed"][0]:
        print("skip error_decomposition.pdf: results predate lp_menu_utility; "
              "re-run `--experiment error_decomposition`")
        return

    policies = [p for p in MAIN_FOUR if p in {r["policy"] for r in records}]
    colors = color_schemes[PALETTE]
    series = {p: _decomposition(records, p) for p in policies}
    eps = series[policies[0]][0]

    dims = (1, 4)
    xlabel = "Noise Level ($\\epsilon$)"
    fig, ax = create_axes(
        dims,
        {"figsize": (15, 2.8), "style_size": "paper", "hide_spines": True,
         "has_grid": True},
        x_labels=[[xlabel] * 4],
        y_labels=[["Additive Error", "", "", ""]],
        titles=[["Total:  $OPT(\\theta) - V$",
                 "Coverage:  $OPT(\\theta) - LP(X)$",
                 "Coordination:  $LP(X) - V$",
                 "Stacked"]],
    )

    # ---- panels 1-3: one term each, one line per policy --------------------
    for col, key in enumerate(("total", "coverage", "coordination")):
        xs = [eps] * len(policies)
        ys = [series[p][1][key][0] for p in policies]
        errs = [series[p][1][key][1] for p in policies]
        labels = [POLICY_LABELS[p] for p in policies]
        plot_line(ax[0][col], xs, ys, errs, labels, LINE_FORMAT)
        plot_scatter(ax[0][col], xs, ys, [], SCATTER_FORMAT)

    # ---- panel 4: the two terms stacked, so they visibly sum to the total ---
    # Groups are spaced by index, not by the epsilon value, so the uneven grid
    # (0.01 then 0.1, 0.2, ...) does not squash the first group against the
    # second.
    width = 0.8 / len(policies)
    idx = np.arange(len(eps))
    for i, policy in enumerate(policies):
        cols = series[policy][1]
        offset = (i - (len(policies) - 1) / 2) * width
        cov = np.array(cols["coverage"][0])
        coord = np.array(cols["coordination"][0])
        ax[0][3].bar(idx + offset, cov, width * 0.9, color=colors[i],
                     edgecolor="white", linewidth=0.4, zorder=2)
        ax[0][3].bar(idx + offset, coord, width * 0.9, bottom=cov, color=colors[i],
                     edgecolor="white", linewidth=0.4, hatch="///", alpha=0.55,
                     zorder=2)
    ax[0][3].set_xticks(idx)
    ax[0][3].set_xticklabels([f"{e:g}" for e in eps])

    # ---- the common scale --------------------------------------------------
    # Taken from the data and rounded up rather than pinned at a round 0.2:
    # offer_one's total reaches 0.249, and a shared axis that clipped the
    # single largest error in the figure would be worse than a slightly less
    # tidy limit.
    top = max(max(series[p][1]["total"][0]) for p in policies)
    ymax = np.ceil(top / 0.02) * 0.02
    for col in range(4):
        ax[0][col].set_ylim(0, ymax)
        if col:
            ax[0][col].set_yticklabels([])

    fig.subplots_adjust(wspace=0.12)
    # Built by hand: the legend has to name two different encodings (policy
    # colour, and term-as-hatch), which `create_legend` -- which reads handles
    # off one axis and assumes one entry per line -- cannot do. Styling
    # matches it (global figure legend, size 12, anchored below).
    handles = [Line2D([0], [0], color=colors[i], marker=plot_markers[i], linewidth=2)
               for i in range(len(policies))]
    labels = [POLICY_LABELS[p] for p in policies]
    handles += [Patch(facecolor="0.45", edgecolor="white"),
                Patch(facecolor="0.45", edgecolor="white", hatch="///", alpha=0.55)]
    labels += ["Coverage (stacked panel)", "Coordination (stacked panel)"]
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, -0.13), fontsize=12)
    _save(fig, out_dir / "error_decomposition.pdf")


def make_market_dynamics(results_dir, out_dir):
    """Congestion: patient/provider ratio and average provider capacity."""
    ratio = load_experiment(results_dir / "patient_provider_ratio")
    for r in ratio:
        r["ratio_NM"] = r["N"] / r["M"]
    cap = load_experiment(results_dir / "provider_capacity")
    # Two separate files rather than one two-panel figure: the panels sweep
    # different x keys, and `line_panels` takes one x key per call.
    line_panels([ratio], "ratio_NM", "normalized_utility",
                ["Patient/Provider (N/M)"], "Norm. Utility",
                out_dir / "utility_vs_patient_provider_ratio.pdf", figsize=(5.5, 2),
                policies=MAIN_FOUR)
    line_panels([cap], "avg_capacity", "normalized_utility",
                ["Provider Capacity ($\\bar{c}$)"], "Norm. Utility",
                out_dir / "utility_vs_provider_capacity.pdf", figsize=(5.5, 2))


def make_choice_model(results_dir, out_dir):
    """Choice-model robustness: MNL responses, and the value of the exit
    option."""
    mnl = load_experiment(results_dir / "mnl_choice")
    present = [p for p in MAIN_FOUR if p in {r["policy"] for r in mnl}]
    bar_panels([mnl], ["normalized_utility"], ["MNL Choice Model"],
               out_dir / "utility_mnl_choice_model.pdf",
               policies=present, figsize=(4, 1.5))

    exits = load_experiment(results_dir / "exit_option")
    line_panels([exits], "exit_utility", "normalized_utility",
                ["Exit Option ($\\theta_{i,M+1}$)"], "Norm. Utility",
                out_dir / "utility_vs_exit_option.pdf", figsize=(5.5, 2))


def make_choice_distributions(results_dir, out_dir):
    """Per-patient number of effective choices and choice utility, as
    densities, one row per policy -- rebuilt from each seed's saved histogram
    (see run_experiments._fig6_extra_metrics)."""
    records = load_experiment(results_dir / "default")
    if not records:
        print("skip choice_distributions.pdf: no data")
        return
    policies = [p for p in MAIN_FOUR if p in {r["policy"] for r in records}]
    by_policy = {r["policy"]: r for r in records}
    dims = (len(policies), 2)
    fig, ax = create_axes(
        dims,
        {"figsize": (9, 1.5 * len(policies)), "style_size": "paper",
         "hide_spines": True},
        y_labels=[[POLICY_LABELS.get(p, p), ""] for p in policies],
        titles=[["#Choices", "Choice Utility"]] + [["", ""]] * (len(policies) - 1),
    )
    from patient.plotting import color_schemes
    colors = color_schemes[PALETTE]
    for row, policy in enumerate(policies):
        cc_sum = cu_sum = cc_edges = cu_edges = None
        for seed_rec in by_policy[policy]["per_seed"]:
            extra = seed_rec.get("extra")
            if extra is None:
                continue
            cc = np.array(extra["choice_count_hist"], dtype=float)
            cu = np.array(extra["choice_utility_hist"], dtype=float)
            cc_sum = cc if cc_sum is None else cc_sum + cc
            cu_sum = cu if cu_sum is None else cu_sum + cu
            cc_edges, cu_edges = extra["choice_count_edges"], extra["choice_utility_edges"]
        if cc_sum is None:
            continue
        for col, (hist, edges) in enumerate([(cc_sum, cc_edges), (cu_sum, cu_edges)]):
            widths = np.diff(edges)
            density = hist / (hist.sum() * widths)
            ax[row][col].bar(edges[:-1], density, width=widths, align="edge",
                             color=colors[row])
        ax[row][0].set_xlim(0, 26)
        ax[row][1].set_xlim(0, 1)
    _save(fig, out_dir / "choice_distributions.pdf")


HARTFORD = (41.7637, -72.6851)
NEW_HAVEN = (41.3083, -72.9279)


def make_zipcode_match_rate(results_dir, out_dir):
    """Connecticut choropleth of per-ZIP match rate, one panel per policy --
    the paper's Figure 7. Reproduces the reference notebook's construction:
    `data/ct.geojson` keyed on ZCTA5CE10, a Blues colormap with a grey
    zero-th entry so ZIPs with no sampled patients (assigned -1, below vmin)
    render grey, vmin/vmax pinned to [0,1] so panels are comparable, Hartford
    and New Haven marked, and one shared colorbar."""
    records = load_experiment(results_dir / "default")
    if not records:
        print("skip match_rate_by_zipcode.pdf: no data")
        return
    try:
        import geopandas as gpd
    except ImportError:
        print("skip match_rate_by_zipcode.pdf: geopandas not installed")
        return
    import matplotlib.colors as mcolors

    policies = [p for p in ["offer_one", "offer_all", "sam"]
                if p in {r["policy"] for r in records}]
    by_policy = {r["policy"]: r for r in records}
    # from the repo, not from results_dir -- results can live anywhere
    ct = gpd.read_file(DATA_DIR / "ct.geojson")

    for policy in policies:
        zip_rates = {}
        for seed_rec in by_policy[policy]["per_seed"]:
            for z, rate in ((seed_rec.get("extra") or {}).get("zip_match_rate") or {}).items():
                zip_rates.setdefault(z, []).append(rate)
        # -1 for ZIPs no patient was sampled into: below vmin, so the colormap's
        # grey zero-th entry picks them up rather than the bottom of the ramp.
        ct[policy] = [np.mean(zip_rates[z]) if z in zip_rates else -1
                      for z in ct["ZCTA5CE10"]]

    cmap = mcolors.ListedColormap(["grey"] + plt.cm.Blues(np.linspace(0, 1, 256)).tolist())
    fig, axes = plt.subplots(1, len(policies), figsize=(16, 5))
    axes = np.atleast_1d(axes)
    for ax, policy in zip(axes, policies):
        ct.plot(column=policy, ax=ax, legend=False, cmap=cmap, vmin=0, vmax=1,
                missing_kwds={"color": "lightgrey"})
        ax.set_title(f"Match Rate ({POLICY_LABELS.get(policy, policy)})", fontsize=14)
        ax.axis("off")
        for (lat, lon), name in [(HARTFORD, "Hartford"), (NEW_HAVEN, "New Haven")]:
            ax.scatter(lon, lat, color="black", s=25, zorder=5)
            ax.text(lon + 0.07, lat + 0.03, name, color="black", fontsize=12, zorder=6)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    _save(fig, out_dir / "match_rate_by_zipcode.pdf")


def make_utility_operationalization(results_dir, out_dir):
    """EC.2.1: the two knobs behind theta -- the distance threshold d_bar and
    the comorbidity weight omega."""
    dist = load_experiment(results_dir / "distance_threshold")
    omega = load_experiment(results_dir / "comorbidity_weight")
    line_panels([dist], "average_distance", "normalized_utility",
                ["Average Distance ($\\bar{d}$, miles)"], "Norm. Utility",
                out_dir / "utility_vs_average_distance.pdf", figsize=(5.5, 2))
    line_panels([omega], "omega", "normalized_utility",
                ["Comorbidity Weight ($\\omega$)"], "Norm. Utility",
                out_dir / "utility_vs_comorbidity_weight.pdf", figsize=(5.5, 2))


def make_sample_count(results_dir, out_dir):
    """EC.2.2: how many scenarios S SAM needs."""
    records = load_experiment(results_dir / "sam_samples")
    line_panels([records], "S", "normalized_utility",
                ["Samples (S)"], "Norm. Utility",
                out_dir / "utility_vs_num_samples.pdf", figsize=(5.5, 2))


def make_theta_distributions(results_dir, out_dir):
    """EC.2.3: uniform / normal / latent theta."""
    records = load_experiment(results_dir / "theta_distribution")
    if not records:
        print("skip utility_by_theta_distribution.pdf: no data")
        return
    dists = ["uniform", "normal", "latent"]
    present = [p for p in MAIN_FOUR if p in {r["policy"] for r in records}]
    bar_panels(
        [[r for r in records if r.get("distribution") == d] for d in dists],
        ["normalized_utility"] * len(dists),
        [d.capitalize() for d in dists],
        out_dir / "utility_by_theta_distribution.pdf",
        policies=present, figsize=(9, 1.5),
    )


def make_theta_spread(results_dir, out_dir):
    """How far apart a patient's options are, at fixed epsilon (not in the
    paper). The general-instance version of `tests/debugging/
    instance_family.py`'s left panel.

    The x axis is the MEASURED spread -- the mean top-1-to-top-k gap in
    theta_hat, in units of epsilon -- rather than the `spread` multiplier that
    produced it. The multiplier is an input to a generator whose output
    dispersion depends on the distribution; the gap is the thing the policies
    actually see, and it is what makes this curve comparable to the toy
    family's, which is indexed the same way."""
    records = load_experiment(results_dir / "theta_spread")
    if not records:
        print("skip utility_vs_theta_spread.pdf: no data")
        return
    for r in records:
        r["spread_kappa"] = r["agg"]["spread_kappa"]
    present = _sorted_policies(records)
    eps = {r["epsilon"] for r in records}
    title = f"$\\epsilon$ fixed at {eps.pop():g}" if len(eps) == 1 else None
    line_panels([records], "spread_kappa", "normalized_utility",
                ["Option Spread (top-1 $-$ top-$k$, in units of $\\epsilon$)"],
                "Norm. Utility", out_dir / "utility_vs_theta_spread.pdf",
                titles=[title] if title else None,
                policies=present, figsize=(5.5, 2))


def make_menu_size(results_dir, out_dir):
    """EC.2.4: noise sweep at three menu-size budgets k."""
    records = load_experiment(results_dir / "menu_size")
    if not records:
        print("skip utility_vs_noise_by_menu_size.pdf: no data")
        return
    ks = sorted({r["k"] for r in records})
    line_panels([[r for r in records if r["k"] == k] for k in ks],
                "epsilon", "normalized_utility",
                ["Noise ($\\epsilon$)"] * len(ks), "Norm. Utility",
                out_dir / "utility_vs_noise_by_menu_size.pdf",
                titles=[f"k = {k}" for k in ks], figsize=(4 * len(ks), 2))


def make_stronger_baselines(results_dir, out_dir):
    """EC.2.5: deferred acceptance and capacity-greedy added to the
    patient/provider-ratio and noise sweeps."""
    ratio = load_experiment(results_dir / "patient_provider_ratio")
    for r in ratio:
        r["ratio_NM"] = r["N"] / r["M"]
    eps = load_experiment(results_dir / "noise")
    line_panels([ratio], "ratio_NM", "normalized_utility",
                ["Patient/Provider (N/M)"], "Norm. Utility",
                out_dir / "stronger_baselines_vs_ratio.pdf", figsize=(5.5, 2))
    line_panels([eps], "epsilon", "normalized_utility",
                ["Noise ($\\epsilon$)"], "Norm. Utility",
                out_dir / "stronger_baselines_vs_noise.pdf", figsize=(5.5, 2))


def make_runtime(results_dir, out_dir):
    """EC.2.6: SAM's single-instance runtime against N and S. Uses the
    per-seed mean `runtime_sec`, not `total_runtime_sec`."""
    n_run = load_experiment(results_dir / "scale")
    s_run = load_experiment(results_dir / "sam_samples")
    line_panels([n_run], "N", "runtime_sec", ["N (Patients)"], "Runtime (s)",
                out_dir / "runtime_vs_patients.pdf", figsize=(5.5, 2))
    line_panels([s_run], "S", "runtime_sec", ["S (Samples)"], "Runtime (s)",
                out_dir / "runtime_vs_samples.pdf", figsize=(5.5, 2))


FIGURES = {
    "main_comparison": make_main_comparison,
    "fairness_gini": make_fairness_gini,
    "noise_sweep": make_noise_sweep,
    "noise_uncapped": make_noise_uncapped,
    "error_decomposition": make_error_decomposition,
    "menu_budget": make_menu_budget,
    "market_dynamics": make_market_dynamics,
    "choice_model": make_choice_model,
    "choice_distributions": make_choice_distributions,
    "zipcode_match_rate": make_zipcode_match_rate,
    "utility_operationalization": make_utility_operationalization,
    "sample_count": make_sample_count,
    "theta_distributions": make_theta_distributions,
    "theta_spread": make_theta_spread,
    "menu_size": make_menu_size,
    "stronger_baselines": make_stronger_baselines,
    "runtime": make_runtime,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", required=True, choices=list(FIGURES) + ["all"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = FIGURES if args.figure == "all" else {args.figure: FIGURES[args.figure]}
    for name, fn in targets.items():
        fn(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
