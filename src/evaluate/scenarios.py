import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.simulate import (TARGET, TravelTables, allocate_units, load_calls, metrics, nearest_policy,
                                   simulate)
from src.evaluate.style import INK, INK_2, MUTED, SERIES, SURFACE
from src.placement.kmeans_depots import node_weights
from src.rl.qlearning import QLearner
from src.routing.router import Router

def scenarios(chosen):
    return [
        ("A · Static baseline", "Uniform grid", "distance", "nearest"),
        ("B · + K-means depots", "K-means (plain)", "distance", "nearest"),
        ("C · + ML demand depots & ML routing", chosen, "ml", "nearest"),
        ("D · RapidRoute (+ Q-learning)", chosen, "ml", "qlearning"),
    ]
COLORS = [MUTED, SERIES[3], SERIES[0], SERIES[2]]


def build(R, depots_all, weights, fleet, chosen):
    out = {}
    cache = {}
    for name, dep_key, mode, policy in scenarios(chosen):
        nodes = [d["node"] for d in depots_all[dep_key]]
        key = (dep_key, mode)
        if key not in cache:
            cache[key] = TravelTables(R, nodes, mode)
        out[name] = (cache[key], allocate_units(R, nodes, fleet, weights), policy)
    return out


def run(calls, tables, units, policy, kappa, Q):
    if policy == "qlearning":
        L = QLearner(len(units), 0.1, 0.9, seed=0)
        L.Q = Q.copy()
        L.train = False
        r, tr, w = simulate(calls, tables, units, kappa, learner=L)
    else:
        r, tr, w = simulate(calls, tables, units, kappa, nearest_policy)
    return r, tr, w


def plot_bars(res, observed, out):
    cols = [("mean_rt_min", "Mean response (min)", False), ("p90_rt_min", "P90 response (min)", False),
            ("pct_within_8", "% of calls ≤ 8 min", True), ("pct_critical_within_8", "% critical ≤ 8 min", True)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (col, lab, higher) in zip(axes, cols):
        v = res[col].values
        bars = ax.bar(range(len(v)), v, color=COLORS, edgecolor=SURFACE, linewidth=2, width=0.7)
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.1f}", ha="center", va="bottom", fontsize=9, color=INK_2)
        ax.set_xticks(range(len(v)), ["A", "B", "C", "D"])
        ax.set_title(lab + (" ↑" if higher else " ↓"), fontsize=11, pad=8)
        ax.set_ylim(0, v.max() * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, res.index, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Scenario comparison on the Nov–Dec 2024 holdout (63,806 real calls replayed)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "scenarios_comparison.png")
    plt.close(fig)


def plot_cdf(resp, out):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for (name, r), c in zip(resp.items(), COLORS):
        x = np.sort(r[np.isfinite(r)]) / 60
        ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=c, lw=2, label=name)
    ax.axvline(TARGET / 60, color=INK, ls="--", lw=1)
    ax.text(TARGET / 60 + 0.2, 5, "8-min target", fontsize=9, color=INK)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Response time (min)")
    ax.set_ylabel("% of calls reached")
    ax.legend(loc="lower right")
    ax.set_title("Share of calls reached within t minutes", pad=24)
    style.subtitle(ax, "Higher and further left is better")
    fig.savefig(out / "scenarios_cdf.png")
    plt.close(fig)


def plot_fleet(sens, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, lab in [(axes[0], "mean_rt_min", "Mean response (min)"), (axes[1], "pct_critical_within_8", "% critical ≤ 8 min")]:
        for (name, g), c in zip(sens.groupby("scenario", sort=False), COLORS):
            ax.plot(g.fleet, g[col], color=c, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.set_xlabel("Fleet size (ambulances)")
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(sens.fleet.unique()))
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].set_title("Sensitivity to fleet size", pad=24)
    style.subtitle(axes[0], "Gains are largest when the fleet is stretched")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "scenarios_fleet_sensitivity.png")
    plt.close(fig)


def plot_ablation(abl, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    names = abl.placement.unique()
    x = np.arange(len(names))
    for i, (mode, lab, c) in enumerate([("distance", "Shortest-distance routing", MUTED), ("ml", "ML travel-time routing", SERIES[0])]):
        v = abl[abl.routing == mode].set_index("placement").loc[names].pct_within_8.values
        bars = ax.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=c, edgecolor=SURFACE, linewidth=2, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, abl.pct_within_8.max() * 1.2)
    ax.set_ylabel("% of calls ≤ 8 min")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("What drives the gain: placement × routing", pad=24)
    style.subtitle(ax, "Nov–Dec replay · nearest-unit dispatch · same fleet everywhere")
    fig.savefig(out / "scenarios_ablation.png")
    plt.close(fig)


def main():
    style.apply()
    proc, figs = path("processed"), path("figures")
    cal = json.loads((proc / "calibration.json").read_text())
    kappa, fleet = cal["kappa"], cal["fleet"]
    R = Router(kappa=1.0)
    depots_all = json.loads((proc / "depots.json").read_text())["depots"]
    weights = node_weights(R)["demand"]
    Q = np.load(proc / "q_table.npy")
    chosen = json.loads((proc / "selection.json").read_text())["placement"]
    calls = load_calls()
    test = calls[calls.split == "test"].reset_index(drop=True)
    test["t"] = test.t - test.t.min()

    obs = {"mean_rt_min": test.response_s.mean() / 60, "median_rt_min": test.response_s.median() / 60,
           "p90_rt_min": test.response_s.quantile(0.9) / 60, "pct_within_8": (test.response_s <= TARGET).mean() * 100,
           "pct_critical_within_8": (test[test.sev == 0].response_s <= TARGET).mean() * 100}

    rows, resp = {}, {}
    for name, (tables, units, policy) in build(R, depots_all, weights, fleet, chosen).items():
        r, tr, w = run(test, tables, units, policy, kappa, Q)
        rows[name] = metrics(test, r, tr, w)
        resp[name] = r
        print(name, {k: round(v, 2) for k, v in rows[name].items()})
    res = pd.DataFrame(rows).T
    res.round(3).to_csv(proc.parent.parent / "reports" / "scenario_results.csv")

    sens = []
    for f in CFG["fleet_sensitivity"]:
        for name, (tables, units, policy) in build(R, depots_all, weights, f, chosen).items():
            r, tr, w = run(test, tables, units, policy, kappa, Q)
            sens.append({"scenario": name, "fleet": f, **metrics(test, r, tr, w)})
    sens = pd.DataFrame(sens)
    sens.round(3).to_csv(proc.parent.parent / "reports" / "scenario_fleet_sensitivity.csv", index=False)
    print(sens.round(2).to_string(index=False))

    abl = []
    for dk in ["Uniform grid", "K-means (plain)", "K-means (demand)", "K-means (risk-weighted)"]:
        nodes = [x["node"] for x in depots_all[dk]]
        units = allocate_units(R, nodes, fleet, weights)
        for mode in ["distance", "ml"]:
            r, tr, w = simulate(test, TravelTables(R, nodes, mode), units, kappa, nearest_policy)
            abl.append({"placement": dk, "routing": mode, **metrics(test, r, tr, w)})
    abl = pd.DataFrame(abl)
    abl.round(3).to_csv(proc.parent.parent / "reports" / "ablation_placement_routing.csv", index=False)
    plot_ablation(abl, figs)

    a, d = res.iloc[0], res.iloc[-1]
    summary = {"fleet": fleet, "kappa": kappa, "observed_test": obs, "scenarios": res.round(3).to_dict(orient="index"),
               "improvement_A_to_D": {"mean_rt_min": float(a.mean_rt_min - d.mean_rt_min),
                                      "pct_within_8_pts": float(d.pct_within_8 - a.pct_within_8),
                                      "pct_critical_within_8_pts": float(d.pct_critical_within_8 - a.pct_critical_within_8)}}
    (proc / "scenario_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["improvement_A_to_D"], indent=2))
    plot_bars(res, obs, figs)
    plot_cdf(resp, figs)
    plot_fleet(sens, figs)


if __name__ == "__main__":
    main()
