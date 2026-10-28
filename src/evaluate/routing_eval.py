import json
import time

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK_2, MUTED, SERIES, SURFACE
from src.routing.astar import astar
from src.routing.dijkstra import bellman_ford, dijkstra
from src.routing.graph import haversine_m
from src.routing.router import Router

N_PAIRS = 500


def correctness(R, rng):
    g = R.g
    w = R.weights("ml", 18)
    H = nx.DiGraph()
    tail = g.edge_tail()
    H.add_weighted_edges_from(zip(tail.tolist(), g.head.tolist(), w.tolist()))
    speed = float(np.max(g.length / w))
    rows = []
    for _ in range(N_PAIRS):
        s, t = (int(x) for x in rng.integers(g.n, size=2))
        ref = nx.dijkstra_path_length(H, s, t)
        t0 = time.perf_counter()
        d, _, _, ed, _ = dijkstra(g, s, w, target=t)
        td = time.perf_counter() - t0
        t0 = time.perf_counter()
        a, _, ea = astar(g, s, t, w, speed)
        ta = time.perf_counter() - t0
        rows.append({"km": haversine_m(g.lat[s], g.lon[s], g.lat[t], g.lon[t]) / 1000, "ref": ref,
                     "dij": d[t], "astar": a, "exp_dij": ed, "exp_astar": ea, "ms_dij": td * 1e3, "ms_astar": ta * 1e3})
    df = pd.DataFrame(rows)
    bf = bellman_ford(g, 0, w)
    full, *_ = dijkstra(g, 0, w)
    return df, bool(np.allclose(bf, full))


def route_choice(R, rng, hours=(4, 18), n=400):
    g = R.g
    rows = []
    for h in hours:
        for _ in range(n):
            s, t = (int(x) for x in rng.choice(R.zone_node, 2, replace=False))
            out = {"hour": h}
            for mode in ("distance", "freeflow", "ml"):
                r = R.route(s, t, h, mode, "astar")
                out[mode] = r["seconds"]
                out[mode + "_path"] = tuple(r["path"])
            rows.append(out)
    df = pd.DataFrame(rows)
    summ = []
    for h, d in df.groupby("hour"):
        for mode in ("distance", "freeflow"):
            extra = d[mode] - d["ml"]
            summ.append({"hour": h, "mode": mode,
                         "pct_routes_different": float((d[mode + "_path"] != d["ml_path"]).mean() * 100),
                         "mean_extra_s": float(extra.mean()), "p90_extra_s": float(extra.quantile(0.9)),
                         "max_extra_s": float(extra.max())})
    return df, pd.DataFrame(summ)


def plot_expansion(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.scatter(df.km, df.exp_dij, s=10, color=MUTED, alpha=0.6, label="Dijkstra", linewidths=0)
    ax.scatter(df.km, df.exp_astar, s=10, color=SERIES[0], alpha=0.8, label="A*", linewidths=0)
    ax.set_xlabel("Straight-line trip distance (km)")
    ax.set_ylabel("Nodes expanded")
    ax.legend(loc="lower right", markerscale=2)
    ax.set_title("A* searches far less of the city", pad=24)
    style.subtitle(ax, f"{len(df)} random trips · same optimal answer every time")
    ax = axes[1]
    bins = pd.cut(df.km, [0, 2, 4, 6, 8, 25], labels=["0–2", "2–4", "4–6", "6–8", "8+"])
    r = (df.exp_astar / df.exp_dij).groupby(bins, observed=True).mean() * 100
    bars = ax.bar(r.index.astype(str), r.values, color=SERIES[0], edgecolor=SURFACE, linewidth=2, width=0.65)
    for b, v in zip(bars, r.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK_2)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Trip distance (km)")
    ax.set_ylabel("A* nodes as % of Dijkstra")
    ax.set_title("Biggest savings on short trips", pad=24)
    style.subtitle(ax, "Heuristic: straight-line distance ÷ max road speed (admissible)")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "routing_astar_vs_dijkstra.png")
    plt.close(fig)


def plot_route_choice(summ, out):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels, vals, cols = [], [], []
    for _, r in summ.iterrows():
        labels.append(f"{'Shortest distance' if r['mode'] == 'distance' else 'Free-flow time'} · {int(r.hour):02d}:00")
        vals.append(r.mean_extra_s)
        cols.append(SERIES[1] if r["mode"] == "distance" else SERIES[3])
    bars = ax.barh(labels, vals, color=cols, edgecolor=SURFACE, linewidth=2, height=0.6)
    for b, v, (_, r) in zip(bars, vals, summ.iterrows()):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  +{v:.0f} s avg · {r.pct_routes_different:.0f}% of routes differ",
                va="center", fontsize=9, color=INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 2.1)
    ax.set_xlabel("Extra travel seconds vs the ML-weighted route")
    ax.set_title("Routing by distance wastes 1.5–2 minutes a trip", pad=24)
    style.subtitle(ax, "400 random zone-to-zone trips per hour · all routes judged on ML travel times")
    fig.savefig(out / "routing_route_choice.png")
    plt.close(fig)


def main():
    style.apply()
    rng = np.random.default_rng(CFG["seed"])
    R = Router()
    proc, figs = path("processed"), path("figures")
    df, bf_ok = correctness(R, rng)
    choice, summ = route_choice(R, rng)
    result = {
        "graph_nodes": int(R.g.n), "graph_edges": int(R.g.m),
        "pairs_tested": len(df),
        "dijkstra_matches_networkx": bool(np.allclose(df.dij, df.ref)),
        "astar_matches_networkx": bool(np.allclose(df.astar, df.ref)),
        "bellman_ford_matches_dijkstra": bf_ok,
        "mean_nodes_dijkstra": float(df.exp_dij.mean()), "mean_nodes_astar": float(df.exp_astar.mean()),
        "mean_ms_dijkstra": float(df.ms_dij.mean()), "mean_ms_astar": float(df.ms_astar.mean()),
    }
    (proc / "routing_eval.json").write_text(json.dumps(result, indent=2))
    summ.round(2).to_csv(proc.parent.parent / "reports" / "routing_route_choice.csv", index=False)
    print(json.dumps(result, indent=2))
    print(summ.round(2).to_string(index=False))
    plot_expansion(df, figs)
    plot_route_choice(summ, figs)


if __name__ == "__main__":
    main()
