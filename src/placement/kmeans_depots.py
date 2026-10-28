import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK, INK_2, MUTED, SERIES, SEQ, SURFACE
from src.routing.router import Router

EVAL_HOURS = (4, 12, 18)
TARGET_S = CFG["response_target_seconds"]


def xy(lat, lon):
    lat0 = np.radians(40.78)
    return np.column_stack([lon * 111320 * np.cos(lat0), lat * 110540])


def node_weights(R):
    proc = path("processed")
    dem = pd.read_csv(proc / "zone_demand.csv").set_index("zone_id").pred_weekly_calls
    risk = pd.read_csv(proc / "zone_risk.csv").set_index("zone_id").risk
    zid = R.zones.zone_id.values[R.node_zone]
    per_zone = pd.Series(zid).value_counts()
    demand = dem.loc[zid].values / per_zone.loc[zid].values
    return {
        "plain": np.ones(R.g.n),
        "demand": demand,
        "risk": demand * (1 + risk.loc[zid].values),
    }


def kmeans_nodes(R, X, w, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X, sample_weight=w)
    nodes = []
    for c in km.cluster_centers_:
        order = np.argsort(((X - c) ** 2).sum(1))
        nodes.append(int(next(n for n in order if n not in nodes)))
    return nodes, km


def uniform_nodes(R, X, k):
    y = X[:, 1]
    edges = np.linspace(y.min(), y.max(), k + 1)
    nodes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.flatnonzero((y >= lo) & (y <= hi))
        c = np.array([np.median(X[m, 0]), (lo + hi) / 2])
        nodes.append(int(m[np.argmin(((X[m] - c) ** 2).sum(1))]))
    return nodes


def evaluate(R, nodes, demand_w):
    zone_demand = np.bincount(R.node_zone, weights=demand_w, minlength=len(R.zones))
    times, cover = [], []
    for h in EVAL_HOURS:
        secs, _ = R.tree(nodes, h, "ml")
        t = secs[R.zone_node]
        times.append(np.average(t, weights=zone_demand))
        cover.append(np.average(t <= TARGET_S, weights=zone_demand))
    return float(np.mean(times)), float(np.mean(cover))


def k_selection(R, X, w, seed):
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), 3000, replace=True, p=w / w.sum())
    rows = []
    for k in range(CFG["k_range"][0], CFG["k_range"][1] + 1):
        nodes, km = kmeans_nodes(R, X, w, k, seed)
        lab = km.predict(X[sample])
        t, c = evaluate(R, nodes, w)
        rows.append({"k": k, "wcss": km.inertia_ / 1e6, "silhouette": silhouette_score(X[sample], lab),
                     "davies_bouldin": davies_bouldin_score(X[sample], lab),
                     "calinski_harabasz": calinski_harabasz_score(X[sample], lab), "mean_travel_s": t, "coverage": c})
    return pd.DataFrame(rows)


def elbow(k, y):
    k, y = np.asarray(k, float), np.asarray(y, float)
    kn = (k - k[0]) / (k[-1] - k[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    d = np.abs(kn + yn - 1) / np.sqrt(2)
    return int(k[np.argmax(d)])


def choose_k(t):
    votes = {
        "Elbow (WCSS)": elbow(t.k, t.wcss),
        "Silhouette": int(t.k[t.silhouette.idxmax()]),
        "Davies–Bouldin": int(t.k[t.davies_bouldin.idxmin()]),
        "Calinski–Harabasz": int(t.k[t.calinski_harabasz.idxmax()]),
    }
    return int(np.median(list(votes.values()))), votes


def plot_k(t, votes, k_vote, k_star, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    spec = [("wcss", "WCSS (×10⁶ m²)", "Elbow (WCSS)"), ("silhouette", "Silhouette (higher better)", "Silhouette"),
            ("davies_bouldin", "Davies–Bouldin (lower better)", "Davies–Bouldin"),
            ("calinski_harabasz", "Calinski–Harabasz (higher better)", "Calinski–Harabasz")]
    for ax, (col, lab, name) in zip(axes, spec):
        ax.plot(t.k, t[col], color=SERIES[0], marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        kv = votes[name]
        ax.plot(kv, t.set_index("k").loc[kv, col], "o", ms=11, color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(f"{name}: K = {kv}", fontsize=11, pad=8)
        ax.set_xlabel("K")
        ax.set_ylabel(lab)
        ax.set_xticks(t.k[::2])
    fig.suptitle(f"Choosing K: four methods vote {sorted(votes.values())} · median {k_vote} · chosen K = {k_star} (elbow, ≈ FDNY EMS stations)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=1.06)
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "placement_k_selection.png")
    plt.close(fig)


def plot_k_ops(t, k_star, out):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t.k, t.mean_travel_s / 60, color=SERIES[0], marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    r = t.set_index("k").loc[k_star]
    ax.plot(k_star, r.mean_travel_s / 60, "o", ms=11, color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"K = {k_star}: {r.mean_travel_s / 60:.1f} min", (k_star, r.mean_travel_s / 60), xytext=(10, 10),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_xlabel("Number of depots (K)")
    ax.set_ylabel("Demand-weighted travel (min)")
    ax.set_xticks(t.k)
    ax.set_title("More depots, shorter trips, with diminishing returns", pad=24)
    style.subtitle(ax, "Multi-source Dijkstra from risk-weighted K-means depots · ML travel times")
    fig.savefig(out / "placement_k_travel.png")
    plt.close(fig)


def plot_maps(R, X, placements, w, out):
    names = list(placements)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 8.2), sharey=True)
    vmax = 12
    for ax, name in zip(axes, names):
        secs, _ = R.tree(placements[name], 18, "ml")
        sc = ax.scatter(R.g.lon, R.g.lat, c=np.clip(secs / 60, 0, vmax), cmap=SEQ, s=4, linewidths=0, vmin=0, vmax=vmax)
        d = placements[name]
        ax.scatter(R.g.lon[d], R.g.lat[d], s=90, marker="P", color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        t, c = evaluate(R, d, w)
        ax.set_title(name, fontsize=11, pad=22)
        ax.text(0, 1.01, f"{t / 60:.1f} min avg · {c * 100:.0f}% in 8 min", transform=ax.transAxes, fontsize=9, color=INK_2)
        ax.set_aspect(1 / np.cos(np.radians(40.78)))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(top=0.86, wspace=0.05, right=0.9)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.92, 0.12, 0.012, 0.66]))
    cb.set_label("Travel time from nearest depot at 18:00 (min)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle(f"Depot placement strategies · K = {len(placements[names[0]])}", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=0.97)
    fig.savefig(out / "placement_maps.png")
    plt.close(fig)


def main():
    style.apply()
    proc, figs = path("processed"), path("figures")
    seed = CFG["seed"]
    R = Router()
    X = xy(R.g.lat, R.g.lon)
    W = node_weights(R)

    t = k_selection(R, X, W["risk"], seed)
    k_vote, votes = choose_k(t)
    k_star = int(CFG.get("k_depots") or k_vote)
    t.round(4).to_csv(proc.parent.parent / "reports" / "placement_k_selection.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("votes", votes, "median vote", k_vote, "used K", k_star)

    placements = {
        "Uniform grid": uniform_nodes(R, X, k_star),
        "K-means (plain)": kmeans_nodes(R, X, W["plain"], k_star, seed)[0],
        "K-means (demand)": kmeans_nodes(R, X, W["demand"], k_star, seed)[0],
        "K-means (risk-weighted)": kmeans_nodes(R, X, W["risk"], k_star, seed)[0],
    }
    rng = np.random.default_rng(seed)
    rand = [evaluate(R, [int(x) for x in rng.choice(R.g.n, k_star, replace=False)], W["demand"]) for _ in range(20)]

    rows = [{"strategy": "Random (mean of 20)", "mean_travel_min": np.mean([r[0] for r in rand]) / 60,
             "coverage_8min_pct": np.mean([r[1] for r in rand]) * 100}]
    for name, nodes in placements.items():
        tt, cc = evaluate(R, nodes, W["demand"])
        rows.append({"strategy": name, "mean_travel_min": tt / 60, "coverage_8min_pct": cc * 100})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(proc.parent.parent / "reports" / "placement_results.csv", index=False)
    print(res.to_string(index=False))

    depots = {name: [{"node": int(n), "lat": float(R.g.lat[n]), "lon": float(R.g.lon[n])} for n in nodes]
              for name, nodes in placements.items()}
    (proc / "depots.json").write_text(json.dumps({"k": k_star, "k_median_vote": k_vote, "votes": votes, "depots": depots}, indent=2))

    plot_k(t, votes, k_vote, k_star, figs)
    plot_k_ops(t, k_star, figs)
    plot_maps(R, X, placements, W["demand"], figs)


if __name__ == "__main__":
    main()
