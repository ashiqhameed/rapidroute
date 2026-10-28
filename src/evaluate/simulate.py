import heapq
import json

import numpy as np
import pandas as pd

from src.config import CFG, path
from src.routing.dijkstra import dijkstra, tree_cost
from src.routing.router import Router

TARGET = CFG["response_target_seconds"]
SEV = {"critical": 0, "urgent": 1, "non_urgent": 2}
BANDS = {"night": 0, "am_peak": 1, "midday": 2, "pm_peak": 3, "evening": 4}


def load_calls():
    proc = path("processed")
    df = pd.read_parquet(proc / "incidents.parquet")
    raw = pd.read_csv(path("ems_raw"), usecols=["incident_id", "first_on_scene_datetime", "incident_close_datetime"])
    svc = (pd.to_datetime(raw.incident_close_datetime) - pd.to_datetime(raw.first_on_scene_datetime)).dt.total_seconds()
    df["service_s"] = df.incident_id.map(pd.Series(svc.values, index=raw.incident_id))
    df["service_s"] = df.service_s.where(df.service_s.between(300, 4 * 3600))
    df["service_s"] = df.service_s.fillna(df.groupby("sev_bucket").service_s.transform("median"))
    quiet = df[df.waiting_city == 0]
    df["proc_s"] = df.sev_bucket.map(quiet.groupby("sev_bucket").dispatch_s.median())
    df["sev"] = df.sev_bucket.map(SEV)
    df["band"] = df.time_band.map(BANDS)
    df["t"] = (df.ts - df.ts.min().normalize()).dt.total_seconds()
    return df


class TravelTables:
    def __init__(self, R, depots, mode):
        self.mode = mode
        self.est = np.zeros((24, len(depots), len(R.zones)))
        self.true = np.zeros_like(self.est)
        for h in range(24):
            w = R.weights(mode, h)
            truth = R.true_seconds(h)
            for k, d in enumerate(depots):
                dist, parent, _, _, pedge = dijkstra(R.g, int(d), w)
                self.est[h, k] = dist[R.zone_node]
                self.true[h, k] = tree_cost(dist, parent, pedge, truth)[R.zone_node]
        self.rank = np.argsort(self.est, axis=1)


def allocate_units(R, depots, fleet, weights):
    secs, origin = R.tree(list(depots), 12, "ml")
    share = np.bincount(origin, weights=weights, minlength=len(depots))
    share = share / share.sum()
    units = np.maximum(1, np.floor(share * fleet)).astype(int)
    while units.sum() < fleet:
        units[np.argmax(share * fleet - units)] += 1
    while units.sum() > fleet:
        units[np.argmax(units)] -= 1
    return units


def nearest_policy(call, free, tables, kappa):
    for k in tables.rank[call.hour, :, call.zone_id]:
        if free[k] > 0:
            return int(k)
    return None


def simulate(calls, tables, units, kappa=1.0, policy=nearest_policy, learner=None):
    free = units.astype(int).copy()
    busy = []
    queue = []
    resp = np.full(len(calls), np.nan)
    travel = np.full(len(calls), np.nan)
    waited = np.zeros(len(calls))
    cols = calls[["t", "zone_id", "hour", "sev", "band", "service_s", "proc_s"]].itertuples(index=False)
    rows = list(cols)

    def dispatch(i, now):
        c = rows[i]
        k = policy(c, free, tables, kappa) if learner is None else learner.act(c, free, tables)
        if k is None or free[k] <= 0:
            k = nearest_policy(c, free, tables, kappa)
        tt = tables.true[c.hour, k, c.zone_id] * kappa
        r = (now - c.t) + c.proc_s + tt
        resp[i], travel[i], waited[i] = r, tt, now - c.t
        free[k] -= 1
        heapq.heappush(busy, (now + c.proc_s + 2 * tt + c.service_s, k))
        if learner is not None:
            learner.observe(c, r)

    def release(until):
        while busy and busy[0][0] <= until:
            f, k = heapq.heappop(busy)
            free[k] += 1
            while queue and free.sum() > 0:
                _, _, j = heapq.heappop(queue)
                dispatch(j, f)

    for i, c in enumerate(rows):
        release(c.t)
        if free.sum() > 0:
            dispatch(i, c.t)
        else:
            heapq.heappush(queue, (c.sev, c.t, i))
    release(np.inf)
    return resp, travel, waited


def metrics(calls, resp, travel, waited):
    crit = calls.sev.values == 0
    return {
        "mean_rt_min": float(np.nanmean(resp) / 60),
        "median_rt_min": float(np.nanmedian(resp) / 60),
        "p90_rt_min": float(np.nanpercentile(resp, 90) / 60),
        "pct_within_8": float(np.nanmean(resp <= TARGET) * 100),
        "pct_critical_within_8": float(np.nanmean(resp[crit] <= TARGET) * 100),
        "median_travel_min": float(np.nanmedian(travel) / 60),
        "pct_waited": float(np.mean(waited > 0) * 100),
    }


def calibrate(R, calls, depots, fleet_sizes, observed):
    tables = TravelTables(R, depots, "distance")
    out = []
    for fleet in fleet_sizes:
        units = allocate_units(R, depots, fleet, np.ones(R.g.n))
        lo, hi = 0.5, 6.0
        for _ in range(18):
            mid = (lo + hi) / 2
            _, tr, _ = simulate(calls, tables, units, mid)
            if np.nanmedian(tr) < observed["median_travel_s"]:
                lo = mid
            else:
                hi = mid
        kappa = (lo + hi) / 2
        r, tr, w = simulate(calls, tables, units, kappa)
        m = metrics(calls, r, tr, w)
        m.update({"fleet": fleet, "kappa": kappa})
        out.append(m)
    return pd.DataFrame(out)


def observed_stats(calls):
    return {
        "median_travel_s": float(calls.travel_s.median()),
        "median_rt_min": float(calls.response_s.median() / 60),
        "pct_within_8": float((calls.response_s <= TARGET).mean() * 100),
        "pct_critical_within_8": float((calls[calls.sev == 0].response_s <= TARGET).mean() * 100),
        "p90_rt_min": float(calls.response_s.quantile(0.9) / 60),
    }


def fit_score(row, obs):
    return (abs(row.p90_rt_min - obs["p90_rt_min"]) / obs["p90_rt_min"]
            + abs(row.median_rt_min - obs["median_rt_min"]) / obs["median_rt_min"]
            + abs(row.pct_within_8 - obs["pct_within_8"]) / obs["pct_within_8"])


def plot_calibration(res, obs, best, out):
    import matplotlib.pyplot as plt
    from src.evaluate import style
    from src.evaluate.style import INK_2, MUTED, SERIES, SURFACE

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, lab in [(axes[0], "median_rt_min", "Median response (min)"),
                         (axes[1], "p90_rt_min", "P90 response (min)"),
                         (axes[2], "pct_within_8", "% within 8 min")]:
        ax.plot(res.fleet, res[col], color=SERIES[0], marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label="Simulated")
        ax.axhline(obs[col], color=MUTED, ls="--", lw=1.2, label="Observed (Oct 2024)")
        ax.axvline(best, color=SERIES[1], lw=1, alpha=0.6)
        ax.set_xlabel("Fleet size (ambulances)")
        ax.set_ylabel(lab)
        if col == "p90_rt_min":
            ax.set_yscale("log")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Calibrating the simulator: fleet = {best}", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.06)
    fig.text(0.01, 0.97, "Speed factor κ re-fitted for each fleet so median travel matches reality", fontsize=9.5,
             color=INK_2, ha="left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "sim_calibration.png")
    plt.close(fig)


def main():
    from src.evaluate import style

    style.apply()
    proc = path("processed")
    R = Router(kappa=1.0)
    calls = load_calls()
    cal = calls[calls.month == 10].reset_index(drop=True)
    obs = observed_stats(cal)
    dep = json.loads((proc / "depots.json").read_text())["depots"]["Uniform grid"]
    res = calibrate(R, cal, [d["node"] for d in dep], CFG["fleet_candidates"], obs)
    res["fit_score"] = res.apply(lambda r: fit_score(r, obs), axis=1)
    best = res.loc[res.fit_score.idxmin()]
    out = {"kappa": float(best.kappa), "fleet": int(best.fleet), "observed_oct": obs,
           "simulated_oct": {k: float(best[k]) for k in ["median_rt_min", "p90_rt_min", "pct_within_8",
                                                          "pct_critical_within_8", "median_travel_min", "pct_waited"]}}
    (proc / "calibration.json").write_text(json.dumps(out, indent=2))
    res.round(3).to_csv(proc.parent.parent / "reports" / "sim_calibration.csv", index=False)
    print(res.round(2).to_string(index=False))
    print(json.dumps(out, indent=2))
    plot_calibration(res, obs, int(best.fleet), path("figures"))
    select_placement(R, calls, out["kappa"], out["fleet"])


def select_placement(R, calls, kappa, fleet):
    from src.placement.kmeans_depots import node_weights

    proc = path("processed")
    depots = json.loads((proc / "depots.json").read_text())["depots"]
    weights = node_weights(R)["demand"]
    val = calls[calls.month == 10].reset_index(drop=True)
    rows = {}
    for key in ["K-means (demand)", "K-means (risk-weighted)"]:
        nodes = [d["node"] for d in depots[key]]
        r, tr, w = simulate(val, TravelTables(R, nodes, "ml"), allocate_units(R, nodes, fleet, weights), kappa)
        rows[key] = metrics(val, r, tr, w)
    best = max(rows, key=lambda k: rows[k]["pct_critical_within_8"])
    (proc / "selection.json").write_text(json.dumps({"placement": best, "validation_oct": rows}, indent=2))
    print("selected placement:", best)


if __name__ == "__main__":
    main()
