import itertools
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.simulate import (TARGET, TravelTables, allocate_units, load_calls, metrics, nearest_policy,
                                   simulate)
from src.evaluate.style import INK_2, MUTED, SERIES, SURFACE
from src.placement.kmeans_depots import node_weights
from src.routing.router import Router

N_ACT = 3
HOLE_PENALTY = 2.0


class QLearner:
    def __init__(self, n_depots, alpha, gamma, eps=1.0, eps_decay=0.995, eps_min=0.02, seed=0):
        self.Q = np.zeros((n_depots, 3, 5, 3, 3, 3, N_ACT))
        self.Q[..., 1:] = -1.0
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min
        self.rng = np.random.default_rng(seed)
        self.train = True
        self.prev = None
        self.ep_reward = 0.0

    def state(self, c, free, tables):
        top = tables.rank[c.hour, :N_ACT, c.zone_id]
        av = tuple(int(min(free[k], 2)) for k in top)
        return (int(top[0]), int(c.sev), int(c.band)) + av, top

    def act(self, c, free, tables):
        s, top = self.state(c, free, tables)
        valid = np.array([free[k] > 0 for k in top])
        if c.sev == 0 or not valid.any():
            self.cur = None
            return None
        q = self.Q[s]
        if self.train and self.rng.random() < self.eps:
            a = int(self.rng.choice(np.flatnonzero(valid)))
        else:
            a = int(np.flatnonzero(valid)[np.argmax(q[valid])])
        if self.train and self.prev is not None:
            ps, pa, pr = self.prev
            target = pr + self.gamma * q[valid].max()
            self.Q[ps + (pa,)] += self.alpha * (target - self.Q[ps + (pa,)])
        first = next(k for k, v in zip(top, valid) if v)
        chosen = int(top[a])
        extra = (tables.true[c.hour, chosen, c.zone_id] - tables.true[c.hour, first, c.zone_id]) / 60
        hole = HOLE_PENALTY if free[chosen] == 1 else 0.0
        self.cur = (s, a, -extra - hole)
        return chosen

    def observe(self, c, response_s):
        if getattr(self, "cur", None) is None:
            self.crit_bonus = getattr(self, "crit_bonus", 0.0) + (0 if c.sev != 0 else (1 if response_s <= TARGET else -1))
            return
        s, a, r = self.cur
        r += getattr(self, "crit_bonus", 0.0)
        self.crit_bonus = 0.0
        self.cur = (s, a)
        self.ep_reward += r
        self.prev = self.cur + (r,)
        self.cur = None

    def end_episode(self):
        if self.train and self.prev is not None:
            ps, pa, pr = self.prev
            self.Q[ps + (pa,)] += self.alpha * (pr - self.Q[ps + (pa,)])
        self.prev = None
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        r, self.ep_reward = self.ep_reward, 0.0
        return r


def train(days, tables, units, kappa, alpha, gamma, eps_decay, episodes, seed):
    rng = np.random.default_rng(seed)
    L = QLearner(units.size, alpha, gamma, eps_decay=eps_decay, seed=seed)
    keys = list(days)
    curve = []
    for ep in range(episodes):
        d = days[keys[rng.integers(len(keys))]]
        simulate(d, tables, units, kappa, learner=L)
        curve.append(L.end_episode() / len(d))
    return L, np.array(curve)


def evaluate(L, calls, tables, units, kappa):
    L.train = False
    L.prev = None
    r, tr, w = simulate(calls, tables, units, kappa, learner=L)
    return metrics(calls, r, tr, w)


def split_days(calls):
    return {d: g.assign(t=g.t - g.t.min()).reset_index(drop=True) for d, g in calls.groupby("date")}


def sweep(days, val, tables, units, kappa, seed):
    rl = CFG["rl"]
    base = rl["default"]
    configs = [dict(base)]
    for key, vals in [("alpha", rl["alpha"]), ("gamma", rl["gamma"]), ("epsilon_decay", rl["epsilon_decay"])]:
        for v in vals:
            c = dict(base)
            c[key] = v
            if c not in configs:
                configs.append(c)
    rows, curves = [], {}
    for c in configs:
        L, curve = train(days, tables, units, kappa, c["alpha"], c["gamma"], c["epsilon_decay"], rl["episodes"], seed)
        m = evaluate(L, val, tables, units, kappa)
        rows.append({**c, **m})
        curves[json.dumps(c)] = curve
        print(c, {k: round(v, 2) for k, v in m.items()})
    return pd.DataFrame(rows), curves


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_sweep(table, curves, nearest_val, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    base = CFG["rl"]["default"]
    ax = axes[0]
    for (key, curve), c in zip(curves.items(), itertools.cycle(SERIES)):
        cfg = json.loads(key)
        lab = ", ".join(f"{k[0] if k != 'epsilon_decay' else 'ε'}={v}" for k, v in cfg.items() if v != base[k]) or "default"
        if lab == "default":
            ax.plot(smooth(curve), color=SERIES[0], lw=2.2, label="default", zorder=3)
        else:
            ax.plot(smooth(curve), color=MUTED, lw=0.9, alpha=0.6)
    ax.set_xlabel("Episode (one simulated day)")
    ax.set_ylabel("Mean reward per call")
    ax.legend(loc="lower right")
    ax.set_title("Q-learning: training curves", pad=24)
    style.subtitle(ax, "Default config highlighted · grey = other settings")
    for ax, key, sym in zip(axes[1:], ["alpha", "gamma", "epsilon_decay"], ["α", "γ", "ε-decay"]):
        others = [k for k in ["alpha", "gamma", "epsilon_decay"] if k != key]
        t = table[(table[others[0]] == base[others[0]]) & (table[others[1]] == base[others[1]])].sort_values(key)
        ax.plot(t[key].astype(str), t.pct_critical_within_8, color=SERIES[0], marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(nearest_val, color=MUTED, ls="--", lw=1)
        ax.text(0.02, nearest_val, " nearest-unit rule", transform=ax.get_yaxis_transform(), fontsize=8.5,
                color=INK_2, va="bottom")
        ax.set_xlabel(sym)
        ax.set_ylabel("% critical ≤ 8 min (Oct)")
        ax.set_title(f"Effect of {sym}", fontsize=11, pad=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "rl_sweep.png")
    plt.close(fig)


def main():
    style.apply()
    proc, figs = path("processed"), path("figures")
    cal = json.loads((proc / "calibration.json").read_text())
    kappa, fleet = cal["kappa"], cal["fleet"]
    seed = CFG["seed"]
    R = Router(kappa=1.0)
    chosen = json.loads((proc / "selection.json").read_text())["placement"]
    depots = [d["node"] for d in json.loads((proc / "depots.json").read_text())["depots"][chosen]]
    units = allocate_units(R, depots, fleet, node_weights(R)["demand"])
    tables = TravelTables(R, depots, "ml")

    calls = load_calls()
    days = split_days(calls[calls.month < 10])
    val = calls[calls.month == 10].reset_index(drop=True)
    r, tr, w = simulate(val, tables, units, kappa, nearest_policy)
    nearest = metrics(val, r, tr, w)

    table, curves = sweep(days, val, tables, units, kappa, seed)
    table.round(3).to_csv(proc.parent.parent / "reports" / "rl_sweep.csv", index=False)
    best = table.loc[table.pct_critical_within_8.idxmax()]
    cfg = {k: float(best[k]) for k in ["alpha", "gamma", "epsilon_decay"]}
    L, curve = train(days, tables, units, kappa, cfg["alpha"], cfg["gamma"], cfg["epsilon_decay"],
                     CFG["rl"]["episodes"], seed)
    np.save(proc / "q_table.npy", L.Q)
    (proc / "rl_best.json").write_text(json.dumps({"config": cfg, "validation_oct": {k: float(best[k]) for k in
                                                    ["pct_critical_within_8", "pct_within_8", "mean_rt_min"]},
                                                   "nearest_oct": nearest, "units": units.tolist()}, indent=2))
    print("nearest (Oct)", {k: round(v, 2) for k, v in nearest.items()})
    print("best", cfg)
    plot_sweep(table, curves, nearest["pct_critical_within_8"], figs)


if __name__ == "__main__":
    main()
