import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK_2, MUTED, SERIES, SEQ, SURFACE

NUM = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "severity",
       "open_city", "open_area", "waiting_city", "area_sqmi", "lat", "lon"]
CAT = ["zone_id", "sev_bucket", "time_band"]
LOAD = ["open_city", "open_area", "waiting_city"]
ROAD_NUM = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "area_sqmi", "lat", "lon"]
QUIET_Q = 0.3


class Encoder:
    def fit(self, d):
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[CAT])
        self.sc = StandardScaler().fit(d[NUM])
        return self

    def __call__(self, d):
        return np.hstack([self.sc.transform(d[NUM]), self.ohe.transform(d[CAT])])


def scores(y, yhat):
    e = np.abs(y - yhat)
    return {"MAE_min": e.mean() / 60, "MedAE_min": np.median(e) / 60, "MAPE_%": (e / y).mean() * 100}


def fit_models(trn, tst, enc, seed):
    Xtr, Xte = enc(trn), enc(tst)
    ytr = np.log(trn.travel_s.values)
    preds = {}

    base = trn.groupby(["zone_id", "hour"]).travel_s.median().rename("b")
    preds["Zone × hour median (baseline)"] = tst.join(base, on=["zone_id", "hour"]).b.values

    dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=300, random_state=seed).fit(Xtr, ytr)
    preds["Decision tree"] = np.exp(dt.predict(Xte))

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(trn), min(120000, len(trn)), replace=False)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=60,
                       early_stopping=True, random_state=seed).fit(Xtr[sub], ytr[sub])
    preds["Neural network (MLP)"] = np.exp(mlp.predict(Xte))

    gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                       max_leaf_nodes=63, random_state=seed).fit(Xtr, ytr)
    preds["Gradient boosting"] = np.exp(gb.predict(Xte))
    return preds, gb


def road_model(trn, seed):
    quiet = trn[(trn.open_area <= trn.open_area.quantile(QUIET_Q)) & (trn.sev_bucket == "critical")]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(quiet[["zone_id"]])

    def enc(d):
        return np.hstack([d[ROAD_NUM].values.astype(float), ohe.transform(d[["zone_id"]])])

    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                      max_leaf_nodes=31, min_samples_leaf=200, random_state=seed)
    m.fit(enc(quiet), np.log(quiet.travel_s.values))
    return m, enc, len(quiet)


def congestion_table(model, enc, zones):
    rows = []
    for dow in range(7):
        for h in range(24):
            g = zones.copy()
            g["hour"], g["dow"] = h, dow
            rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    g["hour_sin"] = np.sin(2 * np.pi * g.hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * g.hour / 24)
    g["dow_sin"] = np.sin(2 * np.pi * g.dow / 7)
    g["dow_cos"] = np.cos(2 * np.pi * g.dow / 7)
    g["is_weekend"] = (g.dow >= 5).astype(int)
    g["is_holiday"] = 0
    g["t_hat"] = np.exp(model.predict(enc(g)))
    t = g.groupby(["zone_id", "hour"]).t_hat.mean().unstack()
    factor = t.div(t.min(axis=1), axis=0)
    return t, factor


def plot_comparison(table, out):
    t = table.sort_values("MAE_min")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [SERIES[0] if i == 0 else MUTED for i in range(len(t))]
    bars = ax.barh(t.index, t.MAE_min, color=colors, edgecolor=SURFACE, linewidth=2, height=0.6)
    for b, v, p in zip(bars, t.MAE_min, t["MAPE_%"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f} min · MAPE {p:.0f}%", va="center", fontsize=9, color=INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, t.MAE_min.max() * 1.45)
    ax.set_xlabel("Mean absolute error (minutes) on Nov–Dec")
    ax.set_title("Predicting ambulance travel time", pad=24)
    style.subtitle(ax, "Lower is better · target is real EMS travel time per incident")
    fig.savefig(out / "travel_model_comparison.png")
    plt.close(fig)


def plot_congestion(factor, zones, out):
    order = zones.sort_values("lat", ascending=False).zone_id
    f = factor.loc[order]
    labels = zones.set_index("zone_id").loc[order].zip.astype(str)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(f.values, aspect="auto", cmap=SEQ, vmin=1)
    ax.set_yticks(range(len(f)), labels, fontsize=7.5)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("ZIP zone (north → south)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("Congestion factor (× fastest hour)", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    ax.set_title("Learned congestion factor by zone and hour", pad=24)
    style.subtitle(ax, f"Learned from critical calls when the local fleet was quiet · peak {f.values.max():.2f}×")
    fig.savefig(out / "travel_congestion_heatmap.png")
    plt.close(fig)


def main():
    style.apply()
    proc, figs = path("processed"), path("figures")
    df = pd.read_parquet(proc / "incidents.parquet").dropna(subset=["travel_s"])
    zones = pd.read_csv(proc / "zones.csv")
    trn, tst = df[df.split == "train"], df[df.split == "test"]
    enc = Encoder().fit(trn)

    preds, gb = fit_models(trn, tst, enc, CFG["seed"])
    table = pd.DataFrame({k: scores(tst.travel_s.values, v) for k, v in preds.items()}).T.round(3)
    table.to_csv(proc.parent.parent / "reports" / "travel_metrics.csv")
    print(table.to_string())

    rm, renc, n_quiet = road_model(trn, CFG["seed"])
    t, factor = congestion_table(rm, renc, zones)
    out = factor.stack().rename("factor").reset_index()
    out.to_csv(proc / "congestion.csv", index=False)
    t.stack().rename("t_hat_s").reset_index().to_csv(proc / "travel_hat.csv", index=False)
    (proc / "congestion_summary.json").write_text(json.dumps({
        "quiet_fleet_training_rows": n_quiet,
        "mean_factor": float(factor.values.mean()), "max_factor": float(factor.values.max()),
        "peak_hour_mean": float(factor.mean().max()), "peak_hour": int(factor.mean().idxmax()),
    }, indent=2))
    print(factor.mean().round(3).to_string())

    plot_comparison(table, figs)
    plot_congestion(factor, zones, figs)


if __name__ == "__main__":
    main()
