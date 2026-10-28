import json

import holidays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK_2, MUTED, SERIES, SURFACE
from src.models.gd_linear import GDLinearRegression

FEATURES = [
    "zone_rate", "zone_how_rate", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "month", "lag_168", "lag_24", "roll_24", "roll_168",
]


def build_panel(df, zones):
    y = CFG["year"]
    hours = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq="h")
    idx = pd.MultiIndex.from_product([zones.zone_id, hours], names=["zone_id", "ts"])
    counts = df.assign(ts=df.ts.dt.floor("h")).groupby(["zone_id", "ts"]).size()
    p = counts.reindex(idx, fill_value=0).rename("calls").reset_index()

    p["hour"] = p.ts.dt.hour
    p["dow"] = p.ts.dt.dayofweek
    p["month"] = p.ts.dt.month
    p["how"] = p.dow * 24 + p.hour
    p["is_weekend"] = (p.dow >= 5).astype(int)
    hol = holidays.country_holidays("US", subdiv="NY", years=y)
    p["is_holiday"] = p.ts.dt.date.map(lambda d: int(d in hol))
    p["hour_sin"] = np.sin(2 * np.pi * p.hour / 24)
    p["hour_cos"] = np.cos(2 * np.pi * p.hour / 24)
    p["dow_sin"] = np.sin(2 * np.pi * p.dow / 7)
    p["dow_cos"] = np.cos(2 * np.pi * p.dow / 7)
    p["split"] = np.where(p.month.isin(CFG["test_months"]), "test", "train")

    g = p.groupby("zone_id").calls
    p["lag_24"] = g.shift(24)
    p["lag_168"] = g.shift(168)
    p["roll_24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    p["roll_168"] = g.transform(lambda s: s.shift(1).rolling(168).mean())

    tr = p[p.split == "train"]
    p["zone_rate"] = p.zone_id.map(tr.groupby("zone_id").calls.mean())
    zh = tr.groupby(["zone_id", "how"]).calls.mean().rename("zone_how_rate")
    p = p.join(zh, on=["zone_id", "how"])
    return p.dropna(subset=FEATURES).reset_index(drop=True)


def metrics(y, yhat):
    return {
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "R2": r2_score(y, yhat),
    }


def fit_models(train, test):
    Xtr, ytr = train[FEATURES].values, train.calls.values
    Xte = test[FEATURES].values
    preds = {"Historical mean (baseline)": test.zone_how_rate.values}

    gd = GDLinearRegression(lr=0.1, n_iter=800, scale=True).fit(Xtr, ytr)
    preds["Linear regression (GD, scratch)"] = gd.predict(Xte)

    dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=200, random_state=CFG["seed"]).fit(Xtr, ytr)
    preds["Decision tree"] = dt.predict(Xte)

    gb = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=100, random_state=CFG["seed"],
    ).fit(Xtr, ytr)
    preds["Gradient boosting (Poisson)"] = gb.predict(Xte)
    return preds, gd


def gd_scaling_experiment(train):
    X, y = train[FEATURES].values, train.calls.values
    runs = {
        "Scaled, lr = 0.1": GDLinearRegression(lr=0.1, n_iter=300, scale=True).fit(X, y),
        "Unscaled, lr = 0.1": GDLinearRegression(lr=0.1, n_iter=300, scale=False).fit(X, y),
        "Unscaled, lr = 1e-4": GDLinearRegression(lr=1e-4, n_iter=300, scale=False).fit(X, y),
    }
    return runs


def plot_gd(runs, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    for (name, m), c in zip(runs.items(), [SERIES[0], SERIES[7], SERIES[3]]):
        loss = np.array(m.loss_)
        ax.plot(np.arange(len(loss)), loss, color=c, label=name + (" (diverged)" if m.diverged_ else ""))
        ax.plot(len(loss) - 1, loss[-1], "o", color=c, ms=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE / 2 (log scale)")
    ax.legend(loc="upper right")
    ax.set_title("Feature scaling makes gradient descent converge", pad=24)
    style.subtitle(ax, "Linear demand model trained from scratch · same data, three settings")
    fig.savefig(out / "demand_gd_scaling.png")
    plt.close(fig)


def plot_comparison(table, ceiling, out):
    t = table.sort_values("RMSE", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    t = table.sort_values("R2", ascending=False)
    for ax, col, lab in [(axes[0], "R2", "R² on zone-hour calls"), (axes[1], "R2_citywide", "R² on citywide hourly calls")]:
        colors = [SERIES[0] if "boosting" in n else MUTED for n in t.index]
        bars = ax.barh(t.index, t[col], color=colors, edgecolor=SURFACE, linewidth=2, height=0.6)
        for b, v in zip(bars, t[col]):
            ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.3f}", va="center", fontsize=9, color=INK_2)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.set_xlim(0, max(t[col].max(), ceiling if col == "R2" else 0) * 1.2)
    axes[0].axvline(ceiling, color=INK_2, ls="--", lw=1)
    axes[0].text(ceiling, 0.5, f" noise\n ceiling\n {ceiling:.3f}", fontsize=8.5, color=INK_2, va="center",
                 transform=axes[0].get_xaxis_transform())
    axes[1].set_yticklabels([])
    axes[0].set_title("Demand forecasting on Nov–Dec holdout", pad=24)
    style.subtitle(axes[0], "Higher is better · zone-hour counts are ~1 call, so random noise caps R²")
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "demand_model_comparison.png")
    plt.close(fig)


def plot_week(test, best, out):
    w = test.assign(pred=best).groupby("ts")[["calls", "pred"]].sum()
    start = w.index[w.index >= f"{CFG['year']}-11-11"][0]
    w = w.loc[start: start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(w.index, w.calls, color=MUTED, lw=1.2, label="Actual")
    ax.plot(w.index, w.pred, color=SERIES[0], lw=2, label="Predicted (gradient boosting)")
    ax.set_ylabel("Calls per hour (Manhattan)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Forecast vs actual, one test week", pad=24)
    style.subtitle(ax, "Sum of 42 zone-level forecasts · 11–18 Nov 2024")
    fig.savefig(out / "demand_test_week.png")
    plt.close(fig)


def main():
    style.apply()
    proc = path("processed")
    figs = path("figures")
    df = pd.read_parquet(proc / "incidents.parquet")
    zones = pd.read_csv(proc / "zones.csv")

    panel = build_panel(df, zones)
    train, test = panel[panel.split == "train"], panel[panel.split == "test"]
    preds, gd = fit_models(train, test)

    rows = {}
    city_y = test.groupby("ts").calls.sum()
    for name, yhat in preds.items():
        m = metrics(test.calls, yhat)
        city = pd.Series(yhat, index=test.index).groupby(test.ts).sum()
        m["R2_citywide"] = r2_score(city_y, city.loc[city_y.index])
        rows[name] = m
    table = pd.DataFrame(rows).T.round(4)
    ceiling = float(1 - test.calls.mean() / test.calls.var())
    table["R2_share_of_ceiling"] = (table.R2 / ceiling).round(3)
    table.to_csv(proc.parent.parent / "reports" / "demand_metrics.csv")
    print(table.to_string())

    best = preds["Gradient boosting (Poisson)"]
    lam = test.assign(lam=best).groupby(["zone_id", "how"]).lam.mean().reset_index()
    lam.to_csv(proc / "demand_lambda.csv", index=False)
    zone_pred = lam.groupby("zone_id").lam.sum().rename("pred_weekly_calls").reset_index()
    zone_pred.to_csv(proc / "zone_demand.csv", index=False)
    panel[["zone_id", "ts", "calls"]].assign(pred=np.nan).to_parquet(proc / "demand_panel.parquet", index=False)

    runs = gd_scaling_experiment(train)
    plot_gd(runs, figs)
    plot_comparison(table, ceiling, figs)
    plot_week(test, best, figs)
    (proc / "demand_ceiling.json").write_text(json.dumps({"poisson_r2_ceiling": ceiling}))
    summary = {k: {"final_loss": float(v.loss_[-1]), "iters": len(v.loss_), "diverged": v.diverged_} for k, v in runs.items()}
    (proc / "gd_scaling.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
