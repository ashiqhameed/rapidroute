import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK, INK_2, MUTED, SERIES, SEQ, SURFACE

NUM = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday", "severity",
       "load_60", "zone_load_60", "zone_how_rate", "zone_late_hist",
       "open_city", "open_area", "waiting_city"]
CAT = ["zone_id", "sev_bucket", "time_band", "call_type"]
SVM_N = 15000
NB_BIN = ["load_60", "zone_load_60", "zone_how_rate", "open_city", "open_area", "waiting_city"]


def rolling_count(ts, window="60min"):
    t = ts.values.astype("datetime64[ns]")
    lo = np.searchsorted(t, t - np.timedelta64(pd.Timedelta(window)), side="left")
    return np.arange(len(t)) - lo


def build_features(df):
    df = df.sort_values("ts").reset_index(drop=True)
    df["load_60"] = rolling_count(df.ts)
    df["zone_load_60"] = 0
    for z, idx in df.groupby("zone_id").groups.items():
        df.loc[idx, "zone_load_60"] = rolling_count(df.loc[idx, "ts"])

    tr = df[df.split == "train"]
    weeks = tr.date.nunique() / 7
    zh = (tr.groupby(["zone_id", "how"]).size() / weeks).rename("zone_how_rate")
    df = df.join(zh, on=["zone_id", "how"])
    df["zone_how_rate"] = df.zone_how_rate.fillna(0)
    df["zone_late_hist"] = df.zone_id.map(tr.groupby("zone_id").late.mean().astype(float))

    top = tr.initial_call_type.value_counts().index[:30]
    df["call_type"] = df.initial_call_type.where(df.initial_call_type.isin(top), "OTHER")
    return df.dropna(subset=["late"]).reset_index(drop=True)


def evaluate(name, y, score, thr):
    pred = (score >= thr).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC_AUC": roc_auc_score(y, score),
    }


def best_threshold(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[int(np.argmax(tpr - fpr))])


def main():
    style.apply()
    proc, figs = path("processed"), path("figures")
    rng = np.random.default_rng(CFG["seed"])
    df = build_features(pd.read_parquet(proc / "incidents.parquet"))
    df["late"] = df.late.astype(int)

    fit = df[(df.split == "train") & (df.month < 10)]
    val = df[df.month == 10]
    trn = df[df.split == "train"]
    test = df[df.split == "test"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(trn[CAT])
    sc = StandardScaler().fit(trn[NUM])

    def dense(d):
        return np.hstack([sc.transform(d[NUM]), ohe.transform(d[CAT])])

    kb = KBinsDiscretizer(n_bins=6, encode="ordinal", strategy="quantile").fit(trn[NB_BIN])
    cat_codes = {c: {v: i for i, v in enumerate(sorted(trn[c].astype(str).unique()))} for c in CAT}

    def nb_matrix(d):
        bins = kb.transform(d[NB_BIN])
        cats = np.column_stack([d[c].astype(str).map(cat_codes[c]).fillna(0).values for c in CAT])
        return np.column_stack([cats, d.hour.values, d.dow.values, d.severity.values, bins]).astype(int)

    results, scores, models = [], {}, {}

    nb = CategoricalNB(alpha=1.0, min_categories=np.max(nb_matrix(df), axis=0) + 1).fit(nb_matrix(fit), fit.late)
    thr = best_threshold(val.late, nb.predict_proba(nb_matrix(val))[:, 1])
    nb.fit(nb_matrix(trn), trn.late)
    scores["Naïve Bayes"] = nb.predict_proba(nb_matrix(test))[:, 1]
    results.append(evaluate("Naïve Bayes", test.late, scores["Naïve Bayes"], thr))

    best_depth, best_auc = None, -1
    for depth in [4, 6, 8, 10, 12, 14]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, random_state=CFG["seed"]).fit(dense(fit), fit.late)
        auc = roc_auc_score(val.late, dt.predict_proba(dense(val))[:, 1])
        if auc > best_auc:
            best_depth, best_auc = depth, auc
    dt = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=100, random_state=CFG["seed"]).fit(dense(fit), fit.late)
    thr = best_threshold(val.late, dt.predict_proba(dense(val))[:, 1])
    dt.fit(dense(trn), trn.late)
    scores["Decision tree"] = dt.predict_proba(dense(test))[:, 1]
    results.append(evaluate(f"Decision tree", test.late, scores["Decision tree"], thr))
    models["Decision tree"] = dt

    s_fit = fit.iloc[rng.choice(len(fit), 5000, replace=False)]
    v = val.sample(10000, random_state=1)
    Xv = dense(v)
    best_cfg, best_auc = None, -1
    for C in [0.3, 1, 3]:
        for gamma in ["scale", 0.01]:
            m = SVC(kernel="rbf", C=C, gamma=gamma).fit(dense(s_fit), s_fit.late)
            auc = roc_auc_score(v.late, m.decision_function(Xv))
            if auc > best_auc:
                best_cfg, best_auc = (C, gamma), auc
    s_trn = trn.iloc[rng.choice(len(trn), SVM_N, replace=False)]
    svm = SVC(kernel="rbf", C=best_cfg[0], gamma=best_cfg[1]).fit(dense(s_trn), s_trn.late)
    thr = best_threshold(v.late, svm.decision_function(Xv))
    scores["SVM (RBF)"] = svm.decision_function(dense(test))
    results.append(evaluate("SVM (RBF)", test.late, scores["SVM (RBF)"], thr))

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                        random_state=CFG["seed"]).fit(dense(fit), fit.late)
    thr = best_threshold(val.late, gb.predict_proba(dense(val))[:, 1])
    gb.fit(dense(trn), trn.late)
    scores["Gradient boosting (ref.)"] = gb.predict_proba(dense(test))[:, 1]
    results.append(evaluate("Gradient boosting (ref.)", test.late, scores["Gradient boosting (ref.)"], thr))
    best_thr = thr

    base = test.late.mean()
    results.append({"model": "Majority class (baseline)", "accuracy": max(base, 1 - base), "precision": base,
                    "recall": 1.0, "F1": 2 * base / (1 + base), "ROC_AUC": 0.5})
    table = pd.DataFrame(results).set_index("model").round(4)
    table.attrs["dt_depth"] = best_depth
    table.to_csv(proc.parent.parent / "reports" / "risk_metrics.csv")
    print(table.to_string())
    print("DT depth", best_depth, "SVM", best_cfg)

    test = test.assign(risk=scores["Gradient boosting (ref.)"])
    test.groupby(["zone_id", "hour"]).risk.mean().reset_index().to_csv(proc / "risk_zone_hour.csv", index=False)
    test.groupby("zone_id").risk.mean().rename("risk").reset_index().to_csv(proc / "zone_risk.csv", index=False)

    plot_roc(test.late.values, scores, figs)
    plot_confusion(test.late.values, (scores["Gradient boosting (ref.)"] >= best_thr).astype(int), figs)
    plot_importance(dt, NUM + list(ohe.get_feature_names_out(CAT)), figs)


def plot_roc(y, scores, out):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (name, s), c in zip(scores.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, label=f"{name} · AUC {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("Late-response risk: ROC on Nov–Dec", pad=24)
    style.subtitle(ax, "Predicting response > 8 min at call time")
    fig.savefig(out / "risk_roc.png")
    plt.close(fig)


def plot_confusion(y, pred, out):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap=SEQ)
    ax.grid(False)
    labels = ["On time", "Late"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n{v / cm.sum() * 100:.1f}%", ha="center", va="center", fontsize=10,
                    color=SURFACE if v > cm.max() * 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion matrix (best model)", pad=24)
    style.subtitle(ax, "Gradient boosting · threshold tuned on October (Youden J)")
    fig.savefig(out / "risk_confusion.png")
    plt.close(fig)


def plot_importance(dt, names, out):
    imp = pd.Series(dt.feature_importances_, index=names)
    groups = {}
    for n, v in imp.items():
        key = next((c for c in CAT if n.startswith(c + "_")), n)
        groups[key] = groups.get(key, 0) + v
    g = pd.Series(groups).sort_values().tail(10)
    nice = {"open_city": "Open incidents, citywide", "open_area": "Open incidents, dispatch area",
            "waiting_city": "Calls waiting for a unit", "load_60": "City calls, last 60 min", "zone_load_60": "Zone calls, last 60 min",
            "zone_how_rate": "Zone demand rate (this hour)", "zone_late_hist": "Zone historical late rate",
            "severity": "Severity code", "call_type": "Call type", "zone_id": "Zone", "sev_bucket": "Severity group",
            "time_band": "Time band", "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)", "dow_sin": "Weekday (sin)",
            "dow_cos": "Weekday (cos)", "is_weekend": "Weekend", "is_holiday": "Holiday"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh([nice.get(k, k) for k in g.index], g.values, color=SERIES[0], edgecolor=SURFACE, linewidth=2, height=0.65)
    for b, v in zip(bars, g.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f}", va="center", fontsize=9, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Decision-tree importance (grouped)")
    ax.set_title("What predicts a late response", pad=24)
    style.subtitle(ax, "Severity and how busy the fleet is dominate")
    fig.savefig(out / "risk_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
