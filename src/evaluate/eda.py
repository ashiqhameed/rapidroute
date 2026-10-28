import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import CFG, path
from src.evaluate import style
from src.evaluate.style import INK, INK_2, MUTED, SERIES, SEV_COLORS, SEQ, SURFACE

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TARGET_MIN = CFG["response_target_seconds"] / 60


def title(ax, main, sub):
    ax.set_title(main, pad=24)
    style.subtitle(ax, sub)


def hour_dow_heatmap(df, out):
    days = df.date.nunique() / 7
    m = df.groupby(["dow", "hour"]).size().unstack() / days
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("Avg calls per hour", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    title(ax, "EMS demand by hour and weekday", "Manhattan 2024 · average incidents per hour slot")
    fig.savefig(out / "eda_hour_dow_heatmap.png")
    plt.close(fig)


def zone_map(df, out):
    z = df.groupby(["zip", "lat", "lon"]).agg(calls=("incident_id", "size"), late=("late", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 9))
    sc = ax.scatter(
        z.lon, z.lat, s=z.calls / z.calls.max() * 900 + 30, c=z.late * 100, cmap=SEQ,
        edgecolor=SURFACE, linewidth=2, vmin=z.late.min() * 100, vmax=z.late.max() * 100,
    )
    for _, r in z.nlargest(6, "calls").iterrows():
        rad = np.sqrt(r.calls / z.calls.max() * 900 + 30) / 2
        east = r.lon > -73.945
        ax.annotate(str(int(r.zip)), (r.lon, r.lat), xytext=((rad + 3) if east else (-rad - 3), 0),
                    textcoords="offset points", fontsize=8.5, color=INK_2, va="center",
                    ha="left" if east else "right")
    ax.set_aspect(1 / np.cos(np.radians(40.78)))
    ax.set_xlim(z.lon.min() - 0.012, z.lon.max() + 0.02)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("% responses over 8 min", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    for n, lab in [(z.calls.quantile(0.25), "low"), (z.calls.max(), "high")]:
        ax.scatter([], [], s=n / z.calls.max() * 900 + 30, color=MUTED, label=f"{int(n):,} calls")
    ax.legend(loc="lower right", labelspacing=1.6, borderpad=1)
    title(ax, "Where calls come from", f"{len(z)} ZIP zones · size = call volume, colour = late share")
    fig.savefig(out / "eda_zone_map.png")
    plt.close(fig)


def response_hist(df, out):
    r = df.response_s.dropna() / 60
    within = (r <= TARGET_MIN).mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r.clip(upper=40), bins=np.arange(0, 40.5, 0.5), color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(TARGET_MIN, color=INK, lw=1.2, ls="--")
    ax.annotate(f"8-min target · {within:.1f}% of calls within", (TARGET_MIN, ax.get_ylim()[1] * 0.94),
                xytext=(14, ax.get_ylim()[1] * 0.94), fontsize=9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.text(39.75, ax.patches[-1].get_height() * 1.02, "40+", fontsize=8.5, color=INK_2, ha="center", va="bottom")
    ax.axvline(r.median(), color=MUTED, lw=1)
    ax.text(r.median() + 0.4, ax.get_ylim()[1] * 0.62, f"median {r.median():.1f} min", fontsize=9, color=INK_2)
    ax.set_xlabel("Incident response time (minutes, clipped at 40)")
    ax.set_ylabel("Incidents")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    title(ax, "Most calls miss the 8-minute target", "Call creation to first unit on scene · valid responses only")
    fig.savefig(out / "eda_response_hist.png")
    plt.close(fig)


def late_by_severity(df, out):
    g = df.dropna(subset=["late"]).groupby(["severity", "sev_bucket"]).late.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 500]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(g.severity.astype(str), g["mean"] * 100, color=g.sev_bucket.map(SEV_COLORS),
                  edgecolor=SURFACE, linewidth=2, width=0.7)
    for b, v in zip(bars, g["mean"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=INK_2)
    handles = [Patch(color=SEV_COLORS[k], label=lab) for k, lab in
               [("critical", "Critical (1–3)"), ("urgent", "Urgent (4–5)"), ("non_urgent", "Non-urgent (6–8)")]]
    ax.legend(handles=handles, loc="upper left")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Initial severity code (1 = most critical)")
    ax.set_ylabel("% responses over 8 min")
    crit = df[df.is_critical == 1].late.mean() * 100
    title(ax, "Late responses rise as severity drops", f"Share of incidents over 8 minutes · {crit:.0f}% of critical calls are still late")
    fig.savefig(out / "eda_late_by_severity.png")
    plt.close(fig)


def hourly_profile(df, out):
    days = df.date.nunique()
    calls = df.groupby("hour").size() / days
    travel = df.groupby("hour").travel_s.median() / 60
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].bar(calls.index, calls.values, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, width=0.85)
    axes[0].set_ylabel("Avg calls per hour")
    title(axes[0], "Demand by hour", "Average incidents per hour of day")
    axes[1].plot(travel.index, travel.values, color=SERIES[1], marker="o", ms=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    lo, hi = travel.idxmin(), travel.idxmax()
    for h in (lo, hi):
        axes[1].annotate(f"{travel[h]:.1f} min", (h, travel[h]), xytext=(0, 9 if h == hi else -15),
                         textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    pad = (travel.max() - travel.min()) * 0.18
    axes[1].set_ylim(travel.min() - pad, travel.max() + pad)
    axes[1].set_ylabel("Median travel time (min)")
    title(axes[1], "Travel time depends on the hour",
          f"Median ambulance travel · {travel[hi] / travel[lo]:.2f}× slower at peak")
    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "eda_hourly_profile.png")
    plt.close(fig)


def summary_table(df, out):
    rows = []
    for name, d in [("All", df), ("Train (Jan–Oct)", df[df.split == "train"]), ("Test (Nov–Dec)", df[df.split == "test"])]:
        rows.append({
            "set": name,
            "incidents": len(d),
            "valid_rt": int(d.response_s.notna().sum()),
            "median_response_min": round(d.response_s.median() / 60, 2),
            "median_travel_min": round(d.travel_s.median() / 60, 2),
            "pct_within_8min": round((1 - d.late.mean()) * 100, 1),
            "pct_critical": round(d.is_critical.mean() * 100, 1),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out.parent / "eda_summary.csv", index=False)
    print(t.to_string(index=False))


def main():
    style.apply()
    df = pd.read_parquet(path("processed") / "incidents.parquet")
    out = path("figures")
    hour_dow_heatmap(df, out)
    zone_map(df, out)
    response_hist(df, out)
    late_by_severity(df, out)
    hourly_profile(df, out)
    summary_table(df, out)


if __name__ == "__main__":
    main()
