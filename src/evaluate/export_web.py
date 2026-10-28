import json

import numpy as np
import pandas as pd

from src.config import CFG, ROOT, path
from src.placement.kmeans_depots import node_weights
from src.routing.router import Router


def table(name):
    p = ROOT / "reports" / name
    return pd.read_csv(p).round(4).to_dict(orient="records") if p.exists() else []


def table_indexed(name):
    p = ROOT / "reports" / name
    if not p.exists():
        return []
    d = pd.read_csv(p).round(4)
    d = d.rename(columns={d.columns[0]: "name"})
    return d.to_dict(orient="records")


def main():
    proc = path("processed")
    R = Router(kappa=1.0)
    g = R.g
    cal = json.loads((proc / "calibration.json").read_text())
    depots = json.loads((proc / "depots.json").read_text())
    selection = json.loads((proc / "selection.json").read_text())
    scen = json.loads((proc / "scenario_summary.json").read_text())
    routing = json.loads((proc / "routing_eval.json").read_text())
    pre = json.loads((proc / "preprocess_summary.json").read_text())
    rl = json.loads((proc / "rl_best.json").read_text())
    dem = pd.read_csv(proc / "zone_demand.csv").set_index("zone_id").pred_weekly_calls
    risk = pd.read_csv(proc / "zone_risk.csv").set_index("zone_id").risk
    inc = pd.read_parquet(proc / "incidents.parquet", columns=["zone_id", "late", "response_s"])
    zstats = inc.groupby("zone_id").agg(calls=("late", "size"), late=("late", "mean"), rt=("response_s", "median"))

    zones = []
    for _, z in R.zones.iterrows():
        zid = int(z.zone_id)
        zones.append({"id": zid, "zip": int(z.zip), "lat": round(float(z.lat), 5), "lon": round(float(z.lon), 5),
                      "node": int(R.zone_node[zid]), "weekly_pred": round(float(dem[zid]), 1),
                      "risk": round(float(risk[zid]), 3), "calls_2024": int(zstats.calls[zid]),
                      "late_pct": round(float(zstats.late[zid]) * 100, 1),
                      "median_rt_min": round(float(zstats.rt[zid]) / 60, 2)})

    data = {
        "meta": {"k": depots["k"], "votes": depots["votes"], "fleet": cal["fleet"], "kappa": round(cal["kappa"], 4),
                 "selected_placement": selection["placement"], "target_s": CFG["response_target_seconds"],
                 "preprocess": pre, "calibration": cal, "routing": routing, "rl": rl},
        "nodes": {"lat": np.round(g.lat, 5).tolist(), "lon": np.round(g.lon, 5).tolist()},
        "edges": {"indptr": g.indptr.tolist(), "head": g.head.tolist(), "length": np.round(g.length, 1).tolist(),
                  "free": np.round(g.free_time, 2).tolist(), "zone": R.edge_zone.tolist()},
        "factor": np.round(R.factor, 4).tolist(),
        "zones": zones,
        "depots": {k: [d["node"] for d in v] for k, v in depots["depots"].items()},
        "scenarios": scen,
        "tables": {
            "demand": table_indexed("demand_metrics.csv"),
            "risk": table_indexed("risk_metrics.csv"),
            "travel": table_indexed("travel_metrics.csv"),
            "placement": table("placement_results.csv"),
            "ablation": table("ablation_placement_routing.csv"),
            "fleet": table("scenario_fleet_sensitivity.csv"),
            "rl": table("rl_sweep.csv"),
            "k": table("placement_k_selection.csv"),
            "eda": table("eda_summary.csv"),
        },
    }
    out = ROOT / "web" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.RR = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
