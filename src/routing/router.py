import json

import numpy as np
import pandas as pd

from src.config import path
from src.routing import graph as rg
from src.routing.astar import astar
from src.routing.dijkstra import dijkstra, path as trace, tree_cost

MODES = ("distance", "freeflow", "ml")


class Router:
    def __init__(self, kappa=None):
        proc = path("processed")
        if kappa is None:
            f = proc / "calibration.json"
            kappa = json.loads(f.read_text())["kappa"] if f.exists() else 1.0
        self.g = rg.load(path("graph_raw"))
        self.zones = pd.read_csv(proc / "zones.csv")
        self.zone_node = np.array([self.g.nearest_node(a, b) for a, b in zip(self.zones.lat, self.zones.lon)])
        self.edge_zone = self._nearest_zone(self.g.edge_lat, self.g.edge_lon)
        self.node_zone = self._nearest_zone(self.g.lat, self.g.lon)
        c = pd.read_csv(proc / "congestion.csv").pivot(index="zone_id", columns="hour", values="factor")
        self.factor = c.loc[self.zones.zone_id].values
        self.kappa = kappa

    def _nearest_zone(self, lat, lon):
        d = np.stack([rg.haversine_m(lat, lon, a, b) for a, b in zip(self.zones.lat, self.zones.lon)])
        return d.argmin(axis=0)

    def weights(self, mode="ml", hour=12):
        if mode == "distance":
            return self.g.length
        if mode == "freeflow":
            return self.g.free_time * self.kappa
        if mode == "ml":
            return self.g.free_time * self.factor[self.edge_zone, hour] * self.kappa
        raise ValueError(mode)

    def true_seconds(self, hour):
        return self.weights("ml", hour)

    def tree(self, sources, hour=12, mode="ml"):
        w = self.weights(mode, hour)
        dist, parent, origin, expanded, pedge = dijkstra(self.g, sources, w)
        secs = tree_cost(dist, parent, pedge, self.true_seconds(hour))
        return secs, origin

    def route(self, src, dst, hour=12, mode="ml", method="astar"):
        w = self.weights(mode, hour)
        if method == "astar":
            speed = float(np.max(self.g.length / np.maximum(w, 1e-9))) if mode != "distance" else 1.0
            cost, parent, expanded = astar(self.g, src, dst, w, speed)
        else:
            dist, parent, _, expanded, _ = dijkstra(self.g, src, w, target=dst)
            cost = dist[dst]
        p = trace(parent, dst)
        secs = float(self.true_seconds(hour)[self.edge_ids(p)].sum()) if len(p) > 1 else 0.0
        return {"path": p, "cost": float(cost), "seconds": secs, "expanded": expanded}

    def edge_ids(self, p):
        ids = []
        for u, v in zip(p[:-1], p[1:]):
            s, e = self.g.indptr[u], self.g.indptr[u + 1]
            ids.append(s + int(np.flatnonzero(self.g.head[s:e] == v)[0]))
        return np.array(ids, dtype=np.int64)

    def depot_to_zone(self, depot_nodes, hour, mode="ml"):
        out = np.zeros((len(depot_nodes), len(self.zones)))
        for k, d in enumerate(depot_nodes):
            secs, _ = self.tree(int(d), hour, mode)
            out[k] = secs[self.zone_node]
        return out
