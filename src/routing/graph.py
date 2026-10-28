from dataclasses import dataclass

import numpy as np

EARTH_R = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_R * 2 * np.arcsin(np.sqrt(a))


@dataclass
class RoadGraph:
    node_ids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    indptr: np.ndarray
    head: np.ndarray
    length: np.ndarray
    free_time: np.ndarray
    edge_lat: np.ndarray
    edge_lon: np.ndarray
    v_max: float

    @property
    def n(self):
        return len(self.node_ids)

    @property
    def m(self):
        return len(self.head)

    def neighbors(self, u):
        s, e = self.indptr[u], self.indptr[u + 1]
        return self.head[s:e], s, e

    def nearest_node(self, lat, lon):
        return int(np.argmin(haversine_m(lat, lon, self.lat, self.lon)))

    def edge_tail(self):
        return np.repeat(np.arange(self.n), np.diff(self.indptr))


def from_networkx(G):
    nodes = list(G.nodes)
    index = {u: i for i, u in enumerate(nodes)}
    lat = np.array([G.nodes[u]["y"] for u in nodes], dtype=float)
    lon = np.array([G.nodes[u]["x"] for u in nodes], dtype=float)

    best = {}
    for u, v, d in G.edges(data=True):
        a, b = index[u], index[v]
        if a == b:
            continue
        L = max(float(d["length"]), 0.5)
        t = max(float(d.get("travel_time", L / 8.33)), 0.05)
        if (a, b) not in best or t < best[(a, b)][1]:
            best[(a, b)] = (L, t)

    tail = np.array([k[0] for k in best], dtype=np.int64)
    head = np.array([k[1] for k in best], dtype=np.int64)
    length = np.array([v[0] for v in best.values()])
    free_time = np.array([v[1] for v in best.values()])
    order = np.lexsort((head, tail))
    tail, head, length, free_time = tail[order], head[order], length[order], free_time[order]
    indptr = np.zeros(len(nodes) + 1, dtype=np.int64)
    np.add.at(indptr, tail + 1, 1)
    indptr = np.cumsum(indptr)

    v_max = float(np.max(length / np.maximum(free_time, 1e-6)))
    return RoadGraph(
        node_ids=np.array(nodes), lat=lat, lon=lon, indptr=indptr, head=head, length=length,
        free_time=free_time, edge_lat=(lat[tail] + lat[head]) / 2, edge_lon=(lon[tail] + lon[head]) / 2,
        v_max=v_max,
    )


def load(path):
    import osmnx as ox

    return from_networkx(ox.load_graphml(path))
