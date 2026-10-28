import heapq

import numpy as np

from src.routing.graph import haversine_m


def astar(g, source, target, weight, speed):
    h = haversine_m(g.lat, g.lon, g.lat[target], g.lon[target]) / speed
    dist = np.full(g.n, np.inf)
    parent = np.full(g.n, -1, dtype=np.int64)
    done = np.zeros(g.n, dtype=bool)
    dist[source] = 0.0
    heap = [(h[source], source)]
    expanded = 0
    while heap:
        _, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expanded += 1
        if u == target:
            break
        s, e = g.indptr[u], g.indptr[u + 1]
        for j in range(s, e):
            v = g.head[j]
            nd = dist[u] + weight[j]
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd + h[v], v))
    return dist[target], parent, expanded
