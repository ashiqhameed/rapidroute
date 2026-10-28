import heapq

import numpy as np


def dijkstra(g, sources, weight, target=None):
    sources = [sources] if np.isscalar(sources) else list(sources)
    dist = np.full(g.n, np.inf)
    parent = np.full(g.n, -1, dtype=np.int64)
    pedge = np.full(g.n, -1, dtype=np.int64)
    origin = np.full(g.n, -1, dtype=np.int64)
    done = np.zeros(g.n, dtype=bool)
    heap = []
    for k, s in enumerate(sources):
        dist[s] = 0.0
        origin[s] = k
        heap.append((0.0, s))
    heapq.heapify(heap)
    expanded = 0
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expanded += 1
        if u == target:
            break
        s, e = g.indptr[u], g.indptr[u + 1]
        for j in range(s, e):
            v = g.head[j]
            nd = d + weight[j]
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                pedge[v] = j
                origin[v] = origin[u]
                heapq.heappush(heap, (nd, v))
    return dist, parent, origin, expanded, pedge


def tree_cost(dist, parent, pedge, cost):
    out = np.full(len(dist), np.inf)
    for v in np.argsort(dist):
        if not np.isfinite(dist[v]):
            break
        out[v] = 0.0 if parent[v] == -1 else out[parent[v]] + cost[pedge[v]]
    return out


def path(parent, target):
    out = []
    v = target
    while v != -1:
        out.append(int(v))
        v = parent[v]
    return out[::-1]


def bellman_ford(g, source, weight):
    dist = np.full(g.n, np.inf)
    dist[source] = 0.0
    tail = g.edge_tail()
    for _ in range(g.n - 1):
        cand = dist[tail] + weight
        new = dist.copy()
        np.minimum.at(new, g.head, cand)
        if np.allclose(new, dist, equal_nan=True) or np.array_equal(new, dist):
            break
        dist = new
    if np.any(dist[tail] + weight < dist[g.head] - 1e-9):
        raise ValueError("negative cycle")
    return dist
