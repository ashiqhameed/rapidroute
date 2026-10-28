const RR = window.RR;
const N = RR.nodes.lat.length;
const LAT = Float64Array.from(RR.nodes.lat);
const LON = Float64Array.from(RR.nodes.lon);
const INDPTR = Int32Array.from(RR.edges.indptr);
const HEAD = Int32Array.from(RR.edges.head);
const LEN = Float64Array.from(RR.edges.length);
const FREE = Float64Array.from(RR.edges.free);
const EZONE = Int32Array.from(RR.edges.zone);
const M = HEAD.length;
const TAIL = new Int32Array(M);
for (let u = 0; u < N; u++) for (let j = INDPTR[u]; j < INDPTR[u + 1]; j++) TAIL[j] = u;
const KAPPA = RR.meta.kappa;
const TARGET = RR.meta.target_s;
const DISPATCH_S = 19;
const FIG = "../reports/figures/";
const figSrc = (n) => (window.RR_FIGS && window.RR_FIGS[n]) || FIG + n;

const state = {
  placement: RR.meta.selected_placement,
  hour: 18,
  mode: "ml",
  algo: "astar",
  layer: "coverage",
  incident: null,
  result: null,
  coverage: null,
  view: { s: 1, x: 0, y: 0 },
};

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const mins = (s) => `${fmt(s / 60, 1)} min`;

class Heap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v;
    let i = k.length;
    k.push(key); v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; i = p;
    }
    k[i] = key; v[i] = val;
  }
  pop() {
    const k = this.k, v = this.v;
    const topK = k[0], topV = v[0];
    const lk = k.pop(), lv = v.pop();
    const n = k.length;
    if (n) {
      let i = 0;
      while (true) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && k[c + 1] < k[c]) c++;
        if (k[c] >= lk) break;
        k[i] = k[c]; v[i] = v[c]; i = c;
      }
      k[i] = lk; v[i] = lv;
    }
    return [topK, topV];
  }
}

const weightCache = new Map();
function weights(mode, hour) {
  const key = mode + hour;
  if (weightCache.has(key)) return weightCache.get(key);
  const w = new Float64Array(M);
  if (mode === "distance") w.set(LEN);
  else for (let j = 0; j < M; j++) w[j] = FREE[j] * RR.factor[EZONE[j]][hour] * KAPPA;
  weightCache.set(key, w);
  return w;
}
const trueSeconds = (hour) => weights("ml", hour);

function dijkstra(sources, w, target = -1) {
  const dist = new Float64Array(N).fill(Infinity);
  const parent = new Int32Array(N).fill(-1);
  const pedge = new Int32Array(N).fill(-1);
  const origin = new Int32Array(N).fill(-1);
  const done = new Uint8Array(N);
  const h = new Heap();
  sources.forEach((s, k) => { dist[s] = 0; origin[s] = k; h.push(0, s); });
  let expanded = 0;
  while (h.size) {
    const [d, u] = h.pop();
    if (done[u]) continue;
    done[u] = 1; expanded++;
    if (u === target) break;
    for (let j = INDPTR[u]; j < INDPTR[u + 1]; j++) {
      const v = HEAD[j], nd = d + w[j];
      if (nd < dist[v]) { dist[v] = nd; parent[v] = u; pedge[v] = j; origin[v] = origin[u]; h.push(nd, v); }
    }
  }
  return { dist, parent, pedge, origin, expanded, visited: done };
}

function haversine(a, b, c, d) {
  const r = Math.PI / 180;
  const x = Math.sin(((c - a) * r) / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin(((d - b) * r) / 2) ** 2;
  return 6371000 * 2 * Math.asin(Math.sqrt(x));
}

function astar(src, tgt, w) {
  let speed = 0;
  for (let j = 0; j < M; j++) speed = Math.max(speed, LEN[j] / w[j]);
  const g = new Float64Array(N).fill(Infinity);
  const parent = new Int32Array(N).fill(-1);
  const pedge = new Int32Array(N).fill(-1);
  const done = new Uint8Array(N);
  const hfn = (v) => haversine(LAT[v], LON[v], LAT[tgt], LON[tgt]) / speed;
  const h = new Heap();
  g[src] = 0; h.push(hfn(src), src);
  let expanded = 0;
  while (h.size) {
    const [, u] = h.pop();
    if (done[u]) continue;
    done[u] = 1; expanded++;
    if (u === tgt) break;
    for (let j = INDPTR[u]; j < INDPTR[u + 1]; j++) {
      const v = HEAD[j], nd = g[u] + w[j];
      if (nd < g[v]) { g[v] = nd; parent[v] = u; pedge[v] = j; h.push(nd + hfn(v), v); }
    }
  }
  return { dist: g, parent, pedge, expanded, visited: done };
}

function tracePath(parent, pedge, t) {
  const nodes = [], edges = [];
  let v = t;
  while (v !== -1) { nodes.push(v); if (pedge[v] !== -1) edges.push(pedge[v]); v = parent[v]; }
  return { nodes: nodes.reverse(), edges: edges.reverse() };
}

function pathStats(edges, hour) {
  const ts = trueSeconds(hour);
  let s = 0, m = 0;
  for (const e of edges) { s += ts[e]; m += LEN[e]; }
  return { seconds: s, meters: m };
}

function treeTrue(res, hour) {
  const ts = trueSeconds(hour);
  const order = Array.from({ length: N }, (_, i) => i).filter((i) => Number.isFinite(res.dist[i]));
  order.sort((a, b) => res.dist[a] - res.dist[b]);
  const out = new Float64Array(N).fill(Infinity);
  for (const v of order) out[v] = res.parent[v] === -1 ? 0 : out[res.parent[v]] + ts[res.pedge[v]];
  return out;
}

const depots = () => RR.depots[state.placement];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function computeCoverage() {
  const res = dijkstra(depots(), weights(state.mode, state.hour));
  const secs = state.mode === "ml" ? res.dist : treeTrue(res, state.hour);
  let within = 0;
  for (let i = 0; i < N; i++) if (secs[i] <= TARGET - DISPATCH_S) within++;
  let wsum = 0, tsum = 0;
  RR.zones.forEach((z) => { wsum += z.weekly_pred; tsum += z.weekly_pred * secs[z.node]; });
  state.coverage = { secs, origin: res.origin, pctNodes: (within / N) * 100, meanDemand: tsum / wsum };
}

function dispatch(node, animate = true) {
  const hour = state.hour;
  const w = weights(state.mode, hour);
  const alts = depots().map((d, k) => {
    const r = dijkstra([d], w, node);
    const p = tracePath(r.parent, r.pedge, node);
    return { k, d, est: r.dist[node], ...pathStats(p.edges, hour) };
  }).sort((a, b) => a.est - b.est);
  const best = alts[0];

  const REPS = 25;
  let dj, as;
  const t0 = performance.now();
  for (let i = 0; i < REPS; i++) dj = dijkstra([best.d], w, node);
  const t1 = performance.now();
  for (let i = 0; i < REPS; i++) as = astar(best.d, node, w);
  const t2 = performance.now();
  const chosen = state.algo === "astar" ? as : dj;
  const path = tracePath(chosen.parent, chosen.pedge, node);
  const ps = pathStats(path.edges, hour);

  const other = state.mode === "ml" ? "distance" : "ml";
  const ow = weights(other, hour);
  const odBest = depots().map((d) => {
    const r = dijkstra([d], ow, node);
    const p = tracePath(r.parent, r.pedge, node);
    return { est: r.dist[node], ...pathStats(p.edges, hour) };
  }).sort((a, b) => a.est - b.est)[0];

  state.result = {
    node, depot: best.d, depotIdx: best.k, path, ...ps, alts,
    dj: { expanded: dj.expanded, ms: (t1 - t0) / REPS, visited: dj.visited },
    as: { expanded: as.expanded, ms: (t2 - t1) / REPS, visited: as.visited },
    other: { mode: other, seconds: odBest.seconds },
    zone: nearestZone(node),
  };
  renderDispatch();
  startRun(animate);
}

function nearestZone(node) {
  let best = null, bd = Infinity;
  for (const z of RR.zones) {
    const d = haversine(LAT[node], LON[node], z.lat, z.lon);
    if (d < bd) { bd = d; best = z; }
  }
  return best;
}

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const LAT0 = 40.78, COS0 = Math.cos((LAT0 * Math.PI) / 180);
const PX = new Float64Array(N), PY = new Float64Array(N);
for (let i = 0; i < N; i++) { PX[i] = LON[i] * COS0 * 1000; PY[i] = -LAT[i] * 1000; }

function fitView() {
  const r = canvas.getBoundingClientRect();
  const wide = r.width > 1100;
  const left = wide ? 332 : 16, right = wide ? 362 : 16;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < N; i++) { x0 = Math.min(x0, PX[i]); x1 = Math.max(x1, PX[i]); y0 = Math.min(y0, PY[i]); y1 = Math.max(y1, PY[i]); }
  const W = r.width - left - right, H = r.height - 32 - (wide ? 0 : 96);
  const s = Math.min(W / (x1 - x0), H / (y1 - y0)) * 0.94;
  state.view = { s, x: left + W / 2 - ((x0 + x1) / 2) * s, y: 16 + H / 2 - ((y0 + y1) / 2) * s };
}

const sx = (i) => PX[i] * state.view.s + state.view.x;
const sy = (i) => PY[i] * state.view.s + state.view.y;
const gx = (lon) => lon * COS0 * 1000 * state.view.s + state.view.x;
const gy = (lat) => -lat * 1000 * state.view.s + state.view.y;

const HEAT = [[0, "#8ff7e8"], [4, "#2dd4bf"], [7.9, "#14897f"], [8, "#f5a524"], [10, "#f97316"], [12, "#e5484d"]];
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
function heat(min) {
  if (!Number.isFinite(min)) return "#2a3954";
  if (min >= HEAT[HEAT.length - 1][0]) return HEAT[HEAT.length - 1][1];
  for (let i = 0; i < HEAT.length - 1; i++) {
    const [a, ca] = HEAT[i], [b, cb] = HEAT[i + 1];
    if (min <= b) {
      const f = (min - a) / (b - a), x = hex(ca), y = hex(cb);
      return `rgb(${x.map((v, k) => Math.round(v + (y[k] - v) * f)).join(",")})`;
    }
  }
  return HEAT[0][1];
}
const heatCSS = HEAT.map(([m, c]) => `${c} ${((m / 12) * 100).toFixed(1)}%`).join(",");
const WARM = ["#2a3954", "#7a5a2e", "#f5a524", "#f97316", "#e5484d"];
function warm(t) {
  t = Math.max(0, Math.min(1, t)) * (WARM.length - 1);
  const i = Math.floor(t), f = t - i, a = hex(WARM[i]), b = hex(WARM[Math.min(i + 1, WARM.length - 1)]);
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`;
}

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

let run = null;
let rafId = null;
const kick = () => { if (rafId === null) rafId = requestAnimationFrame(tick); };
function startRun(animate) {
  const p = state.result.path.nodes;
  const pts = p.map((v) => [PX[v], PY[v]]);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[cum.length - 1] || 1;
  const dur = animate && !reduceMotion ? Math.min(2600, 900 + total * 900) : 0;
  run = { pts, cum, total, t0: performance.now(), dur, progress: dur ? 0 : 1 };
  kick();
}

function tick(now) {
  rafId = null;
  if (!run) return;
  run.progress = run.dur ? Math.min(1, (now - run.t0) / run.dur) : 1;
  draw();
  if (run.progress < 1 || (state.result && !reduceMotion)) kick();
}

function along(f) {
  const target = f * run.total;
  let i = 1;
  while (i < run.cum.length && run.cum[i] < target) i++;
  if (i >= run.pts.length) return { x: run.pts[run.pts.length - 1][0], y: run.pts[run.pts.length - 1][1], i: run.pts.length - 1 };
  const seg = run.cum[i] - run.cum[i - 1] || 1, t = (target - run.cum[i - 1]) / seg;
  return { x: run.pts[i - 1][0] + (run.pts[i][0] - run.pts[i - 1][0]) * t, y: run.pts[i - 1][1] + (run.pts[i][1] - run.pts[i - 1][1]) * t, i };
}

function strokeEdges(filter, color, width) {
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath();
  for (let j = 0; j < M; j++) {
    if (filter && !filter(j)) continue;
    ctx.moveTo(sx(TAIL[j]), sy(TAIL[j])); ctx.lineTo(sx(HEAD[j]), sy(HEAD[j]));
  }
  ctx.stroke();
}

function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  const lw = Math.max(0.7, Math.min(2.6, state.view.s / 800));
  ctx.lineCap = "round";

  if (state.layer === "coverage" && state.coverage) {
    const secs = state.coverage.secs;
    const buckets = new Map();
    for (let j = 0; j < M; j++) {
      const m = secs[TAIL[j]] / 60;
      const key = Number.isFinite(m) ? Math.min(48, Math.round(m * 4)) : -1;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(j);
    }
    ctx.globalAlpha = 0.85;
    for (const [key, js] of buckets) {
      ctx.strokeStyle = key < 0 ? "#2a3954" : heat(key / 4);
      ctx.lineWidth = lw * 1.25;
      ctx.beginPath();
      for (const j of js) { ctx.moveTo(sx(TAIL[j]), sy(TAIL[j])); ctx.lineTo(sx(HEAD[j]), sy(HEAD[j])); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  } else {
    strokeEdges(null, "#2a3954", lw);
  }

  if (state.layer === "roads") {
    const maxD = Math.max(...RR.zones.map((z) => z.weekly_pred));
    for (const z of RR.zones) {
      const rad = 5 + Math.sqrt(z.weekly_pred / maxD) * 18;
      ctx.beginPath();
      ctx.arc(gx(z.lon), gy(z.lat), rad, 0, Math.PI * 2);
      ctx.fillStyle = warm((z.late_pct - 40) / 35);
      ctx.globalAlpha = 0.82; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(12,20,36,.9)"; ctx.stroke();
    }
  }

  if (state.layer === "explored" && state.result) {
    const dot = (vis, color, size) => {
      ctx.fillStyle = color;
      for (let i = 0; i < N; i++) if (vis[i]) { ctx.beginPath(); ctx.arc(sx(i), sy(i), size, 0, Math.PI * 2); ctx.fill(); }
    };
    dot(state.result.dj.visited, "rgba(167,180,204,.55)", 2);
    dot(state.result.as.visited, "#4c8dff", 2.2);
  }

  if (state.result && run) {
    const v = state.view;
    const pos = along(run.progress);
    const line = (upto, color, width, blur) => {
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
      if (blur) { ctx.shadowColor = color; ctx.shadowBlur = blur; }
      ctx.beginPath();
      run.pts.slice(0, upto).forEach(([x, y], i) => (i ? ctx.lineTo(x * v.s + v.x, y * v.s + v.y) : ctx.moveTo(x * v.s + v.x, y * v.s + v.y)));
      if (upto < run.pts.length) ctx.lineTo(pos.x * v.s + v.x, pos.y * v.s + v.y);
      ctx.stroke();
      ctx.restore();
    };
    line(run.pts.length, "rgba(232,238,249,.18)", 6, 0);
    line(pos.i, "#ffffff", 6, 18);
    line(pos.i, "#e5484d", 3.2, 0);
    if (run.progress < 1) {
      const ax = pos.x * v.s + v.x, ay = pos.y * v.s + v.y;
      ctx.save();
      ctx.shadowColor = "#ffffff"; ctx.shadowBlur = 16;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath(); ctx.arc(ax, ay, 6.5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.fillStyle = "#e5484d";
      ctx.fillRect(ax - 1.2, ay - 3.8, 2.4, 7.6); ctx.fillRect(ax - 3.8, ay - 1.2, 7.6, 2.4);
    }
  }

  depots().forEach((d, k) => {
    const x = sx(d), y = sy(d);
    const active = state.result && state.result.depot === d;
    const s = active ? 24 : 20;
    ctx.save();
    if (active) { ctx.shadowColor = "#e5484d"; ctx.shadowBlur = 22; }
    ctx.fillStyle = active ? "#e5484d" : "#e8eef9";
    roundRect(x - s / 2, y - s / 2, s, s, 6);
    ctx.fill();
    ctx.restore();
    ctx.lineWidth = 2; ctx.strokeStyle = "#0c1424";
    roundRect(x - s / 2, y - s / 2, s, s, 6); ctx.stroke();
    ctx.fillStyle = active ? "#ffffff" : "#0c1424";
    ctx.font = "700 13px 'Barlow Condensed', 'Arial Narrow', sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(k + 1), x, y + 0.5);
  });

  if (state.result) {
    const v = state.result.node, x = sx(v), y = sy(v);
    const t = (performance.now() % 1600) / 1600;
    if (!reduceMotion) {
      for (const off of [0, 0.5]) {
        const p = (t + off) % 1;
        ctx.beginPath(); ctx.arc(x, y, 8 + p * 22, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(229,72,77,${0.6 * (1 - p)})`; ctx.lineWidth = 2; ctx.stroke();
      }
    }
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#e5484d"; ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = "#ffffff"; ctx.stroke();
  }
  renderLegend();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

let legendKey = "";
function renderLegend() {
  const key = state.layer + state.hour;
  if (key === legendKey) return;
  legendKey = key;
  const el = document.getElementById("legend");
  if (state.layer === "coverage") {
    el.innerHTML = `Travel time from the nearest depot at ${String(state.hour).padStart(2, "0")}:00
      <div class="bar" style="background:linear-gradient(90deg,${heatCSS})"></div>
      <div class="ticks"><span>0</span><span>4</span><span>8 min target</span><span>12+</span></div>`;
  } else if (state.layer === "roads") {
    el.innerHTML = `Bubble size is predicted weekly calls. Colour is the 2024 late share.
      <div class="bar" style="background:linear-gradient(90deg,${WARM.join(",")})"></div>
      <div class="ticks"><span>40% late</span><span>75% late</span></div>`;
  } else {
    el.innerHTML = `Intersections each algorithm settled for the last call
      <div class="key"><span class="sw" style="background:#a7b4cc"></span>Dijkstra</div>
      <div class="key"><span class="sw" style="background:#4c8dff"></span>A*, a subset of Dijkstra's</div>`;
  }
}

function renderCoverageStats() {
  const c = state.coverage;
  document.getElementById("coverageStats").innerHTML = `
    <div class="stat"><span>Intersections reachable in 8 min</span><b>${fmt(c.pctNodes, 0)}%</b></div>
    <div class="stat"><span>Average trip, weighted by demand</span><b>${fmt(c.meanDemand / 60, 1)} min</b></div>
    <div class="stat"><span>Calibrated speed factor κ</span><b>${fmt(KAPPA, 2)}</b></div>
    <p class="hint">Travel only, every unit free, plus ${DISPATCH_S} s to dispatch.</p>`;
  document.getElementById("statusText").textContent = `${depots().length} depots on duty`;
}

function renderDispatch() {
  const r = state.result;
  const resp = r.seconds + DISPATCH_S;
  const ok = resp <= TARGET;
  const maxExp = Math.max(r.dj.expanded, r.as.expanded);
  const diff = r.other.seconds - r.seconds;
  const fewer = (1 - r.as.expanded / r.dj.expanded) * 100;
  const hh = String(state.hour).padStart(2, "0");
  const otherLabel = r.other.mode === "ml" ? "learned travel time" : "distance";
  const diffText = Math.abs(diff) < 1 ? "would choose the same route" : diff > 0 ? `would take ${fmt(diff, 0)} s longer` : `would be ${fmt(-diff, 0)} s faster`;
  const altRows = r.alts.slice(0, 4).map((a, i) =>
    `<div class="alt ${i === 0 ? "best" : ""}"><span class="chip">${a.k + 1}</span><span>Depot ${a.k + 1}</span><span class="t">${fmt(a.seconds / 60, 1)} min${i ? `<span class="gap">+${fmt((a.seconds - r.alts[0].seconds) / 60, 1)}</span>` : ""}</span></div>`).join("");
  document.getElementById("dispatch").innerHTML = `
    <h2>Call in ZIP ${r.zone.zip}</h2>
    <p class="where">${hh}:00, depot ${r.depotIdx + 1} responds</p>
    <div class="eta"><span class="num">${fmt(r.seconds / 60, 1)}</span><span class="unit">min to scene</span></div>
    <span class="verdict ${ok ? "ok" : "late"}">${ok ? "Meets" : "Misses"} the 8-minute target (${fmt(resp / 60, 1)} min with dispatch)</span>
    <div class="block">
      <h3>Route</h3>
      <div class="kv">
        <span>Distance</span><span>${fmt(r.meters / 1000, 2)} km</span>
        <span>Road segments</span><span>${r.path.edges.length}</span>
        <span>Weighted by</span><span>${state.mode === "ml" ? "Travel time" : "Distance"}</span>
      </div>
      <p class="saving">Routing by ${otherLabel} ${diffText}.</p>
    </div>
    <div class="block">
      <h3>How the search went</h3>
      <div class="race">
        <div class="lane"><span>Dijkstra</span><div class="track"><div class="fill" data-w="${(r.dj.expanded / maxExp) * 100}" style="background:#a7b4cc"></div></div><b>${r.dj.expanded}</b></div>
        <div class="lane"><span>A*</span><div class="track"><div class="fill" data-w="${(r.as.expanded / maxExp) * 100}" style="background:#4c8dff"></div></div><b>${r.as.expanded}</b></div>
      </div>
      <p class="saving">A* settled <b>${fmt(fewer, 0)}% fewer</b> intersections for the same optimal route (${fmt(r.as.ms, 2)} ms vs ${fmt(r.dj.ms, 2)} ms).</p>
    </div>
    <div class="block">
      <h3>Nearest depots</h3>
      <div class="alts">${altRows}</div>
    </div>
    <div class="block">
      <h3>This ZIP in 2024</h3>
      <div class="kv">
        <span>Calls</span><span>${r.zone.calls_2024.toLocaleString()}</span>
        <span>Predicted calls per week</span><span>${fmt(r.zone.weekly_pred, 0)}</span>
        <span>Responses over 8 min</span><span>${fmt(r.zone.late_pct, 1)}%</span>
        <span>Median response</span><span>${fmt(r.zone.median_rt_min, 1)} min</span>
        <span>Model late risk</span><span>${fmt(r.zone.risk * 100, 0)}%</span>
      </div>
    </div>`;
  requestAnimationFrame(() => document.querySelectorAll("#dispatch .fill").forEach((f) => { f.style.width = f.dataset.w + "%"; }));
}

function refresh() {
  computeCoverage();
  renderCoverageStats();
  legendKey = "";
  if (state.result) dispatch(state.result.node, false);
  else draw();
}

function nearestNode(px, py) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < N; i++) {
    const d = (sx(i) - px) ** 2 + (sy(i) - py) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function zoomAt(mx, my, f) {
  state.view.x = mx - (mx - state.view.x) * f;
  state.view.y = my - (my - state.view.y) * f;
  state.view.s *= f;
  draw();
}

let drag = null;
canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y, moved: false }; });
window.addEventListener("mousemove", (e) => {
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; canvas.classList.add("grabbing"); }
    state.view.x = drag.vx + dx; state.view.y = drag.vy + dy;
    draw();
    return;
  }
  hover(e);
});
window.addEventListener("mouseup", (e) => {
  if (drag && !drag.moved && e.target === canvas) {
    const r = canvas.getBoundingClientRect();
    dispatch(nearestNode(e.clientX - r.left, e.clientY - r.top));
  }
  canvas.classList.remove("grabbing");
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });

const tip = document.getElementById("tip");
function hover(e) {
  if (e.target !== canvas) { tip.style.opacity = 0; return; }
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const d = depots().findIndex((n) => (sx(n) - mx) ** 2 + (sy(n) - my) ** 2 < 160);
  let html = "";
  if (d >= 0) html = `<b>Depot ${d + 1}</b><br>${state.placement}`;
  else if (state.layer === "roads") {
    const z = RR.zones.find((z) => (gx(z.lon) - mx) ** 2 + (gy(z.lat) - my) ** 2 < 260);
    if (z) html = `<b>ZIP ${z.zip}</b><br>${z.calls_2024.toLocaleString()} calls in 2024, ${z.late_pct}% late<br>About ${fmt(z.weekly_pred, 0)} calls a week predicted`;
  } else if (state.layer === "coverage" && state.coverage) {
    const n = nearestNode(mx, my);
    if ((sx(n) - mx) ** 2 + (sy(n) - my) ** 2 < 100) html = `${fmt(state.coverage.secs[n] / 60, 1)} min from depot ${state.coverage.origin[n] + 1}`;
  }
  if (html) {
    tip.innerHTML = html; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 14 + "px"; tip.style.opacity = 1;
  } else tip.style.opacity = 0;
}

function seg(id, key) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
    state[key] = b.dataset.v;
    if (key === "layer") { legendKey = ""; draw(); } else refresh();
  }));
}

function initControls() {
  const sel = document.getElementById("placement");
  const names = { "Uniform grid": "Uniform grid (baseline)", "K-means (plain)": "K-means by area", "K-means (demand)": "K-means by demand ★", "K-means (risk-weighted)": "K-means by risk" };
  Object.keys(RR.depots).forEach((k) => {
    const o = document.createElement("option");
    o.value = k; o.textContent = names[k] || k;
    sel.appendChild(o);
  });
  sel.value = state.placement;
  sel.addEventListener("change", () => { state.placement = sel.value; refresh(); });
  const hour = document.getElementById("hour");
  hour.addEventListener("input", () => {
    state.hour = +hour.value;
    document.getElementById("hourLabel").textContent = `${String(state.hour).padStart(2, "0")}:00`;
    refresh();
  });
  seg("mode", "mode"); seg("algo", "algo"); seg("layer", "layer");
  document.getElementById("randomBtn").addEventListener("click", () => {
    const z = RR.zones[Math.floor(Math.random() * RR.zones.length)];
    const near = [];
    for (let i = 0; i < N; i++) if (haversine(LAT[i], LON[i], z.lat, z.lon) < 600) near.push(i);
    dispatch(near.length ? near[Math.floor(Math.random() * near.length)] : z.node);
  });
  const centre = () => { const r = canvas.getBoundingClientRect(); return [r.width / 2, r.height / 2]; };
  document.getElementById("zoomIn").addEventListener("click", () => zoomAt(...centre(), 1.4));
  document.getElementById("zoomOut").addEventListener("click", () => zoomAt(...centre(), 1 / 1.4));
  document.getElementById("resetBtn").addEventListener("click", () => { fitView(); draw(); });
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
    if (t.dataset.tab === "demo") { resize(); draw(); }
  }));
}

function tableHTML(rows, cols, cls = () => "") {
  if (!rows || !rows.length) return "<p class='muted'>Not available.</p>";
  return `<table><thead><tr>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>${rows.map((r) =>
    `<tr class="${cls(r)}">${cols.map(([k, , d]) => `<td>${typeof r[k] === "number" ? fmt(r[k], d ?? 2) : r[k] ?? "–"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function renderResults() {
  const sc = RR.scenarios.scenarios;
  const names = Object.keys(sc);
  const A = sc[names[0]], C = sc[names[2]];
  const kpi = (label, a, c, unit, better, max) => {
    const d = c - a;
    const good = better === "down" ? d < 0 : d > 0;
    const word = unit === "%" ? `${d > 0 ? "+" : "−"}${fmt(Math.abs(d), 1)} points` : `${d > 0 ? "+" : "−"}${fmt(Math.abs(d), 1)} min`;
    return `<div class="kpi"><div class="k">${label}</div><div class="v">${fmt(c, 1)}<small>${unit === "%" ? "%" : "min"}</small></div>
      <div class="d"><b style="color:${good ? "var(--teal)" : "var(--siren)"}">${word}</b> from ${fmt(a, 1)}${unit === "%" ? "%" : " min"} with static dispatch</div>
      <span class="from"><i style="width:${Math.min(100, (c / max) * 100)}%"></i></span></div>`;
  };
  document.getElementById("kpis").innerHTML =
    kpi("Mean response time", A.mean_rt_min, C.mean_rt_min, "min", "down", A.mean_rt_min) +
    kpi("90th-percentile response", A.p90_rt_min, C.p90_rt_min, "min", "down", A.p90_rt_min) +
    kpi("Calls reached in 8 minutes", A.pct_within_8, C.pct_within_8, "%", "up", 100) +
    kpi("Critical calls in 8 minutes", A.pct_critical_within_8, C.pct_critical_within_8, "%", "up", 100);
  const rows = names.map((n) => ({ name: n.replace(" · ", ": "), ...sc[n] }));
  const o = RR.scenarios.observed_test;
  rows.push({ name: "Real FDNY performance, same period", mean_rt_min: o.mean_rt_min, median_rt_min: o.median_rt_min, p90_rt_min: o.p90_rt_min, pct_within_8: o.pct_within_8, pct_critical_within_8: o.pct_critical_within_8, obs: true });
  document.getElementById("scenarioTable").innerHTML = tableHTML(rows, [
    ["name", "Scenario"], ["mean_rt_min", "Mean (min)"], ["median_rt_min", "Median (min)"], ["p90_rt_min", "P90 (min)"],
    ["pct_within_8", "Within 8 min (%)", 1], ["pct_critical_within_8", "Critical within 8 (%)", 1], ["pct_waited", "Queued (%)", 1],
  ], (r) => (r.obs ? "obs" : r.name.startsWith("C") ? "hl" : ""));
  document.getElementById("scenarioNote").textContent =
    `Every one of the ${RR.meta.preprocess.test_rows.toLocaleString()} real calls from November and December 2024 was replayed through the simulator with ${RR.meta.fleet} ambulances and ${RR.meta.k} depots. The speed factor (κ = ${fmt(KAPPA, 2)}) and the depot weighting were both chosen on October data, never on the test months.`;
}

function renderModels() {
  const T = RR.tables;
  const pick = (rows, key, val) => (rows || []).find((r) => String(r[key]).startsWith(val)) || {};
  const riskGB = pick(T.risk, "name", "Gradient"), riskDT = pick(T.risk, "name", "Decision"), riskNB = pick(T.risk, "name", "Na"), riskSVM = pick(T.risk, "name", "SVM");
  const demGD = pick(T.demand, "name", "Linear"), demGB = pick(T.demand, "name", "Gradient");
  const trGB = pick(T.travel, "name", "Gradient"), trMLP = pick(T.travel, "name", "Neural"), trDT = pick(T.travel, "name", "Decision");
  const kmD = pick(T.placement, "strategy", "K-means (demand)"), uni = pick(T.placement, "strategy", "Uniform");
  const rt = RR.meta.routing;
  const sc = RR.scenarios.scenarios, C = sc[Object.keys(sc)[2]], D = sc[Object.keys(sc)[3]];

  const summary = [
    ["Naïve Bayes", "Classification", "Will this call be late?", `AUC ${fmt(riskNB.ROC_AUC, 3)}`],
    ["Support Vector Machine (RBF)", "Classification", "Will this call be late?", `AUC ${fmt(riskSVM.ROC_AUC, 3)}`],
    ["Decision Tree", "Classification, regression", "Late risk, demand, travel time", `AUC ${fmt(riskDT.ROC_AUC, 3)}, travel error ${fmt(trDT.MAE_min, 2)} min`],
    ["Gradient Boosting", "Classification, regression", "Late risk, demand, travel time, congestion", `AUC ${fmt(riskGB.ROC_AUC, 3)}, travel error ${fmt(trGB.MAE_min, 2)} min`],
    ["Linear Regression (gradient descent, from scratch)", "Regression", "Calls per zone per hour", `R² ${fmt(demGD.R2_citywide, 2)} citywide`],
    ["Neural network (MLP)", "Regression", "Ambulance travel time", `error ${fmt(trMLP.MAE_min, 2)} min`],
    ["K-means", "Clustering", "Where to place the depots", `K = ${RR.meta.k}, ${fmt(kmD.mean_travel_min, 1)} min average trip`],
    ["Dijkstra and A*", "Graph search", "Fastest route to every call", `A* searches ${fmt((1 - rt.mean_nodes_astar / rt.mean_nodes_dijkstra) * 100, 0)}% less`],
    ["Q-learning", "Reinforcement learning", "Which depot to send", `${fmt(D.pct_critical_within_8, 1)}% critical calls in 8 min`],
  ];

  const sections = [
    { tag: "Classification", title: "Predicting late responses", models: "Naïve Bayes, SVM, Decision Tree, Gradient Boosting",
      text: [
        "Each model predicts, at the moment a call arrives, whether the ambulance will take longer than 8 minutes. The curve closer to the top-left corner is the better model.",
        `Gradient Boosting is best (AUC ${fmt(riskGB.ROC_AUC, 3)}), with the Decision Tree close behind. Severity and how busy the fleet is are the strongest signals.`],
      img: "risk_roc.png",
      table: tableHTML(T.risk, [["name", "Model"], ["accuracy", "Accuracy", 3], ["F1", "F1", 3], ["ROC_AUC", "AUC", 3]]) },
    { tag: "Regression", title: "Predicting demand and travel time", models: "Linear Regression (gradient descent), Decision Tree, MLP, Gradient Boosting",
      text: [
        "Regression models predict two things: how many calls each zone will get every hour, and how long an ambulance takes to reach a zone. The chart compares the travel-time models against a simple average.",
        `Gradient Boosting has the lowest error (${fmt(trGB.MAE_min, 2)} min per call). The learned travel times become the road weights the shortest-path search uses. The linear model, trained with gradient descent written from scratch, reaches R² ${fmt(demGD.R2_citywide, 2)} on citywide demand against ${fmt(demGB.R2_citywide, 2)} for Gradient Boosting.`],
      img: "travel_model_comparison.png",
      table: tableHTML(T.demand, [["name", "Demand model"], ["RMSE", "RMSE", 3], ["R2_citywide", "R² citywide", 3]]) },
    { tag: "Clustering", title: "Placing the depots", models: "K-means",
      text: [
        "K-means groups Manhattan's 4,545 road intersections into clusters, weighted by predicted demand. Each cluster centre becomes a depot.",
        `Four methods were used to choose K. The elbow gives K = ${RR.meta.k}, close to the number of real FDNY EMS stations. Demand-weighted K-means cuts the average trip from ${fmt(uni.mean_travel_min, 1)} to ${fmt(kmD.mean_travel_min, 1)} minutes compared with evenly spaced depots.`],
      img: "placement_k_selection.png", wide: true },
    { tag: "Graph search", title: "Finding the fastest route", models: "Dijkstra, A*",
      text: [
        `Both algorithms are written from scratch and return exactly the same fastest route as NetworkX on ${rt.pairs_tested} test trips. A* uses straight-line distance as a guide, so it explores less of the city.`,
        `On average A* settles ${Math.round(rt.mean_nodes_astar).toLocaleString()} intersections against Dijkstra's ${Math.round(rt.mean_nodes_dijkstra).toLocaleString()}, and the saving is largest on short trips.`],
      img: "routing_astar_vs_dijkstra.png" },
    { tag: "Reinforcement learning", title: "Choosing which depot to send", models: "Q-learning",
      text: [
        "Critical calls always get the nearest ambulance. For other calls, a Q-learning agent learns when to send a slightly farther unit so an area isn't left uncovered. The left chart shows it learning over 800 simulated days; the others show the effect of α, γ and ε-decay.",
        `It matches the nearest-unit rule on critical calls (${fmt(D.pct_critical_within_8, 1)}% vs ${fmt(C.pct_critical_within_8, 1)}% within 8 minutes) but doesn't beat it overall: with well-placed depots, sending the nearest unit is already close to optimal.`],
      img: "rl_sweep.png", wide: true },
  ];

  document.getElementById("modelSections").innerHTML = `
    <div class="sheet">
      <h2>Models used</h2>
      <p class="muted">Nine models across five families. Each section below shows the one chart that explains that family.</p>
      <div class="tablewrap"><table><thead><tr><th>Model</th><th>Type</th><th>Used for</th><th>Result</th></tr></thead><tbody>
        ${summary.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div>
    </div>
    ${sections.map((s) => `
    <div class="sheet">
      <div class="phase"><span class="n">${s.tag}</span>${s.models}</div>
      <h2>${s.title}</h2>
      <div class="model ${s.wide ? "wide" : ""}">
        <div class="imgs"><img src="${figSrc(s.img)}" alt="${s.title}" loading="lazy"></div>
        <div class="text">${s.text.map((t) => `<p>${t}</p>`).join("")}${s.table ? `<div class="tablewrap">${s.table}</div>` : ""}</div>
      </div>
    </div>`).join("")}`;
}

function renderAbout() {
  document.getElementById("aboutCard").innerHTML = `
    <div class="phase"><span class="n">ML Lab</span>Mini project, TY B.Tech AI &amp; DS, Semester V, 2026–27</div>
    <h2>Why RapidRoute</h2>
    <p>How fast an ambulance arrives depends on where units wait, which unit is sent and which route it drives. RapidRoute replaces each of those static rules with a learned one, and keeps a shortest-path algorithm at the centre of every decision.</p>
    <h2>How it works</h2>
    <ol>
      <li>Clean 395,000 NYC EMS incidents and add time, severity and fleet-load features.</li>
      <li>Forecast demand for every zone and hour.</li>
      <li>Learn road congestion by zone and hour and use it to weight every road.</li>
      <li>Classify each call's risk of a late response.</li>
      <li>Place ${RR.meta.k} depots with demand-weighted K-means.</li>
      <li>Route with Dijkstra, multi-source Dijkstra and A* on the learned weights.</li>
      <li>Train a Q-learning dispatcher.</li>
      <li>Replay every November and December call through a calibrated simulator and compare four ways of running the fleet.</li>
    </ol>
    <h2>Data</h2>
    <p>NYC Open Data EMS Incident Dispatch Data (<code>76xm-jjuj</code>) for Manhattan in 2024, the OpenStreetMap drive network (${RR.meta.routing.graph_nodes.toLocaleString()} intersections, ${RR.meta.routing.graph_edges.toLocaleString()} road segments), and the US Census 2024 ZCTA gazetteer.</p>
    <h2>Limitations</h2>
    <ul>
      <li>Incident locations are only published at ZIP level, so each zone is represented by its centre.</li>
      <li>Ambulance positions and unit status are not public, which limits how accurately single calls can be predicted.</li>
      <li>Simulated units return to their depot after every job, and fire first-responders are not modelled, so critical-call results are conservative.</li>
      <li>Absolute minutes depend on calibration; the improvement between scenarios is the main result.</li>
    </ul>
    <h2>Run it yourself</h2>
    <p>Regenerate everything with <code>python3 run_all.py</code>, then open <code>web/index.html</code>. No server is needed.</p>`;
}

function init() {
  initControls();
  renderResults();
  renderModels();
  renderAbout();
  resize();
  fitView();
  refresh();
  window.addEventListener("resize", () => { resize(); draw(); });
  if (document.fonts) document.fonts.ready.then(draw);
}

init();
