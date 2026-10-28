# RapidRoute
### ML-driven ambulance placement and shortest-path dispatch

> **Author:** Ashiq Hameed
> **Live demo:** https://ashiqhameed.github.io/rapidroute/RapidRoute.html
> **Stack:** Python · scikit-learn · NetworkX · OSMnx · JavaScript
> **Focus:** Applying shortest-path algorithms (Dijkstra, A\*) to emergency response, with machine learning driving placement, dispatch and edge weights.

---

## 1. Problem statement

In an emergency, the time it takes an ambulance to arrive often decides the outcome. That response time depends on three decisions:

1. **Where ambulances wait** (depot placement)
2. **Which ambulance is sent** (dispatch)
3. **Which route it takes** (shortest path)

Classical systems make all three decisions with static rules: fixed stations, "send the nearest unit," and routes that minimise distance. But a city doesn't behave statically. Demand moves by hour and neighbourhood, and the same road takes very different times at 3 AM and 6 PM.

**RapidRoute** replaces each static rule with a learned one, and keeps a shortest path algorithm at the core of every decision.

| Decision | Static baseline | RapidRoute |
|---|---|---|
| Where to wait | Uniformly spread depots | K-means weighted by predicted demand |
| Edge cost for routing | Road length (metres) | Travel time scaled by a learned congestion factor for each zone and hour |
| Which unit to send | Nearest available unit | Nearest unit, or a Q-learning dispatch policy |
| Route | Dijkstra on distance | Dijkstra / A\* on predicted travel time |

**Objective:** minimise mean ambulance response time and maximise the share of incidents reached within **8 minutes**, especially critical calls.

---

## 2. Headline result

All **63,806 real Manhattan EMS calls from Nov–Dec 2024** (never used for training) were replayed through a calibrated discrete-event simulator with 110 ambulances and 8 depots.

| Scenario | Mean RT (min) | P90 RT (min) | % ≤ 8 min | % critical ≤ 8 min |
|---|---|---|---|---|
| A · Static baseline: uniform depots, distance routing, nearest unit | 11.35 | 15.19 | 46.9 | 47.7 |
| B · + plain K-means depots | 11.35 | 13.90 | 46.7 | 46.3 |
| **C · + demand-weighted K-means + ML travel-time routing** | **8.96** | **12.15** | **56.8** | **56.7** |
| D · C + Q-learning dispatch | 9.28 | 12.94 | 55.4 | 56.7 |
| *Observed, real FDNY, same period* | *12.33* | *25.23* | *37.2* | *58.7* |

**Going from A to C cuts mean response time by 2.4 minutes, adds 9.9 percentage points of calls reached within 8 minutes, and adds 9.0 points for critical calls.**

What drives the gain (ablation, % of calls within 8 minutes):

| Placement | Distance routing | ML travel-time routing |
|---|---|---|
| Uniform grid | 46.9 | 48.4 |
| K-means (plain) | 46.7 | 48.0 |
| **K-means (demand-weighted)** | 55.1 | **56.8** |
| K-means (risk-weighted) | 52.3 | 54.1 |

- **Demand-weighted placement** is the largest single gain, at about +8 points.
- **ML travel-time routing** adds about 1.5 minutes of saving to mean response time whatever the placement.
- **Plain K-means** (geography only) does not help.
- **Risk weighting** helped less than demand weighting. Demand weighting was selected on October validation data, never on the test months.

The gains hold across fleet sizes and are largest when the fleet is stretched: at 100 ambulances, C reaches 44.3% of calls within 8 minutes against A's 33.5%.

---

## 3. Why this is an ML project, not just a routing project

Shortest path is the engine. ML decides everything it runs on: the edge weights, the source nodes, which calls are risky, and which unit is sent.

```
            ┌──────────────── ML LAYER ───────────────────┐
 EMS calls ─┤ Demand forecast   → where calls will come   │
            │ Travel-time model → congestion factor/edge  │
            │ Late-risk model   → which calls are at risk │
            │ Weighted K-means  → depot (source) nodes    │
            │ Q-learning agent  → which depot dispatches  │
            └──────────────────────┬──────────────────────┘
                                   ▼
            ┌──────────── SHORTEST PATH CORE ─────────────┐
            │ Multi-source Dijkstra  → coverage map       │
            │ A* (admissible heuristic) → single route    │
            │ Dijkstra / Bellman–Ford from scratch,       │
            │ verified against NetworkX                   │
            └─────────────────────────────────────────────┘
```

### How it maps to the lab experiments

| Exp | Topic | Where it is used |
|---|---|---|
| 1 | Preprocessing, discretisation, feature engineering | Phase 1: cleaning, ZIP-to-zone reduction, cyclic encoding, time bands, fleet-load features |
| 2 | Naïve Bayes | Phase 4: Categorical NB on discretised features |
| 3 | Decision Tree | Phases 2, 3, 4: regressor and classifier, depth tuned |
| 4 | SVM | Phase 4: RBF SVM, C and γ tuned |
| 5 | Gradient descent and feature scaling | Phase 2: linear regression from scratch, scaled vs unscaled loss curves |
| 6 | K-means and K selection | Phase 5: depot placement, K chosen with Elbow, Silhouette, Davies–Bouldin and Calinski–Harabasz |
| 7 | Q-learning | Phase 7: dispatch agent |
| 8 | RL hyperparameter tuning | Phase 7: sweep over α, γ and ε-decay |

---

## 4. Data

| Source | Use | Link |
|---|---|---|
| NYC Open Data · EMS Incident Dispatch Data (`76xm-jjuj`) | 395,488 Manhattan incidents in 2024 | [dataset page](https://data.cityofnewyork.us/d/76xm-jjuj) |
| OpenStreetMap drive network (Overpass API) | 4,545 nodes and 9,750 directed edges | `scripts/get_osm.sh` |
| US Census 2024 ZCTA Gazetteer | ZIP centroids | [zip](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip) |

- **Split:** train on Jan–Oct, with October also used as the validation month for every model-selection decision. Test on Nov–Dec.
- **Locations:** NYC publishes incident locations at ZIP level only, for HIPAA reasons. The 96 raw ZIPs, many of them single buildings, are merged into **42 zones**, and each zone is snapped to its nearest road node.
- **Why NYC, not Mumbai:** Mumbai's 108 EMS call records are not public. The pipeline doesn't depend on the city (see Section 10).

---

## 5. Phases and what each found

### Phase 0: Data acquisition
`src/data/download.py` fetches the EMS incidents and the ZCTA file. The road network comes from `scripts/get_osm.sh`, which tries four Overpass mirrors with retries; this was needed because the macOS system Python cannot connect to Overpass. The script builds `data/raw/manhattan_drive.graphml`.

### Phase 1: Preprocessing and EDA *(Exp 1)*
- 390,951 clean incidents.
- **Features:** cyclic hour and weekday, US holidays, five time bands, three severity groups (critical 1–3, urgent 4–5, non-urgent 6–8), and **fleet load at call time**: open incidents citywide, open incidents in the dispatch area, and calls waiting for a unit.
- **Findings:**
  - Only **43.3%** of calls are reached within 8 minutes. The median response time is 8.9 minutes.
  - **37% of critical calls are late.**
  - Travel time is **1.30× slower at 3 PM than at 5 AM**.
  - Harlem and East Harlem have both the most calls and the highest share of late responses.

### Phase 2: Demand forecasting *(Exp 3, 5)*
Target: calls per zone per hour.

| Model | RMSE | R² (zone-hour) | R² (citywide) |
|---|---|---|---|
| Historical mean | 1.060 | 0.315 | 0.715 |
| Linear regression (GD from scratch) | 1.059 | 0.316 | 0.720 |
| Decision tree | 1.064 | 0.310 | 0.715 |
| **Gradient boosting (Poisson)** | **1.058** | **0.318** | **0.732** |

Zone-hour counts average about 1 call, so random (Poisson) noise caps R² at **0.367**. Gradient boosting reaches 87% of that ceiling. For Exp 5: gradient descent with scaled features converges, the same learning rate on unscaled features diverges within 12 iterations, and a tiny learning rate on unscaled features is still slow after 300.

### Phase 3: Travel time and learned congestion
- **Per-incident travel time:** gradient boosting has a mean error of **4.21 min**, against 4.60 for the zone × hour baseline. The decision tree and MLP both reach 4.25.
- **Why per-call error is limited:** the dataset doesn't record where the ambulance started. Even an "oracle" that uses the test-month medians per zone × hour × severity only reaches 3.96 min. At the zone-hour level the model is used at, the error is **0.79 min (10%)**.
- **Congestion factor** `c(zone, hour)`: learned only from critical calls made while the local fleet was quiet, so that road traffic is separated from ambulance availability. It averages 1.09× and peaks at **1.16× at 18:00** (1.35× worst zone-hour).
- **Dynamic edge weight:** `w(e, h) = free_flow_time(e) × c(zone(e), h) × κ`

### Phase 4: Late-response risk classifier *(Exp 2, 3, 4)*
Target: whether the response will take longer than 8 minutes, predicted from information known when the call arrives.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Majority class | 0.576 | – | 0.500 |
| Naïve Bayes | 0.671 | 0.685 | 0.747 |
| Decision tree | 0.680 | 0.688 | 0.757 |
| SVM (RBF) | 0.675 | 0.679 | 0.745 |
| Gradient boosting (reference) | 0.695 | 0.710 | 0.771 |

- **Why not predict severity?** NYC assigns severity from the call type, so predicting it would be leakage.
- **What limits accuracy:** 32% of calls end up within ±2 minutes of the 8-minute line. Away from that line, accuracy is **78–85%**.

### Phase 5: Depot placement *(Exp 6)*
- **Method:** K-means over the 4,545 road nodes, weighted by nothing (plain), by predicted demand, or by demand × (1 + risk).
- **K selection:** Elbow voted 8, Silhouette 4, Davies–Bouldin 4 and Calinski–Harabasz 14. **K = 8** follows the elbow and is close to the number of real FDNY EMS stations in Manhattan.
- **Results** (calibrated travel time from the nearest depot, all units free):

| Placement | Mean travel (min) | Demand within 8 min |
|---|---|---|
| Random (mean of 20) | 10.34 | 43.2% |
| Uniform grid | 7.51 | 62.2% |
| K-means (plain) | 7.46 | 55.8% |
| **K-means (demand)** | **6.93** | **65.0%** |
| K-means (risk-weighted) | 6.99 | 63.9% |

### Phase 6: Shortest path engine (core algorithm)
- **Implementations, all from scratch:**
  - Dijkstra with a binary heap
  - Multi-source Dijkstra, using a virtual super-source that connects to every depot
  - A\* with the admissible heuristic: haversine distance ÷ maximum edge speed
  - Bellman–Ford as a correctness check
- **Verification:** on 500 random trips they match `networkx.dijkstra_path_length` exactly, and Bellman–Ford matches Dijkstra on the full graph.
- **A\* vs Dijkstra:** A\* settles **36% fewer nodes** on average (2,257 → 1,448), and up to 53% fewer on trips under 2 km.
- **Cost of routing by distance:** routes chosen by shortest distance instead of travel time take **+87 s per trip at 04:00 and +111 s at 18:00**, and 74% of the routes differ.

### Phase 7: Q-learning dispatcher *(Exp 7, 8)*
- **Rule:** critical calls always get the nearest unit.
- **Decision:** for urgent and non-urgent calls, the agent chooses between the three nearest depots that have a free unit.
- **State:** nearest depot, severity, time band, and how many units are free at the 3 nearest depots.
- **Reward:** minus the extra travel minutes, minus a penalty for sending a depot's last unit, plus a bonus or penalty depending on whether the next critical call is reached in time.
- **Sweep on October:**
  - α ∈ {0.05, 0.1, 0.3}, γ ∈ {0.8, 0.9, 0.99}, ε-decay ∈ {0.99, 0.995, 0.999}
  - 800 simulated days per configuration
  - Best: α = 0.1, γ = 0.9, ε-decay = 0.999
- **Finding:** on October the agent slightly beats the nearest-unit rule for critical calls (54.3% vs 53.8% within 8 minutes). On the test months it matches the rule on critical calls but is slightly worse overall. Once depots and fleet are well sized, nearest-unit dispatch is already close to optimal. This is reported as a finding rather than hidden.

### Phase 8: Simulation, calibration and evaluation
- **Simulator:** a discrete-event replay of real calls with real timestamps, zones, severities and on-scene durations. Calls queue when no unit is free (critical calls first), and units return to their depot after each job.
- **Calibration on October** (two parameters):
  - **Speed factor κ = 2.95.** Free-flow map speeds are far faster than ambulances in real Manhattan traffic, which average about 13–14 km/h.
  - **Fleet size = 110 ambulances.** Each call keeps a unit busy for about 80 minutes, so a fleet of 60 would run at roughly 100% utilisation.

  | October | Real | Simulated |
  |---|---|---|
  | Median travel | 8.0 min | 8.0 min |
  | Median response | 9.2 min | 8.4 min |
  | P90 response | 26.6 min | 30.6 min |

- **Evaluation:** the four scenarios, an ablation of placement × routing, and fleet sensitivity at 100, 110 and 120 ambulances, all on Nov–Dec.

### Phase 9: Visuals and web interface
- **Figures:** 24 publication-ready figures in `reports/figures/`.
- **Interface:** `web/index.html` is a self-contained HTML/CSS/JavaScript app with a dark dispatch-console design. It needs no server. Online it uses the Barlow typeface from Google Fonts; offline it falls back to the system font.
  - **Live dispatch:** click anywhere on the full-screen Manhattan road map to place an emergency, and an ambulance animates along the chosen route. Dijkstra and A\* **run in the browser** on the real graph with the learned weights. The panel shows the dispatched depot, route, ETA, whether it meets the 8-minute target, nodes settled by each algorithm, the alternative depots, and the time lost by routing on distance. You can switch placement strategy, hour (congestion changes) and edge weights, and choose a map layer: coverage heat map, demand bubbles, or the nodes each algorithm explored.
  - **Results:** headline figures, the scenario table and comparison charts.
  - **Models:** every phase with its figures and metric tables.
  - **About:** pipeline, data and limitations.

---

## 6. Project structure

```
rapidroute/
├── README.md
├── RapidRoute.html
├── requirements.txt
├── config.yaml
├── run_all.py
├── scripts/
│   └── get_osm.sh
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── download.py
│   │   ├── manhattan_drive.overpassql
│   │   └── preprocess.py
│   ├── models/
│   │   ├── gd_linear.py
│   │   ├── demand.py
│   │   ├── travel_time.py
│   │   └── risk.py
│   ├── placement/
│   │   └── kmeans_depots.py
│   ├── routing/
│   │   ├── graph.py
│   │   ├── dijkstra.py
│   │   ├── astar.py
│   │   └── router.py
│   ├── rl/
│   │   └── qlearning.py
│   └── evaluate/
│       ├── style.py
│       ├── eda.py
│       ├── routing_eval.py
│       ├── simulate.py
│       ├── scenarios.py
│       ├── export_web.py
│       └── build_standalone.py
├── reports/
│   ├── figures/
│   └── *.csv
└── web/
    ├── index.html
    ├── style.css
    ├── app.js
    └── data.js
```

---

## 7. Tech stack

Python 3.9+, pandas, NumPy, scikit-learn, OSMnx, NetworkX, holidays, matplotlib, PyArrow and PyYAML. The web interface is plain HTML, CSS and JavaScript with no libraries.

---

## 8. How to run

**Just viewing:** double-click `RapidRoute.html` in the project root. It is the whole interface in one file, with the data and charts embedded, and can be shared on its own. `web/index.html` is the same interface but needs the rest of the folder next to it.

**Full pipeline:**

```bash
cd rapidroute
python3 -m pip install -r requirements.txt
bash scripts/get_osm.sh
python3 -m src.data.download
python3 run_all.py
open web/index.html
```

- `get_osm.sh` downloads the road map; skip it if `data/raw/manhattan.osm` already exists.
- `src.data.download` builds the road graph and downloads any missing data.
- `run_all.py` takes about 10 minutes.
- To rerun only some steps, name them: `python3 run_all.py scenarios export_web`.

---

## 9. Key design decisions

| Decision | Why |
|---|---|
| Predict late-response risk instead of severity | Severity is assigned from the call type, so predicting it would be leakage |
| Add fleet-load features (open incidents, calls waiting) | They are known to a dispatcher in real time and raised AUC from 0.759 to 0.771 |
| Learn congestion from quiet-fleet critical calls | Busy-fleet calls mix up road traffic with ambulance availability and produced a reversed daily pattern |
| K = 8 | Elbow method, and close to the number of real FDNY EMS stations |
| Fleet = 110 and κ = 2.95 | Both calibrated on October so the simulator reproduces real median travel, median response and P90 |
| Demand-weighted over risk-weighted placement | Chosen on October validation (53.5% vs 50.9% of calls within 8 minutes) |
| Critical calls always get the nearest unit | Never trade time on life-threatening calls; RL decides only for lower-priority calls |

---

## 10. Transfer to Mumbai

- **Road graph:** change `place` in `config.yaml` to `"Mumbai, Maharashtra, India"` and update the Overpass area.
- **Demand:** use 108 EMS logs if available, or a proxy built from ward population density, hospital locations and accident hotspots (MCGM / Mumbai Traffic Police open data).
- **Congestion:** fit the factors on Mumbai trip-time samples, such as the Google Distance Matrix API.
- **Spatial unit:** BMC wards or pincodes instead of NYC ZIPs.

---

## 11. Assumptions and limitations

- Incident locations are ZIP-level centroids, so travel time within a ZIP is approximated.
- Ambulance positions and unit status are not public. This bounds per-call travel-time and lateness prediction (Phases 3–4).
- The congestion factor is learned per zone and hour, not per road.
- The simulator sends units back to their depot after each job and does not model FDNY fire first-responders. Simulated critical-call performance (56.7%) is therefore conservative compared with the real 58.7%.
- The 8-minute target is a common EMS benchmark, not an official NYC standard.
- Absolute minutes depend on calibration. The relative improvement between scenarios is the main result.

---

## 12. References

1. NYC Open Data. *EMS Incident Dispatch Data* (76xm-jjuj). https://data.cityofnewyork.us/d/76xm-jjuj
2. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65.
3. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1.
4. Hart, P., Nilsson, N., Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Trans. SSC*, 4(2).
5. Watkins, C., Dayan, P. (1992). Q-learning. *Machine Learning*, 8.
6. Brotcorne, L., Laporte, G., Semet, F. (2003). Ambulance location and relocation models. *European Journal of Operational Research*, 147(3).
