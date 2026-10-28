import subprocess
import sys
import time

STEPS = [
    ("Phase 1  preprocess", "src.data.preprocess"),
    ("Phase 1  EDA figures", "src.evaluate.eda"),
    ("Phase 2  demand forecasting", "src.models.demand"),
    ("Phase 3  travel time + congestion", "src.models.travel_time"),
    ("Phase 4  late-response risk", "src.models.risk"),
    ("Phase 5  depot placement", "src.placement.kmeans_depots"),
    ("Phase 8  simulator calibration", "src.evaluate.simulate"),
    ("Phase 5  placement metrics (calibrated)", "src.placement.kmeans_depots"),
    ("Phase 6  routing evaluation", "src.evaluate.routing_eval"),
    ("Phase 7  Q-learning dispatcher", "src.rl.qlearning"),
    ("Phase 8  scenario comparison", "src.evaluate.scenarios"),
    ("Phase 9  web export", "src.evaluate.export_web"),
    ("Phase 9  single-file HTML", "src.evaluate.build_standalone"),
]


def main():
    only = set(sys.argv[1:])
    t_all = time.time()
    for name, module in STEPS:
        if only and module.split(".")[-1] not in only:
            continue
        print(f"\n=== {name} ({module})", flush=True)
        t = time.time()
        subprocess.run([sys.executable, "-W", "ignore", "-m", module], check=True)
        print(f"--- done in {time.time() - t:.0f}s", flush=True)
    print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} min. Open web/index.html to explore.")


if __name__ == "__main__":
    main()
