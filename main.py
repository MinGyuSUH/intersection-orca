"""AMRs on the cmpp_shelf warehouse map, avoiding each other with ORCA.

  python main.py                                   viewer
  python main.py run [N] [SEED] [SIM_S] [ALGO]     one headless run
  python main.py run 20 0 600 LiDAR-ORCA

ALGO is LiDAR-ORCA-HybFF (default) or LiDAR-ORCA. Vehicle and run settings are in config.json.
"""
import sys
import time

import numpy as np

from sim.config import Config
from sim.simulation import ALGORITHMS


def make_sim(algo="LiDAR-ORCA-HybFF", n=None, seed=None):
    config = Config.load()
    if n is not None:
        config.num_amrs = n
    if seed is not None:
        config.seed = seed
    return ALGORITHMS[algo](config)


def run(n=20, seed=0, sim_s=600.0, algo="LiDAR-ORCA-HybFF"):
    """Run headless and print arrivals and the number of steps with a car-car or car-wall overlap.
    Unlike the viewer, a headless run does not stop at the first overlap."""
    t0 = time.time()
    sim = make_sim(algo, n, seed)
    R = sim.radius
    car_steps = wall_steps = 0
    for _ in range(int(round(sim_s / sim.dt))):
        sim.step()
        P = np.array([a.pos for a in sim.agents])
        D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        car_steps += bool((D < 2 * R - 1e-3).any())
        wall_steps += any(sim.walls.nearest_wall_fast(p) < 3.0 and sim.walls.nearest_wall(p)[0] < 0.0
                          for p in map(tuple, P))
    print(f"{algo}  n={len(sim.agents)} seed={seed}  {sim_s:.0f}s simulated in {time.time() - t0:.1f}s")
    print(f"  arrivals={sim.arrivals}  ({sim.arrivals / (sim_s / 60):.2f}/min)")
    print(f"  steps with car-car overlap={car_steps}  steps with wall overlap={wall_steps}")


def viewer():
    from sim.viewer import run_viewer
    run_viewer(Config.load())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        a = sys.argv[2:]
        run(int(a[0]) if len(a) > 0 else 20, int(a[1]) if len(a) > 1 else 0,
            float(a[2]) if len(a) > 2 else 600.0, a[3] if len(a) > 3 else "LiDAR-ORCA-HybFF")
    else:
        viewer()
