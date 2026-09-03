import sys; sys.path.insert(0,'src')
import numpy as np
from omegaconf import OmegaConf
from dronesim.core.config import load_config
from dronesim.swarm3d.engagement import Engagement

base = load_config('swarm3d/engagement')
print(f"{'gate x':>7s} {'smooth':>7s} {'lock@':>7s} {'det':>5s} {'kill':>5s} {'med miss':>9s} {'med form':>9s}")
for gate_mult in (1.0, 3.0, 6.0):
    for smooth in (0.25, 0.08):
        cfg = OmegaConf.merge(base, OmegaConf.create(
            {'mesh': {'aim_smoothing': smooth, 'aim_gate_multiplier': gate_mult}}))
        miss=[];form=[];det=0;kill=0;locks=[]
        for seed in range(1, 9):
            e = Engagement(cfg, seed=seed); lock=None
            while not e.step():
                if e._aim_tracking and lock is None: lock = e.t
            r = e._result
            miss.append(r.miss_distance_m); form.append(r.formation_error_m)
            det += r.detonated; kill += r.intercepted
            locks.append(lock if lock is not None else 99.0)
        print(f"{gate_mult:7.1f} {smooth:7.2f} {np.mean(locks):6.1f}s {det:5d} {kill:5d} "
              f"{np.median(miss):8.1f}m {np.median(form):8.1f}m")
