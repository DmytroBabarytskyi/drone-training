import sys; sys.path.insert(0,'src')
import numpy as np
from dronesim.core.config import load_config
from dronesim.swarm3d.engagement import Engagement

cfg = load_config('swarm3d/engagement')
e = Engagement(cfg, seed=4)
lock=None; rows=[]
while not e.step():
    if e._aim_tracking and lock is None:
        lock = e.t
    if lock is not None and abs((e.t-lock) % 0.5) < 1.2/240:
        pos = np.array([d.state.pos for d in e.drones])
        gaps = np.linalg.norm(pos[:,None,:]-e._slots[None,:,:],axis=-1)
        speeds = np.linalg.norm(np.array([d.state.vel for d in e.drones]),axis=1)
        rows.append((e.t-lock, float(np.mean(np.min(gaps,axis=0))), float(np.mean(speeds)),
                     float(np.max(speeds)), float(e._aim_point[0]), e.threat_range_to_plane_m))
print(f'true lateral y = {e.threat_lateral_y:.1f} m   aim locked at t={lock:.1f}s')
print(f"{'t-lock':>7s} {'form_err':>9s} {'mean|v|':>8s} {'max|v|':>7s} {'aim_y':>7s} {'threat rng':>11s}")
for r in rows:
    print(f'{r[0]:7.1f} {r[1]:8.1f}m {r[2]:7.1f} {r[3]:6.1f} {r[4]:7.1f} {r[5]:10.0f}m')
print(f'\nfinal miss {e._result.miss_distance_m:.1f} m')
