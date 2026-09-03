import sys; sys.path.insert(0,'src')
import numpy as np
from dronesim.ml.swarm.intercept_predictor import CueingSample, predict_plane_crossing

print("UNCERTAINTY vs RANGE (live radar feed, fixes every 1 s over the last 6 s)")
print("Kh-101 class: 222 m/s, sensor noise 15 m, kill radius 3 m")
print(f"{'range':>8s} {'t to cross':>11s} {'sigma':>8s} {'drones for 2sigma':>18s}")
rng=np.random.default_rng(0); noise=15.0; v=222.0
for rng_km in (30,20,10,5,2,1,0.5):
    d=rng_km*1000.0
    t_cross=d/v                       # crossing happens t_cross after 'now' (t=0)
    ts=np.arange(-5.0,0.1,1.0)        # six fixes ending now
    radii=[]
    for _ in range(400):
        s=[CueingSample(t=t0, pos=np.array([d - v*t0,0.0,0.0])+rng.normal(0,noise,3)) for t0 in ts]
        p=predict_plane_crossing(s,0.0,noise)
        if p: radii.append(p.confidence_radius)
    sig=np.mean(radii)/np.sqrt(2)
    print(f"{rng_km:6.1f}km {t_cross:10.1f}s {sig:7.1f}m {4*sig**2/9.0:17.0f}")
