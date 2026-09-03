import sys; sys.path.insert(0,'src')
import numpy as np
from dronesim.ml.swarm.intercept_predictor import CueingSample, predict_plane_crossing

print("A. BATTERY, NOT TIME, LIMITS STANDOFF")
print("   (interceptor quad, open-source figures: 5-15 min endurance)")
for endur_min, dash_frac in [(6,0.8),(12,0.6),(20,0.5)]:
    t = endur_min*60*dash_frac
    for v in (40.0, 55.0):
        # out-and-loiter: it does not need to come back, so full one-way
        print(f"   endurance {endur_min:2d} min, {int(dash_frac*100)}% at {v:.0f} m/s"
              f"  -> one-way dash {v*t/1000:5.1f} km")
print()

print("B. UNCERTAINTY COLLAPSES AS THE TARGET CLOSES (live radar feed)")
print("   Kh-101 class: 222 m/s, sensor noise 15 m, fixes every 1 s")
rng = np.random.default_rng(0); noise=15.0; v=222.0
print(f"   {'range to mesh':>14s} {'t_cross':>8s} {'sigma':>8s} {'drones for 2s':>14s}")
for rng_km in (30,20,10,5,2,1):
    d = rng_km*1000.0
    # fixes over the last 6 s of track before 'now'
    ts=[d/v - k for k in range(6,0,-1)]
    radii=[]
    for _ in range(400):
        s=[CueingSample(t=t0, pos=np.array([v*(d/v-t0),0,0])+rng.normal(0,noise,3)) for t0 in ts]
        p=predict_plane_crossing(s,0.0,noise)
        if p: radii.append(p.confidence_radius)
    sig=np.mean(radii)/np.sqrt(2)
    n_needed=4*sig**2/9.0   # kill radius 3 m
    print(f"   {rng_km:11d} km {d/v:7.1f}s {sig:7.1f}m {n_needed:13.0f}")
