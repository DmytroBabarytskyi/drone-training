import numpy as np

def radar_horizon_km(h_radar_m, h_target_m):
    """Geometric radar horizon, 4/3-Earth refraction: d(km) = 4.12*(sqrt(hr)+sqrt(ht))."""
    return 4.12*(np.sqrt(h_radar_m)+np.sqrt(h_target_m))

threats = [
    # name, speed_kmh, cruise_alt_m
    ("Shahed-136 (barrage drone)",      180,  1500),
    ("Shahed, low profile",             180,   100),
    ("Cruise missile (Kh-101/Kalibr)",  800,    75),
    ("Cruise missile, higher",          800,   500),
    ("Iskander-M (terminal)",          7000, 20000),
]

print("RADAR-HORIZON-LIMITED WARNING TIME (ground radar mast 20 m)")
print(f"{'threat':34s} {'v m/s':>7s} {'alt m':>7s} {'horizon km':>11s} {'warning s':>10s}")
for name, kmh, alt in threats:
    v = kmh/3.6
    d = radar_horizon_km(20, alt)
    print(f"{name:34s} {v:7.0f} {alt:7.0f} {d:11.1f} {d*1000/v:10.0f}")

print()
print("INTERCEPTOR SCRAMBLE BUDGET (from cold, on the ground)")
climb = 12.0   # m/s, loaded interceptor quad
vmax   = 55.0  # m/s (~200 km/h) dash
spin   = 5.0   # s arm + spin-up + launch
for name, kmh, alt in threats:
    v = kmh/3.6
    warn = radar_horizon_km(20, alt)*1000/v
    t_climb = alt/climb
    budget = warn - spin - t_climb
    reach = max(0.0, vmax*budget - 0.5*vmax*(vmax/20.0)) if budget>0 else 0.0
    verdict = "IMPOSSIBLE" if budget<=0 else ("tight" if reach<2000 else "workable")
    print(f"{name:34s} warn={warn:6.0f}s climb={t_climb:6.0f}s left={budget:7.0f}s "
          f"lateral reach={reach/1000:5.1f} km  {verdict}")
