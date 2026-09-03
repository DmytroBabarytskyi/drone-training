import numpy as np
print("THE THREE LEVERS, from N ~ 4*sigma^2 / r^2")
print("Scenario: cruise missile, mesh cued at 5 km (sigma = 90 m), 22 s to crossing")
print()
sigma=90.0
print(f"{'effective kill radius':>24s} {'mechanism':<34s} {'drones needed':>14s}")
for r,mech in [(3,"direct hit, bare airframe"),
               (10,"small frag warhead"),
               (30,"large frag / airburst"),
               (60,"net / tethered obstacle"),
               (120,"deployed curtain")]:
    print(f"{r:21.0f} m {mech:<34s} {4*sigma**2/r**2:14.0f}")
print()
print("Same, if cueing improves so the mesh is cued at 1 km (sigma = 26 m):")
sigma=26.0
for r in (3,10,30,60):
    print(f"{r:21.0f} m {'':34s} {4*sigma**2/r**2:14.0f}")
print()
print("COST EXCHANGE (order-of-magnitude, open-source figures)")
for name,cost,note in [("PAC-3 MSE interceptor", 4_000_000, "vs ballistic"),
                       ("AMRAAM (NASAMS)",       1_000_000, "vs cruise/aircraft"),
                       ("IRIS-T SLM",              400_000, "vs cruise"),
                       ("Gepard burst",              5_000, "vs Shahed, short range"),
                       ("FPV interceptor",           1_000, "vs Shahed, fielded"),
                       ("FPV + frag warhead",        3_000, "hypothetical")]:
    print(f"  {name:24s} ~${cost:>9,}   {note}")
print()
for n in (36,100,300):
    print(f"  mesh of {n:3d} FPV @ $3k = ${n*3000:>9,}  -> "
          f"{'cheaper' if n*3000 < 1_000_000 else 'NOT cheaper'} than one AMRAAM")
