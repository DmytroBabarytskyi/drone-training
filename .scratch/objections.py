import numpy as np

print("OBJECTION 1: MANOEUVRE DESTROYS LONG-HORIZON PREDICTION")
print("A straight-line fit assumes constant velocity. Lateral deviation from a")
print("turn grows as 0.5*a*t^2, so prediction horizon is bounded by manoeuvre,")
print("not by sensor noise.")
print()
v=222.0
print(f"{'t ahead':>9s} {'sigma from noise':>17s} {'deviation @1g':>14s} {'@3g':>10s}")
for t in (1,2,5,10,20,45):
    sigma_noise = 15.0*np.sqrt(1/6 + (t**2)/(2.5))   # rough OLS growth, 6 fixes over 5 s
    for_g = lambda g: 0.5*g*9.81*t*t
    print(f"{t:8.0f}s {sigma_noise:16.0f}m {for_g(1):13.0f}m {for_g(3):9.0f}m")
print()
print("-> beyond ~2-5 s ahead, a single turn dwarfs every sensor-noise term.")
print("-> long-horizon crossing prediction is not recoverable with better radar.")
print()

print("OBJECTION 2: CAN A NET/TETHER ACTUALLY KILL IT?")
m=2400.0   # Kh-101 class, kg
ke=0.5*m*v*v
print(f"  cruise missile KE at {v:.0f} m/s, {m:.0f} kg = {ke/1e6:.0f} MJ")
print(f"  TNT equivalent                          ~ {ke/4.184e6:.0f} kg")
print("  A net cannot absorb this. Stopping by energy capture is not the mechanism;")
print("  the only credible net/tether mechanism is shearing a control surface or")
print("  intake, which is a small target and a low probability per encounter.")
print()
print("  Honest effective kill radius for a 1-3 kg FPV payload vs a hardened")
print("  cruise-missile airframe (open-source SAM frag data, scaled down):")
for r,mech,pk in [(3,"contact / direct hit",0.9),
                  (5,"1 kg frag",0.5),
                  (8,"3 kg frag",0.4),
                  (60,"net, geometric span",0.05)]:
    print(f"    r={r:3d} m  {mech:22s} P(kill|within r) ~ {pk:.2f}"
          f"   effective r ~ {r*np.sqrt(pk):.1f} m")
print()

print("REVISED SIZING WITH HONEST NUMBERS (point defence, terminal engagement)")
print("Mesh sits over the defended asset; the missile must come to it, so no")
print("route prediction is needed - only the last seconds, where sigma is small.")
print()
print(f"{'cue range':>10s} {'sigma':>7s} {'r=5m eff 3.5':>13s} {'r=8m eff 5.1':>13s}")
for rng_km,sigma in [(5,90.0),(2,41.7),(1,25.9),(0.5,18.1)]:
    print(f"{rng_km:9.1f}km {sigma:6.1f}m {4*sigma**2/3.5**2:13.0f} {4*sigma**2/5.1**2:13.0f}")
