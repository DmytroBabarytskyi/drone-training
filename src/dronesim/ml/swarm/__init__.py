"""Planar swarm-formation layer: a mesh of interceptor drones held in a
fixed plane facing an incoming threat, which redistributes its own density
toward the point where the threat is predicted to cross.

This is deliberately separate from ``dronesim.vehicles`` / ``dronesim.physics``
(the full 6DOF Bullet simulation used elsewhere in this project). The problem
here is not "fly to a point in 3D space" — it is "N agents constrained to a
2D plane redistribute themselves without colliding, given an uncertain,
intermittently-updated cueing signal about where a fast-moving target will
cross that plane." Modelling that with full rigid-body flight dynamics would
add cost without adding anything the swarm-formation problem needs, so agents
here are point masses with a bounded acceleration and speed.

Two responsibilities are kept explicitly separate:

- ``intercept_predictor``: deterministic trajectory extrapolation (classical
  kinematics, not learned) — the same kind of computation a real fire-control
  system performs from radar cueing.
- ``swarm_env``: the learned part — a centralized policy that positions all
  N drones in the plane given the predictor's output, trained with PPO.
"""
