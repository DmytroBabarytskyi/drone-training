"""Three-dimensional swarm interception engagement.

Where ``ml/swarm`` models placement as a planar point-mass abstraction for
training and analysis, this package runs the same engagement with the
project's real vehicle stack: Bullet rigid-body physics, the full multirotor
model (motor lag, battery sag, drag, ground effect) and the engineered
position-hold controller from ``vehicles/controllers``. It exists to check
whether the conclusions drawn from the planar abstraction survive contact
with actual flight dynamics — a quadrotor must tilt before it can accelerate
sideways, which the planar model ignored entirely.

The threat is modelled kinematically (a cruise-missile-class body on a
straight leg), not as a controlled vehicle: nothing here simulates a weapon,
only the geometry and timing of intercepting one.
"""
