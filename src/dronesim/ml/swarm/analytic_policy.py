"""Analytic (non-learned) placement policy — the baseline the trained policy
has to beat to justify existing.

The placement problem has more structure than a general RL task: the target's
crossing point is approximately a circular 2-D Gaussian centred on the
predictor's output (``intercept_predictor.py``), and each drone covers a disc
of fixed radius. Maximizing "at least one drone within kill radius of the
target" over that density is close to a classical disc-packing problem, and
packing has a known good answer — a hexagonal lattice, non-overlapping,
centred on the mode.

Two properties make this a genuinely strong baseline rather than a strawman:

- Spacing at exactly ``2 * hit_radius`` means adjacent kill discs touch but do
  not overlap. Overlap wastes coverage; gaps leak targets. Hex packing is the
  densest packing of equal circles in the plane, so this is the most probability
  mass N discs can cover for a radially symmetric density.
- Assignment of drones to lattice slots is solved optimally (Hungarian
  algorithm) rather than greedily, minimizing total travel so the formation is
  reached within the engagement window.

If a learned policy cannot beat this on both hit rate and separation
violations, the learned component is not earning its complexity — and that is
a legitimate result to report, not a failure to hide.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["hex_lattice_slots", "AnalyticPlacementPolicy"]


def hex_lattice_slots(n_slots: int, spacing: float, centre: np.ndarray) -> np.ndarray:
    """The ``n_slots`` hexagonal-lattice points closest to ``centre``.

    Generates a lattice comfortably larger than needed, then keeps the closest
    points — which yields a roughly circular cluster, matching the radial
    symmetry of the target's error distribution.
    """
    # Rings needed to contain n_slots points: 1 + 3k(k+1) points within k rings.
    rings = int(np.ceil((-3 + np.sqrt(9 + 12 * max(n_slots - 1, 0))) / 6)) + 1
    basis_a = np.array([spacing, 0.0])
    basis_b = np.array([spacing * 0.5, spacing * np.sqrt(3.0) / 2.0])

    points = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            points.append(i * basis_a + j * basis_b)
    points = np.array(points)

    order = np.argsort(np.linalg.norm(points, axis=1))
    return points[order[:n_slots]] + centre


class AnalyticPlacementPolicy:
    """Steers a swarm onto a hex-packed formation centred on the prediction.

    Emits the same action format as the learned policy (desired velocity per
    drone, normalized to [-1, 1]) so both can be evaluated through the exact
    same environment code path — otherwise the comparison would not be
    like-for-like.
    """

    def __init__(
        self,
        n_drones: int,
        hit_radius_m: float,
        min_separation_m: float,
        max_speed_mps: float,
        max_accel_mps2: float,
        repulsion_gain: float = 1.2,
    ):
        # Touching, non-overlapping kill discs — but never closer than the
        # swarm's own separation constraint allows.
        self.spacing = max(2.0 * hit_radius_m, min_separation_m)
        self.n_drones = n_drones
        self.min_separation_m = min_separation_m
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.repulsion_gain = repulsion_gain
        self._slots: np.ndarray | None = None

    def reset(self, predicted_point: np.ndarray) -> None:
        self._slots = hex_lattice_slots(self.n_drones, self.spacing, predicted_point)

    def act(self, positions: np.ndarray) -> np.ndarray:
        """Desired-velocity command (flattened, in [-1, 1]) toward assigned slots.

        Approach speed follows the time-optimal braking profile
        ``v = sqrt(2 * a * d)``, capped at ``max_speed``: the fastest a drone
        can close on a slot while still being able to stop on it. A fixed
        taper distance does not work here — at 25 m/s with 15 m/s^2 of
        authority the stopping distance is already 20.8 m, so a naive
        "slow down within 4 m" rule overshoots the slot by an order of
        magnitude and leaves the formation permanently unsettled (measured:
        mean slot left 3.2 m unfilled, larger than the 3 m kill radius, which
        alone roughly halved the hit rate).

        A short-range repulsion term is added on top. Without it, drones
        converging on the formation from opposite sides of the sector cross
        paths mid-transit and violate separation even though the *destination*
        formation is perfectly spaced — an artifact of ignoring transit, not a
        flaw in the packing. Including it keeps this an honest baseline rather
        than one handicapped on the exact axis the learned policy is expected
        to do well on.
        """
        if self._slots is None:
            raise RuntimeError("reset(predicted_point) must be called before act()")

        cost = np.linalg.norm(positions[:, None, :] - self._slots[None, :, :], axis=-1)
        drone_idx, slot_idx = linear_sum_assignment(cost)

        assigned = np.empty_like(positions)
        assigned[drone_idx] = self._slots[slot_idx]

        delta = assigned - positions
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        direction = delta / np.maximum(distance, 1e-9)
        braking_speed = np.sqrt(2.0 * self.max_accel_mps2 * distance)
        speed_frac = np.minimum(1.0, braking_speed / self.max_speed_mps)
        command = direction * speed_frac

        offsets = positions[:, None, :] - positions[None, :, :]
        gaps = np.linalg.norm(offsets, axis=-1)
        np.fill_diagonal(gaps, np.inf)
        # Linear falloff, reaching full strength at contact and zero at the
        # separation limit — enough to deflect a crossing pair without
        # destabilizing the approach.
        crowding = np.clip(1.0 - gaps / self.min_separation_m, 0.0, 1.0)
        push = np.sum(
            offsets / np.maximum(gaps, 1e-9)[..., None] * crowding[..., None], axis=1
        )
        command = command + self.repulsion_gain * push

        magnitude = np.linalg.norm(command, axis=1, keepdims=True)
        return (command / np.maximum(magnitude, 1.0)).ravel()
