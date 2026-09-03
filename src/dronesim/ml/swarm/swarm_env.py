"""Gymnasium environment for the swarm density-control problem.

The scenario: N interceptor drones are constrained to a fixed plane (the
"net") facing an incoming threat. Recon has already produced a predicted
crossing point and a confidence radius (``intercept_predictor.py`` — plain
kinematics, computed once per episode and held fixed, standing in for a
radar cueing report). The swarm's job is to redistribute itself within the
plane so that when the target actually arrives, enough drones are close
enough to it to intercept it — while never letting two drones collide, and
without knowing in advance exactly where within the confidence radius the
target will really be.

This is intentionally NOT the same problem as ``ml/rl/env.py``'s single-quad
navigation task. There the agent chases a point in 3D space using full
flight dynamics. Here N agents jointly solve a 2D formation problem, and the
whole swarm is controlled by one centralized policy — the action space is
the concatenation of every drone's acceleration command, the observation
space is the concatenation of every drone's state plus the shared cueing
signal. That is a deliberate simplification for a first working prototype:
a truly decentralized swarm (each drone acting on local information only,
learning to coordinate through communication or emergent convention) is a
harder problem and a natural next step, not this one.

Drones are point masses (position + velocity, bounded acceleration and
speed), not full 6DOF vehicles — see the package docstring in ``__init__.py``
for why full flight dynamics would add cost without adding anything this
problem needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronesim.ml.swarm.intercept_predictor import CueingSample, predict_plane_crossing

__all__ = ["SwarmInterceptEnv", "SwarmRewardConfig"]


@dataclass
class SwarmRewardConfig:
    """Numeric reward parameters — read from ``configs/ml/swarm_intercept.yaml``,
    never hardcoded (project convention, see SKILL.md rule 3)."""

    cluster_scale: float = 8.0          # per-step reward for sitting where target density is high
    collision_penalty: float = 5.0      # per violating pair, per step
    out_of_bounds_penalty: float = 1.0  # per drone outside the coverage rectangle, per step
    time_penalty: float = 0.01          # per step, encourages settling quickly
    hit_bonus_per_drone: float = 50.0   # awarded per drone within hit_radius at the final step
    hit_saturation_count: int = 3       # bonus stops growing past this many drones on target
    miss_penalty: float = 100.0         # applied once if zero drones are on target at the end


@dataclass
class SwarmSceneConfig:
    """Geometry and physics parameters for one episode. All numeric —
    read from YAML, never hardcoded."""

    n_drones: int = 9
    grid_spacing_m: float = 6.0          # lattice spacing drones start from
    coverage_half_extent_m: float = 18.0  # coverage rectangle is [-extent, extent]^2 in (y, z)
    min_separation_m: float = 2.5        # closer than this between two drones counts as a collision
    hit_radius_m: float = 2.0            # distance to the true crossing point that counts as "on target"
    max_speed_mps: float = 12.0
    max_accel_mps2: float = 8.0
    dt_s: float = 0.1
    decision_steps: int = 40             # episode length; dt_s * decision_steps ≈ engagement window

    plane_x_m: float = 0.0
    target_speed_range_mps: tuple[float, float] = (100.0, 160.0)  # ~360-575 km/h
    target_start_distance_range_m: tuple[float, float] = (400.0, 700.0)
    # Crossing point is drawn from a wider area than the coverage rectangle so the
    # policy also sees (and must accept losing) targets that are simply out of reach.
    crossing_area_half_extent_m: float = 24.0
    n_cueing_samples: int = 3
    cueing_interval_s: float = 1.0
    sensor_noise_std_m: float = 4.0


class SwarmInterceptEnv(gym.Env):
    """Centralized formation-control environment for N planar interceptor drones."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_cfg: SwarmSceneConfig | None = None,
        reward_cfg: SwarmRewardConfig | None = None,
    ):
        super().__init__()
        self.scene = scene_cfg or SwarmSceneConfig()
        self.reward_cfg = reward_cfg or SwarmRewardConfig()
        n = self.scene.n_drones

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * n,), dtype=np.float32)
        # obs = [predicted_point(2), confidence_radius(1), time_remaining_frac(1),
        #        per-drone pos(2) + vel(2)] * n
        obs_dim = 4 + 4 * n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._positions = np.zeros((n, 2), dtype=np.float64)
        self._velocities = np.zeros((n, 2), dtype=np.float64)
        self._predicted_point = np.zeros(2)
        self._confidence_radius = 0.0
        self._sigma = 1.0
        self._true_crossing_point = np.zeros(2)
        self._step_count = 0

    def _lattice_start_positions(self) -> np.ndarray:
        """Square-ish lattice of starting cells centred on the origin."""
        n = self.scene.n_drones
        side = int(round(np.sqrt(n)))
        coords = []
        for i in range(n):
            row, col = divmod(i, side)
            coords.append(
                [
                    (col - (side - 1) / 2.0) * self.scene.grid_spacing_m,
                    (row - (side - 1) / 2.0) * self.scene.grid_spacing_m,
                ]
            )
        return np.array(coords[:n], dtype=np.float64)

    def _simulate_target_and_cueing(self) -> tuple[np.ndarray, list[CueingSample]]:
        """Draw a random target trajectory, return its true (y, z) crossing
        point and the noisy cueing samples an observer would have reported
        while it approached."""
        rng = self.np_random
        extent = self.scene.crossing_area_half_extent_m
        true_point = rng.uniform(-extent, extent, size=2)

        speed = rng.uniform(*self.scene.target_speed_range_mps)
        start_distance = rng.uniform(*self.scene.target_start_distance_range_m)
        # Straight-line approach ending exactly at (plane_x, *true_point) — start_x is
        # ahead of the plane by start_distance, moving toward it at `speed`.
        start_x = self.scene.plane_x_m + start_distance
        travel_time = start_distance / speed
        start_pos = np.array([start_x, true_point[0], true_point[1]])
        velocity = np.array([-speed, 0.0, 0.0])

        samples = []
        for k in range(self.scene.n_cueing_samples):
            t = k * self.scene.cueing_interval_s
            if t >= travel_time:
                break
            true_pos_at_t = start_pos + velocity * t
            noise = rng.normal(0.0, self.scene.sensor_noise_std_m, size=3)
            samples.append(CueingSample(t=t, pos=true_pos_at_t + noise))

        return true_point, samples

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._positions = self._lattice_start_positions()
        self._velocities = np.zeros_like(self._positions)
        self._step_count = 0

        # Re-roll the cueing until it actually yields a prediction (a degenerate
        # fit is rare but possible with few, noisy samples) so every episode is
        # a well-posed engagement.
        for _ in range(20):
            true_point, samples = self._simulate_target_and_cueing()
            prediction = predict_plane_crossing(
                samples, self.scene.plane_x_m, self.scene.sensor_noise_std_m
            )
            if prediction is not None:
                break
        else:
            raise RuntimeError("Could not produce a valid cueing prediction after 20 tries")

        self._true_crossing_point = true_point
        self._predicted_point = prediction.point
        self._confidence_radius = prediction.confidence_radius
        self._sigma = max(self._confidence_radius / np.sqrt(2.0), 1e-6)

        return self._make_obs(), {}

    def _make_obs(self) -> np.ndarray:
        time_remaining_frac = 1.0 - self._step_count / self.scene.decision_steps
        head = np.array(
            [*self._predicted_point, self._confidence_radius, time_remaining_frac],
            dtype=np.float32,
        )
        body = np.concatenate([self._positions, self._velocities], axis=1).ravel()
        return np.concatenate([head, body]).astype(np.float32)

    def step(self, action: np.ndarray):
        n = self.scene.n_drones
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).reshape(n, 2)

        # The action is a DESIRED velocity (direction + magnitude, up to
        # max_speed_mps), not a raw acceleration — the environment tracks it
        # with a rate limiter (bounded by max_accel_mps2) rather than having
        # the policy integrate acceleration into velocity itself.
        #
        # An earlier version made the action a raw acceleration and let the
        # policy figure out braking on its own. Every reward-shaping attempt
        # on top of that (see `_compute_reward`'s docstring history in
        # docs/swarm_intercept_eval.md) produced the same failure: drones
        # accelerating straight to the coverage boundary and getting stuck
        # there, regardless of what the dense reward was pulling them toward.
        # That is the signature of a control problem, not a reward problem —
        # asking PPO to learn point-to-point steering AND its own braking
        # from scratch, inside a ~4-second episode, with only sparse-ish
        # shaping to go on. `ml/rl/env.py` never asks its agent to do this:
        # the RL policy there outputs navigation intent, and an engineered
        # `PosHoldController` (vehicles/controllers/position.py) handles
        # stabilization and braking underneath it — the same separation of
        # concerns this environment was missing. Desired-velocity-with-a-
        # rate-limiter is that same idea in a planar, physics-free setting:
        # the policy says where it wants to be heading, the rate limiter
        # supplies the "don't overshoot" that used to have to be learned.
        target_velocity = action * self.scene.max_speed_mps
        velocity_delta = target_velocity - self._velocities
        delta_norm = np.linalg.norm(velocity_delta, axis=1, keepdims=True)
        max_delta = self.scene.max_accel_mps2 * self.scene.dt_s
        scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-9))
        self._velocities += velocity_delta * scale

        self._positions += self._velocities * self.scene.dt_s

        extent = self.scene.coverage_half_extent_m
        out_of_bounds = np.any(np.abs(self._positions) > extent, axis=1)
        # Clamp position at the boundary and kill outward velocity — a soft wall,
        # not a crash: leaving the coverage area is undesirable, not catastrophic.
        clipped = np.clip(self._positions, -extent, extent)
        left_wall = clipped != self._positions
        self._velocities[left_wall] = 0.0
        self._positions = clipped

        self._step_count += 1
        is_final_step = self._step_count >= self.scene.decision_steps

        reward, info = self._compute_reward(out_of_bounds, is_final_step)
        obs = self._make_obs()
        return obs, reward, is_final_step, False, info

    def _compute_reward(self, out_of_bounds: np.ndarray, is_final_step: bool):
        cfg = self.reward_cfg
        n = self.scene.n_drones

        # Each drone is individually rewarded for sitting where the target is
        # actually likely to be — the Gaussian density (unnormalized, peak 1
        # at the predicted point) implied by the predictor's confidence
        # radius, not a hand-picked "N drones per m^2 near the centre" curve.
        # Collision avoidance (below) is what stops every drone from simply
        # stacking on that one peak — the "spread out" pressure comes from
        # repulsion, not from this term. (An earlier, more elaborate
        # discretized-checkpoint version of this same idea is documented,
        # with why it was replaced, in docs/swarm_intercept_eval.md — it
        # turned out not to be the actual problem; see the note on
        # acceleration vs. velocity control in ``step()``.)
        r_over_sigma = np.linalg.norm(self._positions - self._predicted_point, axis=1) / self._sigma
        density_reward = np.exp(-0.5 * r_over_sigma**2)
        reward = cfg.cluster_scale * float(np.mean(density_reward)) * self.scene.dt_s
        # Mean, not sum — same N-invariance argument as the separation penalty below.
        reward -= cfg.out_of_bounds_penalty * float(np.mean(out_of_bounds)) * self.scene.dt_s
        reward -= cfg.time_penalty

        pairwise = np.linalg.norm(
            self._positions[:, None, :] - self._positions[None, :, :], axis=-1
        )
        too_close = np.triu(pairwise < self.scene.min_separation_m, k=1)
        violating_pairs = int(np.sum(too_close))

        # Charge separation violations PER DRONE and scaled by dt, not per
        # unordered pair as a raw count. The pair count grows as N^2 while the
        # coverage reward above is a mean over drones and stays O(1), so a raw
        # per-pair penalty silently becomes the only term that matters as the
        # mesh grows: at N=25 a handful of violations outweighed the entire
        # achievable coverage reward by more than an order of magnitude, and
        # the optimal policy degenerated to "scatter to the boundary and
        # ignore the target" — which is exactly the failure that was observed
        # and misdiagnosed several times (see docs/swarm_intercept_eval.md).
        # Normalizing keeps the trade-off between "cover the likely area" and
        # "keep your distance" invariant to mesh size.
        violating_fraction = float(np.sum(too_close) * 2.0 / n)
        reward -= cfg.collision_penalty * violating_fraction * self.scene.dt_s

        hits = 0
        if is_final_step:
            dist_to_truth = np.linalg.norm(self._positions - self._true_crossing_point, axis=1)
            hits = int(np.sum(dist_to_truth <= self.scene.hit_radius_m))
            reward += cfg.hit_bonus_per_drone * min(hits, cfg.hit_saturation_count)
            if hits == 0:
                reward -= cfg.miss_penalty

        info = {
            "collision_pairs": violating_pairs,
            "hits": hits if is_final_step else None,
            "true_crossing_point": self._true_crossing_point.copy() if is_final_step else None,
        }
        return float(reward), info

    def close(self) -> None:
        pass
