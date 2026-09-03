"""Full-physics 3-D interception engagement.

Runs the phases of a real engagement end to end: the mesh sits landed until a
threat is detected, scrambles, climbs, forms up in a vertical plane across the
threat approach corridor, refines its aim as cueing improves, and detonates
when the threat enters lethal radius.

Every interceptor is a real ``Multirotor`` on Bullet physics driven through the
engineered ``PosHoldController`` — the same stack a human pilot flies in this
simulator. The planar prototype in ``ml/swarm`` treated drones as point masses
that could change velocity instantly in any direction; here a drone must tilt
before it can accelerate sideways, so the formation is reached later and less
precisely. Comparing the two is the point of this module.

The threat is a kinematic body on a straight leg, not a controlled vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from dronesim.core.contracts import ControlCommand, FlightMode
from dronesim.ml.swarm.analytic_policy import hex_lattice_slots
from dronesim.ml.swarm.intercept_predictor import CueingSample, predict_plane_crossing
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.controllers.position import PosHoldController
from dronesim.vehicles.multirotor import Multirotor

__all__ = ["Phase", "EngagementResult", "Engagement"]


class Phase(Enum):
    """Engagement phases, in order. Each has a different control objective."""

    LANDED = "landed"        # on the pad, threat not yet detected
    CLIMB = "climb"          # ascending to intercept altitude
    FORM = "form"            # moving to assigned slot in the intercept plane
    TERMINAL = "terminal"    # seeker handover: refine aim on own observations
    RESOLVED = "resolved"    # detonated, or threat passed through


@dataclass
class EngagementResult:
    """Outcome of one engagement."""

    intercepted: bool = False
    detonated: bool = False
    miss_distance_m: float = float("inf")
    time_to_resolution_s: float = 0.0
    formation_error_m: float = float("inf")
    drones_within_lethal: int = 0
    phase_timeline: dict = field(default_factory=dict)


class Engagement:
    """One threat, one mesh, run to resolution."""

    def __init__(self, cfg: DictConfig, seed: int | None = None, parent=None):
        """``parent`` is an optional Panda3D NodePath. Passing the renderer's
        scene root makes every interceptor visible; omitting it keeps the whole
        engagement headless, which is how the batch evaluation runs it."""
        self.cfg = cfg
        self.parent = parent
        self.rng = np.random.default_rng(seed)

        self.dt = 1.0 / float(cfg.sim.physics_hz)
        self.control_every = max(1, int(cfg.sim.physics_hz / cfg.sim.control_hz))
        self.plane_x = float(cfg.mesh.intercept_plane_x_m)

        self.world = PhysicsWorld()
        self.world.add_ground_plane(z=0.0)

        self.drones: list[Multirotor] = []
        self.controllers: list[PosHoldController] = []
        for pad in self._pad_positions():
            drone = Multirotor(
                self.world, config_name=str(cfg.mesh.vehicle_config), parent=parent
            )
            drone.reset(pos=pad)
            controller = PosHoldController(drone)
            controller.reset(throttle_neutral=0.5)
            self.drones.append(drone)
            self.controllers.append(controller)

        # True threat track: flies along -X at fixed altitude, offset laterally.
        # The mesh never sees this directly, only noisy fixes of it.
        low, high = cfg.threat.lateral_offset_range_m
        self.threat_lateral_y = float(self.rng.uniform(low, high))
        self.threat_pos = np.array(
            [float(cfg.threat.spawn_x_m), self.threat_lateral_y, float(cfg.threat.altitude_m)]
        )
        self.threat_vel = np.array([-float(cfg.threat.speed_mps), 0.0, 0.0])

        self.phase = Phase.LANDED
        self.t = 0.0
        self._step_count = 0
        self._cueing: list[CueingSample] = []
        self._next_fix_t = 0.0
        # Where the mesh sits before the prediction is trustworthy: straight
        # across the corridor at intercept altitude. The asset is at the
        # origin, so this is the threat's most likely crossing point given no
        # usable track.
        self._nominal_aim = np.array([0.0, float(cfg.mesh.climb_target_alt_m)])
        self._aim_point = self._nominal_aim.copy()
        self._aim_tracking = False
        self._aim_confidence_m = float("inf")
        self._slots: np.ndarray | None = None
        self._commands: list[ControlCommand] = []
        self._phase_timeline: dict[str, float] = {}
        self._result = EngagementResult()

    def _pad_positions(self) -> list[np.ndarray]:
        """Ground pads in a grid, sited at the intercept plane."""
        n = int(self.cfg.mesh.n_drones)
        spacing = float(self.cfg.mesh.pad_spacing_m)
        side = int(np.ceil(np.sqrt(n)))
        pads = []
        for i in range(n):
            row, col = divmod(i, side)
            pads.append(
                np.array(
                    [
                        self.plane_x + (col - (side - 1) / 2.0) * spacing,
                        (row - (side - 1) / 2.0) * spacing,
                        0.5,  # just above the pad, not intersecting the ground plane
                    ]
                )
            )
        return pads

    # ---- threat and sensing ----------------------------------------------

    def _advance_threat(self) -> None:
        self.threat_pos = self.threat_pos + self.threat_vel * self.dt

    @property
    def threat_range_to_plane_m(self) -> float:
        return float(self.threat_pos[0] - self.plane_x)

    def _current_sensor_noise(self) -> float:
        """Radar cueing until the threat is close enough for onboard seekers."""
        handover = float(self.cfg.radar.seeker_handover_range_m)
        if self.threat_range_to_plane_m <= handover:
            return float(self.cfg.radar.seeker_noise_std_m)
        return float(self.cfg.radar.noise_std_m)

    def _collect_fix(self) -> None:
        if self.t < self._next_fix_t:
            return
        noise = self._current_sensor_noise()
        self._cueing.append(
            CueingSample(t=self.t, pos=self.threat_pos + self.rng.normal(0.0, noise, 3))
        )
        self._next_fix_t = self.t + float(self.cfg.radar.fix_interval_s)

    def _update_aim(self) -> None:
        """Re-predict the crossing point and slew the formation centre toward it.

        Uses a trailing window rather than the whole track: the straight-line
        model is only locally valid, and old fixes drag the fit back toward
        where the threat used to be.

        The aim point is *rate limited*, not gated or low-passed. Crossing-point
        uncertainty grows with the square of the extrapolation time, so early
        predictions are mostly noise — hundreds of metres of it — and steering
        at that signal makes the mesh chase a marker jittering across the sky,
        arriving as a clump that never forms a lattice. Two weaker fixes were
        tried first and both failed for the same underlying reason: a hard
        confidence gate leaves the mesh sitting at the nominal point until far
        too late to reposition, and a plain low-pass filter still lags a
        sustained offset by more than the kill radius.

        A slew limit solves both at once, because the quantity that should
        bound the formation's motion is not a filter constant but the vehicles
        themselves: the centre may move no faster than the drones can carry it.
        Fast jitter is rejected because it averages out inside the limit, while
        a real, sustained offset is tracked at exactly the rate the mesh can
        physically follow. Absurd predictions are still discarded outright —
        beyond a few formation radii the fit is not measuring anything.
        """
        window = self._cueing[-6:]
        prediction = predict_plane_crossing(window, self.plane_x, self._current_sensor_noise())
        if prediction is None:
            return

        self._aim_confidence_m = prediction.confidence_radius
        formation_radius = float(self.cfg.mesh.formation_spacing_m) * np.sqrt(len(self.drones))
        reject_beyond = formation_radius * float(self.cfg.mesh.aim_reject_multiplier)
        if prediction.confidence_radius > reject_beyond:
            return

        self._aim_tracking = True

        # Confidence-weighted update, in the form of a Kalman gain: trust a
        # prediction in proportion to how tight it is relative to the formation
        # the mesh is trying to hold. A prediction far coarser than the
        # formation barely moves the aim; one much tighter than it is followed
        # almost outright. This is the part that rejects noise - a pure slew
        # limit does not, it only slows the random walk down, and the aim was
        # measured wandering +/-15 m around the truth under one.
        confidence_ratio = prediction.confidence_radius / max(formation_radius, 1e-6)
        gain = 1.0 / (1.0 + confidence_ratio**2)

        # Slew limit on top, for the physical constraint the gain cannot express:
        # the formation centre is only meaningful if the drones can actually be
        # there, and they cannot move faster than this.
        delta = gain * (prediction.point - self._aim_point)
        distance = float(np.linalg.norm(delta))
        max_step = float(self.cfg.mesh.aim_slew_mps) * (self.control_every * self.dt)
        if distance > max_step:
            delta = delta * (max_step / distance)
        self._aim_point = self._aim_point + delta

    def _rebuild_slots(self) -> None:
        """Hex formation in the intercept plane, centred on the current aim.

        The intercept plane is vertical, spanned by world Y and Z, so a 2-D
        slot maps to (y, z) and every drone holds the same x.

        Slots are assigned to drones by solving the linear assignment problem,
        not by list order. Index-order assignment sends drones to whichever
        slot happens to share their index, so they cross the formation to swap
        places and arrive as a bunch; optimal assignment minimizes total travel
        and preserves the spread the lattice is for.
        """
        slots_2d = hex_lattice_slots(
            len(self.drones), float(self.cfg.mesh.formation_spacing_m), self._aim_point
        )
        ground_clearance_m = 15.0
        slots = np.column_stack(
            [
                np.full(len(slots_2d), self.plane_x),
                slots_2d[:, 0],
                np.maximum(slots_2d[:, 1], ground_clearance_m),
            ]
        )

        positions = np.array([d.state.pos for d in self.drones])
        cost = np.linalg.norm(positions[:, None, :] - slots[None, :, :], axis=-1)
        drone_idx, slot_idx = linear_sum_assignment(cost)
        assigned = np.empty_like(slots)
        assigned[drone_idx] = slots[slot_idx]
        self._slots = assigned

    # ---- control ----------------------------------------------------------

    def _desired_positions(self) -> np.ndarray:
        """Where each drone should be right now, given the phase."""
        if self.phase in (Phase.LANDED, Phase.CLIMB):
            targets = []
            for drone in self.drones:
                pos = drone.state.pos.copy()
                pos[2] = float(self.cfg.mesh.climb_target_alt_m)
                targets.append(pos)
            return np.array(targets)

        if self._slots is None:
            self._rebuild_slots()
        return self._slots

    def _recompute_commands(self) -> None:
        """Refresh each drone's stick command. Called at control rate."""
        targets = self._desired_positions()
        commands = []
        for drone, target in zip(self.drones, targets):
            delta = target - drone.state.pos
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                # PosHoldController takes stick-style input, so feed it a
                # bounded direction-to-target tapered near the slot; without
                # the taper the formation rings instead of settling.
                command_vector = (delta / distance) * min(1.0, distance / 10.0)
            else:
                command_vector = np.zeros(3)
            commands.append(
                ControlCommand(
                    roll=float(np.clip(command_vector[1], -1.0, 1.0)),
                    pitch=float(np.clip(command_vector[0], -1.0, 1.0)),
                    yaw=0.0,
                    throttle=float(np.clip(0.5 + command_vector[2] * 0.5, 0.0, 1.0)),
                    arm=True,
                    mode=FlightMode.POS_HOLD,
                )
            )
        self._commands = commands

    def _apply_control(self) -> None:
        """Run the inner control loop and apply motor thrust.

        Called on EVERY physics step, not at control rate. Thrust is a force
        applied per step, so skipping steps means the vehicle is unpowered for
        those steps — the same substep pattern ``scripts/eval_autonomous.py``
        uses, where the high-level command is held constant across substeps
        while the controller and motors keep running.
        """
        for drone, controller, cmd in zip(self.drones, self.controllers, self._commands):
            drone.apply_motor_commands(controller.update(cmd, drone.state, self.dt), self.dt)

    # ---- phase machine ----------------------------------------------------

    def _update_phase(self) -> None:
        previous = self.phase
        if self.phase is Phase.LANDED:
            # The engagement begins at detection range by construction, so the
            # mesh scrambles on the first control tick.
            self.phase = Phase.CLIMB
        elif self.phase is Phase.CLIMB:
            altitudes = np.array([d.state.pos[2] for d in self.drones])
            if float(np.mean(altitudes)) >= 0.9 * float(self.cfg.mesh.climb_target_alt_m):
                self.phase = Phase.FORM
        elif self.phase is Phase.FORM:
            if self.threat_range_to_plane_m <= float(self.cfg.radar.seeker_handover_range_m):
                self.phase = Phase.TERMINAL

        if self.phase is not previous:
            self._phase_timeline[self.phase.value] = round(self.t, 2)

    def _check_resolution(self) -> bool:
        """Detonate if the threat is inside lethal radius of any drone, or
        record a leaker once it has passed the intercept plane."""
        positions = np.array([d.state.pos for d in self.drones])
        distances = np.linalg.norm(positions - self.threat_pos, axis=1)
        self._result.miss_distance_m = min(
            self._result.miss_distance_m, float(np.min(distances))
        )

        lethal = float(self.cfg.warhead.lethal_radius_m)
        within = int(np.sum(distances <= lethal))
        if within > 0:
            self._result.detonated = True
            self._result.drones_within_lethal = within
            # Fragmentation against a hardened airframe is not a guaranteed
            # kill. Each drone in range contributes an independent chance, so
            # more drones on target help without making the outcome certain.
            pk = float(self.cfg.warhead.kill_probability)
            self._result.intercepted = bool(self.rng.random() < 1.0 - (1.0 - pk) ** within)
            return True

        if self.threat_pos[0] < self.plane_x - 50.0:
            return True  # leaker: through the plane and away
        return False

    # ---- main loop --------------------------------------------------------

    def step(self) -> bool:
        """Advance one physics step. Returns True when the engagement resolves."""
        self._collect_fix()

        if self._step_count % self.control_every == 0 or not self._commands:
            self._update_phase()
            if self.phase in (Phase.FORM, Phase.TERMINAL):
                self._update_aim()
                self._rebuild_slots()
            self._recompute_commands()

        self._apply_control()
        self.world.step(self.dt)
        for drone in self.drones:
            drone.advance_time(self.dt)
        self._advance_threat()

        self.t += self.dt
        self._step_count += 1

        if self._check_resolution() or self.t >= float(self.cfg.sim.max_duration_s):
            self._finalize()
            return True
        return False

    def _finalize(self) -> None:
        self.phase = Phase.RESOLVED
        self._result.time_to_resolution_s = round(self.t, 2)
        self._result.phase_timeline = dict(self._phase_timeline)
        if self._slots is not None:
            positions = np.array([d.state.pos for d in self.drones])
            gaps = np.linalg.norm(positions[:, None, :] - self._slots[None, :, :], axis=-1)
            self._result.formation_error_m = float(np.mean(np.min(gaps, axis=0)))

    def run(self) -> EngagementResult:
        while not self.step():
            pass
        return self._result

    def snapshot(self) -> dict:
        """Current state, for recording a trajectory to render later."""
        return {
            "t": self.t,
            "phase": self.phase.value,
            "threat": self.threat_pos.copy(),
            "drones": np.array([d.state.pos for d in self.drones]),
            "aim": self._aim_point.copy(),
        }
