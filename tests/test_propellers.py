"""Тести візуальних гвинтів (render/propellers.py, доп. фаза поліш)."""

import numpy as np

from dronesim.render.propellers import PropellerRig, add_propeller_visual

_MOTOR_POSITIONS = np.array(
    [
        (0.3, -0.3, 0.0),
        (0.3, 0.3, 0.0),
        (-0.3, -0.3, 0.0),
        (-0.3, 0.3, 0.0),
    ]
)
_SPIN_DIRS = np.array([1.0, -1.0, -1.0, 1.0])


def test_add_propeller_visual_is_positioned_correctly(engine):
    parent = engine.render.attachNewNode("test_parent_prop")
    try:
        prop = add_propeller_visual(parent, (0.3, -0.3, 0.0))
        pos = prop.getPos(engine.render)
        assert abs(pos.x - 0.3) < 1e-6
        assert abs(pos.y - (-0.3)) < 1e-6
    finally:
        parent.removeNode()


def test_propeller_rig_creates_four_propellers(engine):
    parent = engine.render.attachNewNode("test_parent_rig")
    try:
        rig = PropellerRig(parent, _MOTOR_POSITIONS, _SPIN_DIRS)
        assert len(rig._props) == 4
    finally:
        rig.close()
        parent.removeNode()


def test_propeller_rig_spins_faster_with_more_thrust(engine):
    parent = engine.render.attachNewNode("test_parent_rig2")
    try:
        rig = PropellerRig(parent, _MOTOR_POSITIONS, _SPIN_DIRS, max_spin_deg_per_s=360.0)
        max_thrust = 10.0

        rig.update(np.array([10.0, 10.0, 10.0, 10.0]), max_thrust, dt=1.0)
        full_throttle_heading = abs(rig._headings[0])

        rig2 = PropellerRig(parent, _MOTOR_POSITIONS, _SPIN_DIRS, max_spin_deg_per_s=360.0)
        rig2.update(np.array([2.0, 2.0, 2.0, 2.0]), max_thrust, dt=1.0)
        low_throttle_heading = abs(rig2._headings[0])
        rig2.close()

        assert full_throttle_heading > low_throttle_heading > 0.0
    finally:
        rig.close()
        parent.removeNode()


def test_propeller_rig_opposite_motors_spin_opposite_directions(engine):
    parent = engine.render.attachNewNode("test_parent_rig3")
    try:
        rig = PropellerRig(parent, _MOTOR_POSITIONS, _SPIN_DIRS)
        rig.update(np.full(4, 5.0), max_thrust_per_motor=10.0, dt=1.0)
        assert np.sign(rig._headings[0]) != np.sign(rig._headings[1])
    finally:
        rig.close()
        parent.removeNode()
