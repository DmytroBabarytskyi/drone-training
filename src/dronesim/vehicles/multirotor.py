"""Спільна фізична реалізація мультиротора (X-конфігурація моторів) у Bullet.

Виділено з ``quad_fpv.py`` у фазі 3, коли з'явився другий конкретний апарат
(``quad_large.py``) з ІДЕНТИЧНОЮ фізичною моделлю (тверде тіло + 4 мотори у
X-конфігурації), що відрізняється лише параметрами з YAML. Обидва — тонкі
підкласи цього класу (docs/DECISIONS.md).

Фізична модель: одне тверде тіло (``BulletRigidBodyNode``, box-shape) + 4 точки
кріплення моторів. Кожен мотор штовхає вздовж локальної +Z осі тіла; сума
реактивних моментів дає обертання навколо Z (yaw); квадратичний опір (drag)
гальмує рух. Усі числові параметри — з ``configs/vehicles/*.yaml``.

``apply_motor_commands`` приймає вже обчислені тяги в Ньютонах (контракт
``Vehicle``, vehicles/base.py) — переклад нормованої команди мотора (0..1) у
Ньютони робить ``physics/aerodynamics.motor_thrust`` окремо (на боці
контролера), щоб апарат не залежав від конкретної кривої тяги.

ДОП. ФАЗА "РЕАЛІЗМ ФІЗИКИ" (усі нижче — опційні, увімкнені лише якщо конфіг
апарата має відповідне поле; відсутність поля = стара поведінка, повна
зворотна сумісність з апаратами/тестами без цих полів):
- Інерція моторів (``motor_tau_s``): тяга не стрибає миттєво, а розкручується/
  спадає з константою часу — потребує ``dt`` в ``apply_motor_commands``
  (``dt=None`` -> старий миттєвий відгук, як і раніше).
- Просідання батареї (``battery_full_throttle_s``/``battery_sag_coeff``):
  заряд (SoC) спадає з часом під навантаженням, миттєве просідання тяги під
  газом — ``VehicleState.battery`` тепер РЕАЛЬНО змінюється (був завжди 1.0).
- Екранний ефект (``rotor_radius_m``): більше тяги біля землі (формула
  Cheeseman-Bennett).
- Анізотропний квадратичний опір (``drag_coeff`` як 3-вектор, не скаляр):
  реальний опір ∝v², порахований У ЗВ'ЯЗАНИХ ОСЯХ тіла (різна ефективна площа
  вздовж X/Y/Z), а не у світових.
"""

from __future__ import annotations

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import NodePath, Point3, Quat, Vec3

from dronesim.core.config import load_config
from dronesim.core.contracts import VehicleState
from dronesim.physics import aerodynamics
from dronesim.physics.world import PhysicsWorld
from dronesim.vehicles.base import Vehicle

# X-конфігурація моторів: (x, y) знаки відносно центру (масштабуються arm_length)
# і напрямок обертання гвинта (spin_dir: +1 за годинниковою, -1 проти) — сусідні
# мотори обертаються назустріч, щоб компенсувати реактивний момент при завису.
# Структурна геометрія однакова для будь-якого X-квада; масштаб — з arm_length.
MOTOR_LAYOUT = (
    (1, -1, +1),  # перед-право, CW
    (1, 1, -1),  # перед-ліво, CCW
    (-1, -1, -1),  # зад-право, CCW
    (-1, 1, +1),  # зад-ліво, CW
)


class Multirotor(Vehicle):
    """X-квадрокоптер — кероване тверде тіло у Bullet. Базовий клас для конкретних апаратів."""

    MOTOR_LAYOUT = MOTOR_LAYOUT

    def __init__(self, physics_world: PhysicsWorld, config_name: str, parent: NodePath | None = None):
        """``parent`` — куди прикріпити ``NodePath`` тіла в графі сцени (типово
        ``engine.render``, фаза 2); ``None`` лишає апарат поза сценою (headless фаза 1)."""
        self.cfg = load_config(config_name)

        self._motor_positions = np.array(
            [(x * self.cfg.arm_length, y * self.cfg.arm_length, 0.0) for x, y, _ in MOTOR_LAYOUT]
        )
        self._spin_dirs = np.array([s for _, _, s in MOTOR_LAYOUT], dtype=np.float64)

        # Доп. фаза "реалізм фізики" — опційні параметри (0/відсутність поля =
        # ефект вимкнено, стара поведінка). ``cfg.get`` — той самий патерн, що
        # й в інших опційних полях YAML цього проєкту (напр. world/scene.py::kind).
        self._motor_tau_s = float(self.cfg.get("motor_tau_s", 0.0))
        self._battery_full_throttle_s = float(self.cfg.get("battery_full_throttle_s", 0.0))
        self._battery_sag_coeff = float(self.cfg.get("battery_sag_coeff", 0.0))
        self._rotor_radius_m = float(self.cfg.get("rotor_radius_m", 0.0))
        drag_cfg = self.cfg.get("drag_coeff", 0.0)
        self._drag_coeff_body = np.atleast_1d(np.asarray(drag_cfg, dtype=np.float64))

        body = BulletRigidBodyNode(self.__class__.__name__)
        body.addShape(BulletBoxShape(Vec3(*self.cfg.body_half_extent)))
        body.setMass(float(self.cfg.mass))
        ixx, iyy, izz = self.cfg.inertia
        body.setInertia(Vec3(ixx, iyy, izz))
        body.setDeactivationEnabled(False)  # апарат не повинен "засинати" у Bullet

        self._node = body
        self._np = physics_world.attach(body, parent=parent)
        self._t = 0.0
        self._motor_thrust_state = np.zeros(4)
        # Заряд батареї НЕ скидається в reset() (той самий акумулятор триває
        # крізь рестарти після краху в межах сесії, як і в реальності) — лише
        # тут, при СПРАВЖНЬОМУ конструюванні апарата.
        self._battery_soc = 1.0
        self.reset()

    def reset(self, pos: np.ndarray | None = None) -> None:
        pos = np.zeros(3) if pos is None else np.asarray(pos, dtype=np.float64)
        self._node.clearForces()
        self._node.setLinearVelocity(Vec3(0, 0, 0))
        self._node.setAngularVelocity(Vec3(0, 0, 0))
        self._np.setPos(Point3(*pos))
        self._np.setQuat(Quat.identQuat())
        self._t = 0.0
        self._motor_thrust_state = np.zeros(4)  # мотори "спинились" при рестарті

    def apply_motor_commands(self, motor_thrusts: np.ndarray, dt: float | None = None) -> None:
        """``dt`` — опційний: якщо задано (реальний крок фізики), вмикає
        інерцію моторів і просідання батареї для цього кроку; ``None``
        (дефолт) відтворює СТАРУ поведінку — миттєвий відгук тяги, без
        батареї (усі наявні тести/скрипти, що не передають ``dt``, лишаються
        коректними без жодної зміни)."""
        motor_thrusts = np.asarray(motor_thrusts, dtype=np.float64)
        if motor_thrusts.shape != (4,):
            raise ValueError(f"Очікується 4 тяги моторів, отримано форму {motor_thrusts.shape}")

        max_thrust = float(self.cfg.max_thrust_per_motor)
        target_thrusts = motor_thrusts

        if dt is not None and dt > 0.0:
            # Просідання батареї: миттєвий коеф. від поточного навантаження +
            # повільний drain SoC за час польоту (лише якщо параметри задані).
            load_frac = float(np.mean(target_thrusts)) / max_thrust if max_thrust > 0.0 else 0.0
            if self._battery_full_throttle_s > 0.0:
                self._battery_soc = max(
                    0.0, self._battery_soc - load_frac * dt / self._battery_full_throttle_s
                )
            if self._battery_sag_coeff > 0.0 or self._battery_full_throttle_s > 0.0:
                factor = aerodynamics.battery_thrust_factor(
                    self._battery_soc, load_frac, self._battery_sag_coeff
                )
                target_thrusts = target_thrusts * factor

            # Екранний ефект: більше РЕАЛІЗОВАНОЇ тяги біля землі.
            if self._rotor_radius_m > 0.0:
                height_m = float(self._np.getPos().z)
                ground_factor = aerodynamics.ground_effect_factor(height_m, self._rotor_radius_m)
                target_thrusts = target_thrusts * ground_factor

            # Інерція моторів: розкрутка/спад до цільової тяги, не стрибок.
            if self._motor_tau_s > 0.0:
                self._motor_thrust_state = aerodynamics.motor_lag(
                    self._motor_thrust_state, target_thrusts, self._motor_tau_s, dt
                )
                applied_thrusts = self._motor_thrust_state
            else:
                applied_thrusts = target_thrusts
        else:
            applied_thrusts = target_thrusts

        quat = self._np.getQuat()
        body_up_world = quat.xform(Vec3(0, 0, 1))  # напрям тяги (+Z тіла) у світових координатах

        for i, thrust in enumerate(applied_thrusts):
            world_offset = quat.xform(Vec3(*self._motor_positions[i]))
            self._node.applyForce(body_up_world * float(thrust), world_offset)

            torque_mag = aerodynamics.reactive_torque(
                float(thrust), self.cfg.thrust_to_torque, float(self._spin_dirs[i])
            )
            self._node.applyTorque(body_up_world * torque_mag)

        # Квадратичний опір — У ЗВ'ЯЗАНИХ ОСЯХ тіла (анізотропний, якщо
        # drag_coeff — 3-вектор), потім сила перекладається назад у світові.
        vel_world = self._node.getLinearVelocity()
        vel_body = quat.conjugate().xform(vel_world)
        drag_body = aerodynamics.quadratic_drag(
            np.array([vel_body.x, vel_body.y, vel_body.z]), self._drag_coeff_body
        )
        drag_world = quat.xform(Vec3(*drag_body))
        self._node.applyCentralForce(drag_world)

    def apply_external_force(self, force_world: np.ndarray) -> None:
        """Прикласти зовнішню силу (Н) у світових координатах до центру мас.

        Для збурень, що не походять від власних моторів апарата — вітер
        (``physics/wind.py``, фаза 3), а в майбутньому, можливо, зіткнення тощо.
        Сила в центрі мас не створює моменту (тільки лінійне прискорення) —
        свідоме спрощення моделі вітру (docs/DECISIONS.md).
        """
        self._node.applyCentralForce(Vec3(*force_world))

    def apply_external_torque(self, torque_world: np.ndarray) -> None:
        """Прикласти зовнішній момент (Н·м) у світових координатах (доп. фаза
        "реалізм фізики" — момент від турбулентного вітру, ``physics/wind.py
        ::Wind.torque_at``). Окремо від ``apply_external_force``, бо стала
        сила вітру (без турбулентності) свідомо БЕЗ моменту (docs/DECISIONS.md,
        фаза 3) — момент лише для стохастичної складової пориву."""
        self._node.applyTorque(Vec3(*torque_world))

    def advance_time(self, dt: float) -> None:
        """Викликати після кожного ``PhysicsWorld.step(dt)``, щоб ``state.t`` був коректним."""
        self._t += dt

    @property
    def node_path(self) -> NodePath:
        return self._np

    @property
    def motor_positions(self) -> np.ndarray:
        """Локальні (тіло-фрейм) позиції 4 моторів (доп. фаза поліш: розміщення
        візуальних гвинтів, render/propellers.py) — мірор ``GateRaceScenario.
        next_gate_index`` (фаза 7): публічний акцесор замість читання
        приватного ``_motor_positions`` з іншого модуля."""
        return self._motor_positions.copy()

    @property
    def motor_spin_dirs(self) -> np.ndarray:
        """Напрямок обертання кожного мотора (+1/-1, ``MOTOR_LAYOUT``) — для
        візуального обертання гвинтів у правильний бік (доп. фаза поліш)."""
        return self._spin_dirs.copy()

    @property
    def state(self) -> VehicleState:
        pos = self._np.getPos()
        quat = self._np.getQuat()
        vel = self._node.getLinearVelocity()
        ang_vel = self._node.getAngularVelocity()
        return VehicleState(
            pos=np.array([pos.x, pos.y, pos.z]),
            # УВАГА: скалярна (real) частина кватерніона Panda3D — це getR(), НЕ
            # getW() (getW() успадкований від LVecBase4-подібного класу і завжди
            # повертає 0 для Quat — виявлено емпірично у фазі 3, docs/DECISIONS.md).
            quat=np.array([quat.getI(), quat.getJ(), quat.getK(), quat.getR()]),  # -> xyzw
            vel=np.array([vel.x, vel.y, vel.z]),
            ang_vel=np.array([ang_vel.x, ang_vel.y, ang_vel.z]),
            motor_rpm=np.zeros(4),  # обертання моторів (spin-up) не моделюється
            battery=self._battery_soc,
            t=self._t,
        )
