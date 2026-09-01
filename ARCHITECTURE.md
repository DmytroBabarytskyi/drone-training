# Архітектура

Мета архітектури — щоб кожен модуль мав **одну зрозумілу відповідальність** і
**вузький інтерфейс**, тож слабша модель-виконавець може працювати над одним
модулем, не тримаючи в голові весь проєкт.

## Головний принцип: цикл симуляції

Усе крутиться навколо фіксованого кроку симуляції (`dt`, за замовчуванням 1/240 с
для фізики, рендер/керування — рідше):

```
                 ┌─────────────────────────────────────────────┐
                 │                 SimLoop (core)                │
                 └─────────────────────────────────────────────┘
   input ──▶ ControlSource ──▶ FlightController ──▶ Vehicle ──▶ Physics
  (клав./                        (режими: Acro/       (мотори,      (Bullet у
   геймпад/                       Angle/GPS-hold)      маса, тяга)    Panda3D)
   ML-агент)                                                            ▼
                                                                    World state
   HUD/Render ◀── Renderer ◀── Camera(FPV) ◀───────────────────────────┘
                                    │
                                    ▼
                            ML: Detection / RL
```

Ключова ідея: **джерело керування взаємозамінне**. Людина з геймпада і
ML-агент виставляють один і той самий вектор команд
`[roll_rate, pitch_rate, yaw_rate, throttle]` (для Acro) або цільову позу.
Тому автономний режим — це просто інша реалізація `ControlSource`.

## Модулі (`src/dronesim/`)

| Пакет | Відповідальність | Ключовий інтерфейс |
|-------|------------------|--------------------|
| `core/` | Цикл симуляції, годинник, шина подій, реєстр компонентів | `SimLoop`, `SimClock`, `Config` |
| `physics/` | Обгортка Bullet (через `panda3d.bullet`): світ, гравітація, крок, колізії | `PhysicsWorld.step()`, `add_body()` |
| `vehicles/` | Моделі апаратів: FPV-квадрокоптер, великий мультиротор, fixed-wing | `Vehicle.apply_controls()`, `Vehicle.state` |
| `physics/aerodynamics` | Тяга моторів, момент, опір, вітер | `motor_thrust()`, `drag()` |
| `input/` | Читання клавіатури/геймпада → нормовані осі | `ControlSource.get_command()` |
| `vehicles/controllers` | Польотні режими: Acro/Rate, Angle, GPS/Alt-hold, PID | `FlightController.update()` |
| `render/` | Panda3D-сцена, FPV-камера, вікно, PBR-освітлення, HUD-оверлей | `Renderer.draw()`, `Camera.get_frame()` |
| `world/` | Сцена, ландшафт, перешкоди, спавн об'єктів | `Scene.load()`, `Scene.spawn()` |
| `scenarios/` | Місії: воріт-рейс, патруль, полігон із цілями, підрахунок очок | `Scenario.reset()`, `Scenario.step()` |
| `ml/detection/` | YOLO: інференс, навчання, експорт | `Detector.predict(frame)` |
| `ml/rl/` | Gymnasium-середовище, тренування RL-агента наведення | `DroneEnv`, `train.py` |
| `ml/data/` | Генерація синтетичних датасетів із симуляції (кадр+bbox) | `DatasetRecorder` |
| `ui/` | CLI, меню, налаштування keybind/curves | `app.py` |
| `utils/` | Математика (кватерніони, PID), логування, конфіг | `pid.py`, `math3d.py` |

## Контракти даних (щоб модулі не «злипались»)

- **`ControlCommand`** — dataclass: `roll, pitch, yaw, throttle ∈ [-1, 1]`
  (+ `arm: bool`, `mode: FlightMode`). Спільний для людини і ML.
- **`VehicleState`** — dataclass: `pos, quat, vel, ang_vel, motor_rpm, battery`.
- **`CameraFrame`** — `rgb: np.ndarray (H,W,3)`, `depth`, `pose`, `t`.
- **`Detection`** — `bbox_xywh, class_id, conf`.
- **`Observation`/`Action`** для RL — визначені у `ml/rl/env.py` через
  Gymnasium `spaces`.

Ці контракти живуть у `core/contracts.py`. **Змінювати їх — узгоджено**, бо від
них залежать усі інші пакети (див. правило в SKILL.md).

## Взаємозамінні реалізації (стратегія розширення)

- **Рендер:** Panda3D-сцена з PBR-освітленням (`simplepbr`), скайбоксом і тінями —
  «вигляд справжнього ігрового двигуна» без Unity/Unreal. FPV-камера — це друга
  камера Panda3D, прив'язана до носа дрона. Все за інтерфейсом `Camera`/`Renderer`,
  тож рушій можна замінити (напр. на Godep-міст), не чіпаючи решту коду.
- **Фізика і рендер — один пакет.** Panda3D постачає Bullet (`panda3d.bullet`),
  тож твердотільна динаміка й візуалізація живуть в одній системі координат.
  Для headless-навчання ML вікно ОС не відкривається (`window-type offscreen` — контекст
  рендеру є, вікна немає), а не `window-type none`, який контексту не створює взагалі.
- **Апарат:** усі транспорти реалізують один інтерфейс `Vehicle`, тож FPV,
  великий мультиротор і fixed-wing додаються без змін у ядрі.
- **Джерело керування:** `KeyboardSource`, `GamepadSource`, `RLAgentSource`,
  `ScriptedSource` — усі дають `ControlCommand`.

## Потік ML

1. **Дані:** `ml/data` літає скриптованими траєкторіями, зберігає кадри камери +
   автоматичні bbox (координати цілей відомі із симуляції) → датасет YOLO.
2. **Детекція:** `ml/detection` навчає YOLO на цьому датасеті → `Detector`.
3. **Наведення:** `ml/rl` — `DroneEnv` дає агенту спостереження (стан + детекції),
   агент видає `ControlCommand`; винагорода за зближення/влучання по віртуальній цілі.
4. **Інтеграція:** `RLAgentSource` під'єднує навченого агента до звичайного
   `SimLoop` як ще одне джерело керування.

Дотримуйся меж модулів: ML споживає `CameraFrame`/`VehicleState` і повертає
`ControlCommand` — тими самими контрактами, що й людина.
