"""Головне вікно GUI-лаунчера (доп. фаза "GUI-лаунчер"): вкладки Політ,
Автономний, Калібрування, Моделі. Кожна вкладка лише збирає стан форми й
делегує чистим функціям (``gui/scan.py``, ``gui/process.py``,
``gui/calibration_store.py``) — сам віджет нічого не рахує (SKILL.md
правило "рендер/UI не рахує", тут — GUI не рахує).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dronesim.gui import calibration_store, process, rate_profile_store, scan


def _seed_row() -> tuple[QHBoxLayout, QCheckBox, QSpinBox]:
    layout = QHBoxLayout()
    random_seed = QCheckBox("Випадковий")
    random_seed.setChecked(True)
    seed_spin = QSpinBox()
    seed_spin.setRange(0, 2_147_483_647)
    seed_spin.setEnabled(False)
    random_seed.toggled.connect(lambda checked: seed_spin.setEnabled(not checked))
    layout.addWidget(random_seed)
    layout.addWidget(seed_spin)
    return layout, random_seed, seed_spin


class FlyTab(QWidget):
    """Ручний політ: апарат, ввід, сценарій+рівень, seed, вітер, камера."""

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)

        self.vehicle = QComboBox()
        self.vehicle.addItems(scan.list_vehicles())
        form.addRow("Апарат:", self.vehicle)

        self.input_source = QComboBox()
        self.input_source.addItems(["keyboard", "gamepad"])
        form.addRow("Ввід:", self.input_source)

        self.scenario = QComboBox()
        self.scenario.addItems(["", *scan.list_scenarios()])
        self.scenario.currentTextChanged.connect(self._on_scenario_changed)
        form.addRow("Сценарій:", self.scenario)

        self.level = QComboBox()
        form.addRow("Рівень:", self.level)

        seed_layout, self._random_seed, self._seed_spin = _seed_row()
        form.addRow("Seed:", seed_layout)

        self.wind = QDoubleSpinBox()
        self.wind.setRange(0.0, 50.0)
        self.wind.setSuffix(" Н")
        form.addRow("Вітер:", self.wind)

        cam_layout = QHBoxLayout()
        self.cam_width = QSpinBox()
        self.cam_width.setRange(32, 1920)
        self.cam_width.setValue(320)
        self.cam_height = QSpinBox()
        self.cam_height.setRange(32, 1080)
        self.cam_height.setValue(240)
        cam_layout.addWidget(self.cam_width)
        cam_layout.addWidget(QLabel("x"))
        cam_layout.addWidget(self.cam_height)
        form.addRow("Камера ML (WxH):", cam_layout)

        self.launch_button = QPushButton("Запустити політ")
        self.launch_button.clicked.connect(self._launch)
        form.addRow(self.launch_button)

        self._on_scenario_changed(self.scenario.currentText())

    def _on_scenario_changed(self, scenario: str) -> None:
        self.level.clear()
        if scenario:
            self.level.addItems(scan.list_scenario_levels(scenario))
        self.level.setEnabled(bool(scenario))

    def _launch(self) -> None:
        seed = None if self._random_seed.isChecked() else self._seed_spin.value()
        argv = process.build_fly_argv(
            vehicle=self.vehicle.currentText(),
            input_source=self.input_source.currentText(),
            scenario=self.scenario.currentText() or None,
            level=self.level.currentText() or None,
            seed=seed,
            cam_width=self.cam_width.value(),
            cam_height=self.cam_height.value(),
            wind=self.wind.value(),
        )
        process.launch(argv)


class AutonomousTab(QWidget):
    """Автономний ML-режим: чекпойнт, апарат, сценарій+рівень, seed."""

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)

        self.checkpoint = QComboBox()
        self.checkpoint.setEditable(True)
        for model in scan.list_rl_checkpoints():
            self.checkpoint.addItem(model.display_name, userData=str(model.path))
        form.addRow("Чекпойнт (PPO):", self.checkpoint)

        self.vehicle = QComboBox()
        self.vehicle.addItems(scan.list_vehicles())
        idx = self.vehicle.findText("quad_large")
        if idx >= 0:
            self.vehicle.setCurrentIndex(idx)
        form.addRow("Апарат:", self.vehicle)

        self.scenario = QComboBox()
        self.scenario.addItems(scan.list_scenarios())
        idx = self.scenario.findText("strike_range")
        if idx >= 0:
            self.scenario.setCurrentIndex(idx)
        self.scenario.currentTextChanged.connect(self._on_scenario_changed)
        form.addRow("Сценарій:", self.scenario)

        self.level = QComboBox()
        form.addRow("Рівень:", self.level)

        seed_layout, self._random_seed, self._seed_spin = _seed_row()
        form.addRow("Seed:", seed_layout)

        self.launch_button = QPushButton("Запустити автономний режим")
        self.launch_button.clicked.connect(self._launch)
        form.addRow(self.launch_button)

        self._on_scenario_changed(self.scenario.currentText())

    def _on_scenario_changed(self, scenario: str) -> None:
        self.level.clear()
        if scenario:
            self.level.addItems(scan.list_scenario_levels(scenario))

    def _launch(self) -> None:
        checkpoint = self.checkpoint.currentData() or self.checkpoint.currentText()
        if not checkpoint:
            QMessageBox.warning(self, "Немає чекпойнта", "Обери або вкажи шлях до навченого PPO (.zip).")
            return
        seed = None if self._random_seed.isChecked() else self._seed_spin.value()
        argv = process.build_autonomous_argv(
            checkpoint=checkpoint,
            scenario=self.scenario.currentText(),
            level=self.level.currentText() or None,
            vehicle=self.vehicle.currentText(),
            seed=seed,
        )
        process.launch(argv)


class CalibrationTab(QWidget):
    """Калібрування вводу (deadzone/expo) + rate-профілю Acro (доп. фаза
    "справжній Liftoff" — реальний фідбек користувача: "вилітаю як ракета" —
    дає можливість самому підлаштувати "різкість" керування без ручного
    редагування YAML)."""

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)

        input_form = QFormLayout()
        self.input_source = QComboBox()
        self.input_source.addItems(["keyboard", "gamepad"])
        self.input_source.currentTextChanged.connect(self._load_input)
        input_form.addRow("Джерело вводу:", self.input_source)

        self.deadzone = QDoubleSpinBox()
        self.deadzone.setRange(0.0, 0.9)
        self.deadzone.setSingleStep(0.01)
        input_form.addRow("Deadzone:", self.deadzone)

        self.expo = QDoubleSpinBox()
        self.expo.setRange(0.0, 1.0)
        self.expo.setSingleStep(0.01)
        input_form.addRow("Expo (вхід, НЕ rate-крива польоту):", self.expo)

        self.save_input_button = QPushButton("Зберегти калібрування вводу")
        self.save_input_button.clicked.connect(self._save_input)
        input_form.addRow(self.save_input_button)
        outer.addLayout(input_form)

        outer.addWidget(QLabel("Rate-профіль Acro (BetaFlight Actual Rates) для апарата:"))

        vehicle_row = QHBoxLayout()
        self.vehicle = QComboBox()
        self.vehicle.addItems(scan.list_vehicles())
        self.vehicle.currentTextChanged.connect(self._load_rate_profile)
        vehicle_row.addWidget(self.vehicle)
        outer.addLayout(vehicle_row)

        self.rate_table = QTableWidget(len(rate_profile_store.AXES), len(rate_profile_store.PARAMS))
        self.rate_table.setHorizontalHeaderLabels(list(rate_profile_store.PARAMS))
        self.rate_table.setVerticalHeaderLabels([a.capitalize() for a in rate_profile_store.AXES])
        outer.addWidget(self.rate_table)

        self.no_rate_profile_label = QLabel(
            "Цей апарат не має rate_profile (літає в Angle/Alt-hold/Pos-hold, не Acro)."
        )
        self.no_rate_profile_label.setVisible(False)
        outer.addWidget(self.no_rate_profile_label)

        self.save_rate_button = QPushButton("Зберегти rate-профіль")
        self.save_rate_button.clicked.connect(self._save_rate_profile)
        outer.addWidget(self.save_rate_button)

        self._load_input(self.input_source.currentText())
        self._load_rate_profile(self.vehicle.currentText())

    def _load_input(self, input_source: str) -> None:
        cfg = calibration_store.load_calibration(input_source)
        self.deadzone.setValue(float(cfg.get("deadzone", 0.0)))
        self.deadzone.setEnabled("deadzone" in cfg)
        self.expo.setValue(float(cfg.get("expo", 0.0)))

    def _save_input(self) -> None:
        updates: dict[str, float] = {"expo": self.expo.value()}
        if self.deadzone.isEnabled():
            updates["deadzone"] = self.deadzone.value()
        calibration_store.save_calibration(self.input_source.currentText(), updates)
        QMessageBox.information(self, "Збережено", "Калібрування вводу записано в configs/input/.")

    def _load_rate_profile(self, vehicle: str) -> None:
        has_profile = bool(vehicle) and rate_profile_store.has_rate_profile(vehicle)
        self.rate_table.setVisible(has_profile)
        self.no_rate_profile_label.setVisible(not has_profile)
        self.save_rate_button.setEnabled(has_profile)
        if not has_profile:
            return
        profile = rate_profile_store.load_rate_profile(vehicle)
        for row, axis in enumerate(rate_profile_store.AXES):
            for col, param in enumerate(rate_profile_store.PARAMS):
                self.rate_table.setItem(row, col, QTableWidgetItem(str(profile[axis][param])))

    def _save_rate_profile(self) -> None:
        vehicle = self.vehicle.currentText()
        profile: dict[str, dict[str, float]] = {}
        for row, axis in enumerate(rate_profile_store.AXES):
            profile[axis] = {}
            for col, param in enumerate(rate_profile_store.PARAMS):
                item = self.rate_table.item(row, col)
                profile[axis][param] = float(item.text())
        rate_profile_store.save_rate_profile(vehicle, profile)
        QMessageBox.information(
            self, "Збережено", f"Rate-профіль {vehicle} записано в configs/vehicles/."
        )


class ModelsTab(QWidget):
    """Перелік навчених моделей на диску (RL-чекпойнти, ваги YOLO)."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("RL-чекпойнти (runs/rl/**/*.zip):"))
        self.rl_table = self._make_table()
        layout.addWidget(self.rl_table)

        layout.addWidget(QLabel("Ваги YOLO (runs/detect/**/weights/*.pt):"))
        self.yolo_table = self._make_table()
        layout.addWidget(self.yolo_table)

        refresh = QPushButton("Оновити")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)

        self.refresh()

    @staticmethod
    def _make_table() -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Шлях", "Розмір (МБ)"])
        table.horizontalHeader().setStretchLastSection(True)
        return table

    @staticmethod
    def _fill(table: QTableWidget, models) -> None:
        table.setRowCount(len(models))
        for row, model in enumerate(models):
            table.setItem(row, 0, QTableWidgetItem(model.display_name))
            table.setItem(row, 1, QTableWidgetItem(f"{model.size_bytes / 1_048_576:.1f}"))

    def refresh(self) -> None:
        self._fill(self.rl_table, scan.list_rl_checkpoints())
        self._fill(self.yolo_table, scan.list_yolo_weights())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("dronesim — лаунчер")
        self.resize(560, 480)

        tabs = QTabWidget()
        tabs.addTab(FlyTab(), "Політ")
        tabs.addTab(AutonomousTab(), "Автономний")
        tabs.addTab(CalibrationTab(), "Калібрування")
        tabs.addTab(ModelsTab(), "Моделі")
        self.setCentralWidget(tabs)
