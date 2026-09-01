@echo off
REM Запуск GUI-лаунчера dronesim (доп. фаза "GUI-лаунчер").
REM Якщо використовуєш venv — розкоментуй і поправ шлях:
REM call .venv\Scripts\activate.bat
cd /d "%~dp0"
python -m dronesim.gui.launcher
pause
