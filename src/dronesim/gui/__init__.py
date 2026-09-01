"""GUI-лаунчер (доп. фаза "справжній Liftoff") — графічний вхід замість CLI.

Окремий процес: цей пакет НІКОЛИ не імпортує Panda3D/torch напряму, лише
запускає ``dronesim.app`` через subprocess (``gui/process.py``) — тож усі
DLL-порядкові пастки (docs/DECISIONS.md, torch-до-Panda3D) просто не
стосуються GUI-процесу.
"""
