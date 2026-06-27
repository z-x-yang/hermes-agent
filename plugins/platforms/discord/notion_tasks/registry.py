"""Module-level active-controller registry.

Lives in its own module so buttons.py and controller.py can both import it
without an import cycle: buttons needs the getter for its DynamicItem
callback (rebuilt from custom_id after a restart, with no view instance to
reach), and the adapter sets the controller at startup.
"""
from __future__ import annotations

_active = None


def set_active_controller(controller) -> None:
    global _active
    _active = controller


def get_active_controller():
    return _active
