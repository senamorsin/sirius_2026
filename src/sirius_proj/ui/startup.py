"""Проверка системных библиотек перед открытием окна Qt."""

import os
import sys
from pathlib import Path

from sirius_proj.installation import load_xcb_cursor


class MissingDesktopLibrary(RuntimeError):
    pass


def prepare_qt(root: Path) -> None:
    platform = os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0]
    if sys.platform != "linux" or platform not in ("", "xcb"):
        return
    if not platform and os.environ.get("WAYLAND_DISPLAY"):
        return
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Нет графической сессии DISPLAY. Для проверки без окна используйте check."
        )
    if load_xcb_cursor(root):
        return
    raise MissingDesktopLibrary(
        "Qt не может загрузить libxcb-cursor.so.0. "
        "Выполните uv run sirius-proj install "
        "или установите системный пакет libxcb-cursor0."
    )
