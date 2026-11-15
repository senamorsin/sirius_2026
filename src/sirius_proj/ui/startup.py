"""Resolve native desktop dependencies before Qt can abort the process."""

import ctypes
import os
import sys
from pathlib import Path


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
    candidates = ["libxcb-cursor.so.0"]
    candidates.extend(
        str(path) for path in (root / "runtime/qt").glob("usr/lib/**/libxcb-cursor.so.0")
    )
    for library in candidates:
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    raise MissingDesktopLibrary(
        "Qt не может загрузить libxcb-cursor.so.0. "
        "Выполните uv run sirius-proj install "
        "или установите системный пакет libxcb-cursor0."
    )
