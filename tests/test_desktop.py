"""Проверки системных библиотек Qt и запуска настоящего окна X11."""

import os
import subprocess
import sys

import pytest

from sirius_proj.ui.startup import prepare_qt


def test_missing_xcb_dependency_has_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setattr("sirius_proj.installation.load_xcb_cursor", lambda root: False)

    def unavailable(destination):
        raise FileNotFoundError("apt-get")

    monkeypatch.setattr("sirius_proj.installation._install_cursor", unavailable)
    with pytest.raises(RuntimeError, match="libxcb-cursor0"):
        prepare_qt(tmp_path)


def test_offscreen_tests_do_not_need_xcb(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("sirius_proj.ui.startup.prepare_desktop", lambda root: pytest.fail("xcb"))
    prepare_qt(tmp_path)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Нужна настоящая графическая сессия X11")
def test_cli_opens_a_real_desktop_window():
    code = """
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from sirius_proj.cli import main
original = QApplication.exec
def timed_exec(app):
    def verify():
        windows = app.topLevelWidgets()
        assert app.platformName() == 'xcb', app.platformName()
        assert any(w.isVisible() and w.windowHandle().isExposed() for w in windows)
        app.quit()
    QTimer.singleShot(700, verify)
    return original()
QApplication.exec = timed_exec
sys.argv = ['sirius-proj']
raise SystemExit(main())
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
