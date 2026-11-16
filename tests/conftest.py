"""Настройка окружения для тестов интерфейса без видимого окна."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
