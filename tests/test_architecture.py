"""Проверки зависимостей между частями приложения."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "layer,forbidden",
    [
        ("domain", ("PySide6", "httpx", "numpy", "json", "sirius_proj.adapters", "sirius_proj.ui")),
        ("application", ("PySide6", "httpx", "numpy", "sirius_proj.adapters", "sirius_proj.ui")),
        ("ui", ("sirius_proj.adapters", "sirius_proj.bootstrap")),
    ],
)
def test_dependencies_point_towards_application_policy(layer, forbidden):
    for path in (Path("src/sirius_proj") / layer).glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert not any(name.startswith(forbidden) for name in modules), str(path)
