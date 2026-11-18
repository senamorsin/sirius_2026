"""Замена отчёта после успешной записи временного файла."""

import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_atomic(path: Path, save: Callable[[Path], None]) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
