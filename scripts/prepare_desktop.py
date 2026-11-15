"""Prepare the missing Ubuntu/Debian Qt library locally, without sudo."""

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not all(shutil.which(tool) for tool in ("apt-get", "dpkg-deb")):
        raise SystemExit(
            "Установите libxcb-cursor0 средствами вашей ОС; эта подготовка для Debian/Ubuntu."
        )
    destination = ROOT / "runtime/qt"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sirius-qt-") as temporary:
        subprocess.run(["apt-get", "download", "libxcb-cursor0"], cwd=temporary, check=True)
        for package in Path(temporary).glob("*.deb"):
            subprocess.run(["dpkg-deb", "-x", str(package), str(destination)], check=True)
    print("Зависимость Qt подготовлена локально. Запуск: uv run sirius-proj")


if __name__ == "__main__":
    main()
