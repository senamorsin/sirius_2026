"""Подготовка моделей и других компонентов. Документы и чек-листы сюда не передаются."""

import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from sirius_proj.config import Settings, default_settings

BUILD = "b12420"


@contextmanager
def component_lock(root: Path):
    """После сбоя система снимает блокировку сама. Файл блокировки можно оставить."""
    with (root / ".installation.lock").open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            acquire = partial(msvcrt.locking, stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = partial(msvcrt.locking, stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            acquire = partial(fcntl.flock, stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = partial(fcntl.flock, stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError("Подготовка компонентов уже выполняется другим окном") from error
        try:
            yield
        finally:
            release()


@dataclass(frozen=True)
class Artifact:
    path: str
    url: str
    sha256: str


E5 = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3"
QWEN = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f"
MODELS = (
    Artifact(
        "models/e5/model.onnx",
        E5 + "/onnx/model.onnx",
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
    Artifact(
        "models/e5/tokenizer.json",
        E5 + "/tokenizer.json",
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    Artifact(
        "models/qwen7b/model-00001-of-00002.gguf",
        QWEN + "/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
    ),
    Artifact(
        "models/qwen7b/model-00002-of-00002.gguf",
        QWEN + "/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a",
    ),
)
RUNTIMES = {
    ("linux", "x86_64"): (
        "ubuntu-vulkan-x64.tar.gz",
        "8e6246482dff0df16a467bb2648e53ac50e5b32aa190f97a695794be04936b29",
    ),
    ("darwin", "x86_64"): (
        "macos-x64.tar.gz",
        "8bfd2a32644feae37a7b03daae5bf253b6bde626823b1fbc6e575e7889a6b3ef",
    ),
    ("darwin", "arm64"): (
        "macos-arm64.tar.gz",
        "1b9e418c21f9a32488de140d447bc4d791a09303b56c584102d9e855dbd85e4b",
    ),
    ("win32", "x86_64"): (
        "win-vulkan-x64.zip",
        "d2684cf095adb6824cba9c3ffcfb0348801b0ed84b1a31241d77309458d86b8e",
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(
    artifact: Artifact, root: Path, progress: Callable[[str], None], cancelled: Callable[[], bool]
) -> Path:
    destination = root / artifact.path
    if cancelled():
        raise InterruptedError("Операция отменена")
    if destination.is_file() and digest(destination) == artifact.sha256:
        progress(f"Готово: {artifact.path}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    progress(f"Загрузка {artifact.path}…")
    try:
        request = urllib.request.Request(artifact.url, headers={"User-Agent": "sirius-2026"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            total = 0
            while block := source.read(1024 * 1024):
                if cancelled():
                    raise InterruptedError("Операция отменена")
                target.write(block)
                total += len(block)
                if total % (16 * 1024 * 1024) == 0:
                    progress(f"{destination.name}: загружено {total // (1024 * 1024)} МиБ")
        if digest(temporary) != artifact.sha256:
            raise ValueError(f"Не совпала контрольная сумма {destination.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def runtime_artifact() -> Artifact:
    machine = platform.machine().casefold()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    selection = RUNTIMES.get((sys.platform, machine))
    if selection is None:
        raise RuntimeError(f"Нет подготовленной сборки llama.cpp для {sys.platform}/{machine}")
    suffix, checksum = selection
    name = f"llama-{BUILD}-bin-{suffix}"
    return Artifact(
        "runtime/" + name,
        f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/{name}",
        checksum,
    )


def _runtime_member(name: str) -> bool:
    name = Path(name).name
    return name in ("llama-server", "llama-server.exe", "LICENSE") or any(
        extension in name for extension in (".so", ".dll", ".dylib")
    )


def extract_runtime(archive: Path, destination: Path, name: str):
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if (
                    not (destination / member.filename)
                    .resolve()
                    .is_relative_to(destination.resolve())
                ):
                    raise ValueError("Небезопасный путь внутри архива runtime")
                if _runtime_member(member.filename):
                    package.extract(member, destination)
    else:
        with tarfile.open(archive) as package:
            members = [
                member for member in package if member.isdir() or _runtime_member(member.name)
            ]
            package.extractall(destination, members=members, filter="data")
    (destination / "manifest.json").write_text(json.dumps({"archive": name}), encoding="utf-8")
    archive.unlink()


def load_xcb_cursor(root: Path) -> bool:
    for library in [
        "libxcb-cursor.so.0",
        *map(str, (root / "runtime/qt").glob("usr/lib/**/libxcb-cursor.so.0")),
    ]:
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False


def prepare_desktop(root: Path, progress: Callable[[str], None] = print) -> None:
    if sys.platform != "linux" or load_xcb_cursor(root):
        return
    progress("Подготовка компонента Qt для рабочего стола…")
    destination = root / "runtime/qt"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        _install_cursor(destination)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Установите системную libxcb-cursor0. Для автоматической установки нужны "
            "apt-get и dpkg-deb."
        ) from error
    except subprocess.SubprocessError as error:
        raise RuntimeError(
            "Не удалось подготовить libxcb-cursor0. Проверьте доступ к репозиториям "
            "Debian/Ubuntu и повторите sirius-proj install."
        ) from error
    if not load_xcb_cursor(root):
        raise RuntimeError("Qt не может загрузить libxcb-cursor.so.0 после установки")


def _install_cursor(destination: Path):
    with tempfile.TemporaryDirectory(prefix="sirius-desktop-") as directory:
        subprocess.run(
            ["apt-get", "download", "libxcb-cursor0"],
            cwd=directory,
            check=True,
            capture_output=True,
        )
        for package in Path(directory).glob("*.deb"):
            subprocess.run(
                ["dpkg-deb", "-x", str(package), str(destination)], check=True, capture_output=True
            )


def install_components(
    root: Path,
    progress: Callable[[str], None] = print,
    cancelled: Callable[[], bool] = lambda: False,
) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    with component_lock(root):
        return _install_components(root, progress, cancelled)


def _install_components(root, progress, cancelled) -> Settings:
    if cancelled():
        raise InterruptedError("Операция отменена")
    prepare_desktop(root, progress)
    for artifact in MODELS:
        download(artifact, root, progress, cancelled)
    if cancelled():
        raise InterruptedError("Операция отменена")
    artifact = runtime_artifact()
    runtime = root / "runtime/llama"
    marker = runtime / "manifest.json"
    settings = default_settings(root)
    try:
        installed = json.loads(marker.read_text()).get("archive")
    except OSError, ValueError:
        installed = None
    ready = settings.server_binary.is_file() and installed == Path(artifact.path).name
    if not ready:
        archive = download(artifact, root, progress, cancelled)
        extract_runtime(archive, runtime, archive.name)
    settings = default_settings(root)
    if missing := settings.missing_components:
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(f"После установки отсутствуют файлы: {names}")
    progress("Все компоненты подготовлены. Можно запускать проверку.")
    return settings
