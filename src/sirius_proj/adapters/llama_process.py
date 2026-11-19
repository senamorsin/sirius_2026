"""Запуск и остановка сервера llama.cpp. При ошибке также закрываем процесс и соединения."""

import os
import re
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from sirius_proj.adapters.llama_client import LlamaJudge
from sirius_proj.config import Settings


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def device_arguments(binary: Path, use_gpu: bool) -> list[str]:
    if not use_gpu:
        return ["--device", "none", "--no-op-offload"]
    devices = subprocess.run(
        [str(binary), "--list-devices"], capture_output=True, text=True, timeout=15, check=True
    )
    for line in (devices.stdout + devices.stderr).splitlines():
        match = re.search(r"^\s*(\w+):.*NVIDIA", line)
        if match:
            return ["--device", match[1]]
    return []


@contextmanager
def open_llama(
    settings: Settings, cancelled: Callable[[], bool] = lambda: False
) -> Iterator[LlamaJudge]:
    """Запускает сервер на время проверки и закрывает его при выходе."""
    settings.validate()
    port = available_port()
    with (
        tempfile.TemporaryFile(mode="w+b") as log,
        httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=settings.request_timeout
        ) as client,
    ):
        command = _command(settings, port)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")
        }
        environment["LD_LIBRARY_PATH"] = (
            str(settings.server_binary.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
        )
        process = subprocess.Popen(
            command, stdout=log, stderr=log, env=environment, stdin=subprocess.DEVNULL
        )
        try:
            _wait_ready(process, client, log, settings.startup_timeout, cancelled)
            yield LlamaJudge(client, settings.output_tokens)
        finally:
            _stop_process(process)


def _command(settings: Settings, port: int) -> list[str]:
    return [
        str(settings.server_binary),
        "--model",
        str(settings.model_file),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(settings.context_size),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99" if settings.use_gpu else "0",
        "--threads",
        "8",
        "--no-webui",
        "--offline",
        *device_arguments(settings.server_binary, settings.use_gpu),
    ]


def _wait_ready(process, client, log, timeout, cancelled):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancelled():
            raise InterruptedError("Проверка отменена")
        if process.poll() is not None:
            log.seek(0)
            details = log.read().decode("utf-8", errors="replace")[-2500:]
            raise RuntimeError(f"llama.cpp завершился при запуске. {details}")
        try:
            if client.get("/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.15)
    raise TimeoutError("Модель не загрузилась за отведённое время")


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
