"""Запуск и остановка сервера llama.cpp. При ошибке также закрываем процесс и соединения."""

import os
import re
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
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


class LlamaServer:
    def __init__(self, settings: Settings, cancelled: Callable[[], bool] = lambda: False):
        self.settings = settings
        self.cancelled = cancelled
        self.process = None
        self.client = None
        self.log = None
        self._resources = ExitStack()

    def __enter__(self) -> LlamaJudge:
        self.settings.validate()
        with ExitStack() as resources:
            self.log = resources.enter_context(tempfile.TemporaryFile(mode="w+b"))
            self.client = resources.enter_context(
                httpx.Client(
                    base_url=f"http://127.0.0.1:{available_port()}",
                    trust_env=False,
                    timeout=self.settings.request_timeout,
                )
            )
            command = self._command()
            environment = {
                key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")
            }
            environment["LD_LIBRARY_PATH"] = (
                str(self.settings.server_binary.parent)
                + ":"
                + environment.get("LD_LIBRARY_PATH", "")
            )
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=self.log, env=environment, stdin=subprocess.DEVNULL
            )
            resources.callback(self._stop_process)
            self._wait_ready()
            self._resources = resources.pop_all()
        return LlamaJudge(self.client, self.settings.output_tokens)

    def _command(self) -> list[str]:
        port = self.client.base_url.port
        return [
            str(self.settings.server_binary),
            "--model",
            str(self.settings.model_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(self.settings.context_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            "99" if self.settings.use_gpu else "0",
            "--threads",
            "8",
            "--no-webui",
            "--offline",
            *device_arguments(self.settings.server_binary, self.settings.use_gpu),
        ]

    def _wait_ready(self):
        deadline = time.monotonic() + self.settings.startup_timeout
        while time.monotonic() < deadline:
            if self.cancelled():
                raise InterruptedError("Проверка отменена")
            if self.process.poll() is not None:
                self.log.seek(0)
                details = self.log.read().decode("utf-8", errors="replace")[-2500:]
                raise RuntimeError(f"llama.cpp завершился при запуске. {details}")
            try:
                if self.client.get("/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.15)
        raise TimeoutError("Модель не загрузилась за отведённое время")

    def _stop_process(self):
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def close(self):
        self._resources.close()

    def __exit__(self, *args):
        self.close()
