"""Проверки запуска и остановки сервера модели, тайм-аутов и выбора GPU."""

from dataclasses import replace

import httpx
import pytest

import sirius_proj.adapters.llama_process as llama
from sirius_proj.config import Settings


@pytest.fixture
def settings(tmp_path):
    server, model, e5 = tmp_path / "server", tmp_path / "model.gguf", tmp_path / "e5"
    server.touch()
    model.touch()
    e5.mkdir()
    for name in ("model.onnx", "tokenizer.json"):
        (e5 / name).touch()
    return Settings(server, model, e5)


class Process:
    terminated = False
    killed = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        return 0


def mock_runtime(monkeypatch, code=200, process=None):
    process = process or Process()
    captured = {}

    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    original = httpx.Client
    monkeypatch.setattr(llama.subprocess, "Popen", start)
    monkeypatch.setattr(llama, "device_arguments", lambda binary, gpu: [])
    monkeypatch.setattr(
        llama.httpx,
        "Client",
        lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(code, json={}))
        ),
    )
    return process, captured


def test_owned_server_is_local_offline_and_terminated(settings, monkeypatch):
    process, captured = mock_runtime(monkeypatch)
    monkeypatch.setenv("LLAMA_ARG_RPC", "remote.invalid:1234")
    runtime = llama.LlamaServer(settings)
    with runtime as judge:
        assert runtime.process is process
        assert judge.client is runtime.client
        assert "127.0.0.1" in captured["command"]
        assert "--offline" in captured["command"]
        assert "LLAMA_ARG_RPC" not in captured["env"]
        assert runtime.client.trust_env is False
    assert process.terminated
    assert runtime.log.closed


def test_model_startup_timeout_cleans_up(settings, monkeypatch):
    process, _ = mock_runtime(monkeypatch, code=503)
    with pytest.raises(TimeoutError):
        with llama.LlamaServer(replace(settings, startup_timeout=0.01)):
            pytest.fail("Сервер ещё не готов к работе")
    assert process.terminated


def test_cancel_during_startup_cleans_up(settings, monkeypatch):
    process, _ = mock_runtime(monkeypatch)
    with pytest.raises(InterruptedError):
        with llama.LlamaServer(settings, lambda: True):
            pytest.fail("Запуск сервера должен быть отменён")
    assert process.terminated


def test_process_crash_explains_startup_failure(settings, monkeypatch):
    class DeadProcess(Process):
        def poll(self):
            return 1

    mock_runtime(monkeypatch, process=DeadProcess())
    with pytest.raises(RuntimeError, match="завершился"):
        with llama.LlamaServer(settings):
            pytest.fail("Процесс сервера уже завершился")


def test_cpu_mode_disables_gpu_operation_offload(settings):
    assert llama.device_arguments(settings.server_binary, False) == [
        "--device",
        "none",
        "--no-op-offload",
    ]


def test_nvidia_is_preferred_over_integrated_gpu(monkeypatch, settings):
    from types import SimpleNamespace

    monkeypatch.setattr(
        llama.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="  Vulkan0: AMD Radeon\n  Vulkan1: NVIDIA GeForce RTX 5060\n", stderr=""
        ),
    )
    assert llama.device_arguments(settings.server_binary, True) == ["--device", "Vulkan1"]


def test_device_detection_failure_closes_resources(settings, monkeypatch):
    mock_runtime(monkeypatch)

    def fail(*args):
        raise RuntimeError("Не удалось определить доступные устройства")

    monkeypatch.setattr(llama, "device_arguments", fail)
    server = llama.LlamaServer(settings)
    with pytest.raises(RuntimeError, match="Не удалось определить"):
        with server:
            pytest.fail("Сервер не должен запускаться после ошибки выбора устройства")
    assert server.log.closed
    assert server.client.is_closed
    assert server.process is None
