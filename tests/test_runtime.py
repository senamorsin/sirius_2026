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
    with llama.open_llama(settings) as judge:
        assert "127.0.0.1" in captured["command"]
        assert "--offline" in captured["command"]
        assert "LLAMA_ARG_RPC" not in captured["env"]
        assert judge.client.trust_env is False
    assert process.terminated
    assert captured["stdout"].closed
    assert judge.client.is_closed


def test_model_startup_timeout_cleans_up(settings, monkeypatch):
    process, _ = mock_runtime(monkeypatch, code=503)
    with pytest.raises(TimeoutError):
        with llama.open_llama(replace(settings, startup_timeout=0.01)):
            pytest.fail("Сервер ещё не готов к работе")
    assert process.terminated


def test_cancel_during_startup_cleans_up(settings, monkeypatch):
    process, _ = mock_runtime(monkeypatch)
    with pytest.raises(InterruptedError):
        with llama.open_llama(settings, lambda: True):
            pytest.fail("Запуск сервера должен быть отменён")
    assert process.terminated


def test_process_crash_explains_startup_failure(settings, monkeypatch):
    class DeadProcess(Process):
        def poll(self):
            return 1

    mock_runtime(monkeypatch, process=DeadProcess())
    with pytest.raises(RuntimeError, match="завершился"):
        with llama.open_llama(settings):
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
    _, captured = mock_runtime(monkeypatch)
    clients = []
    original = llama.httpx.Client

    def track_client(**kwargs):
        client = original(**kwargs)
        clients.append(client)
        return client

    def fail(*args):
        raise RuntimeError("Не удалось определить доступные устройства")

    monkeypatch.setattr(llama.httpx, "Client", track_client)
    monkeypatch.setattr(llama, "device_arguments", fail)
    with pytest.raises(RuntimeError, match="Не удалось определить"):
        with llama.open_llama(settings):
            pytest.fail("Сервер не должен запускаться после ошибки выбора устройства")
    assert clients[0].is_closed
    assert not captured
