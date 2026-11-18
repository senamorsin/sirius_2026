"""Настройки приложения: пути к компонентам, режим работы и ограничения."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    server_binary: Path
    model_file: Path
    embedding_dir: Path
    use_gpu: bool = True
    request_timeout: float = 120.0
    startup_timeout: float = 180.0
    context_size: int = 16384
    output_tokens: int = 1024
    data_dir: Path | None = None
    force_rag: bool = False

    @property
    def missing_components(self) -> tuple[Path, ...]:
        paths = [
            self.server_binary,
            self.model_file,
            self.embedding_dir / "model.onnx",
            self.embedding_dir / "tokenizer.json",
        ]
        if "00001-of-00002" in self.model_file.name:
            paths.append(
                self.model_file.with_name(
                    self.model_file.name.replace("00001-of-00002", "00002-of-00002")
                )
            )
        return tuple(path for path in paths if not path.is_file())

    def validate(self) -> None:
        if not self.server_binary.is_file():
            raise ValueError(
                "Не найден llama-server. Подготовьте компоненты в окне приложения "
                "или выполните sirius-proj install."
            )
        if not self.model_file.is_file():
            raise ValueError(
                "Не найдена GGUF-модель. Подготовьте компоненты в окне приложения "
                "или выполните sirius-proj install."
            )
        for name in ("model.onnx", "tokenizer.json"):
            if not (self.embedding_dir / name).is_file():
                raise ValueError(f"Не найден {name} в каталоге E5")
        if self.missing_components:
            raise ValueError(f"Не подготовлен компонент: {self.missing_components[0].name}")
        if min(self.request_timeout, self.startup_timeout) <= 0:
            raise ValueError("Тайм-аут должен быть положительным")
        if self.context_size < 2048 or not 1 <= self.output_tokens < self.context_size:
            raise ValueError("Некорректный размер контекста или ответа")


def default_data_dir() -> Path:
    if os.environ.get("SIRIUS_DATA_DIR"):
        return Path(os.environ["SIRIUS_DATA_DIR"]).expanduser().resolve()
    source = Path(__file__).resolve().parents[2]
    if not getattr(sys, "frozen", False) and (source / "pyproject.toml").is_file():
        return source
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "SiriusReview"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/SiriusReview"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "sirius-review"


def default_settings(root: Path | None = None) -> Settings:
    root = (root or default_data_dir()).resolve()
    executable = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates = sorted((root / "runtime" / "llama").glob("**/" + executable))
    server = candidates[0] if candidates else root / "runtime" / executable
    model = root / "models/qwen7b/model-00001-of-00002.gguf"
    return Settings(
        server.resolve(),
        model.resolve(),
        (root / "models/e5").resolve(),
        data_dir=root,
    )
