"""One place for runtime defaults and input configuration."""

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
    context_size: int = 8192
    output_tokens: int = 1024

    def validate(self) -> None:
        if not self.server_binary.is_file():
            raise ValueError(
                "Не найден llama-server. Выполните scripts/prepare_models.py или укажите путь."
            )
        if not self.model_file.is_file():
            raise ValueError(
                "Не найдена GGUF-модель. Выполните scripts/prepare_models.py или укажите путь."
            )
        for name in ("model.onnx", "tokenizer.json"):
            if not (self.embedding_dir / name).is_file():
                raise ValueError(f"Не найден {name} в каталоге E5")
        if min(self.request_timeout, self.startup_timeout) <= 0:
            raise ValueError("Тайм-аут должен быть положительным")
        if self.context_size < 2048 or not 1 <= self.output_tokens < self.context_size:
            raise ValueError("Некорректный размер контекста или ответа")


def default_settings(root: Path) -> Settings:
    candidates = sorted((root / "runtime" / "llama").glob("**/llama-server"))
    server = candidates[0] if candidates else root / "runtime" / "llama-server"
    model = root / "models/qwen7b/model-00001-of-00002.gguf"
    return Settings(
        server.resolve(),
        model.resolve(),
        (root / "models/e5").resolve(),
    )
