"""Local E5 ONNX adapter, using masked mean pooling and L2 normalization."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from sirius_proj.domain.models import Block


class E5Encoder:
    def __init__(self, folder: Path) -> None:
        ort.disable_telemetry_events()
        for name in ("model.onnx", "tokenizer.json"):
            if not (folder / name).is_file():
                raise ValueError(
                    f"Нет файла E5 {folder / name}. Выполните scripts/prepare_models.py"
                )
        self.raw_tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.raw_tokenizer.no_truncation()
        self.raw_tokenizer.no_padding()
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        prefix = "query: " if query else "passage: "
        vectors = []
        for start in range(0, len(texts), 16):
            tokens = self.tokenizer.encode_batch(
                [prefix + text for text in texts[start : start + 16]]
            )
            vectors.append(self._encode_batch(tokens))
        return np.concatenate(vectors)

    def _encode_batch(self, tokens) -> np.ndarray:
        feeds = {
            "input_ids": np.array([item.ids for item in tokens], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in tokens], dtype=np.int64),
            "token_type_ids": np.array([item.type_ids for item in tokens], dtype=np.int64),
        }
        names = {item.name for item in self.session.get_inputs()}
        hidden = self.session.run(
            None, {key: value for key, value in feeds.items() if key in names}
        )[0]
        mask = feeds["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.astype(np.float32)

    def split(
        self,
        blocks: Sequence[Block],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> tuple[Block, ...]:
        chunks = []
        for block in blocks:
            if cancelled():
                raise InterruptedError("Проверка отменена")
            offsets = self.raw_tokenizer.encode(block.text, add_special_tokens=False).offsets
            if len(offsets) <= 400:
                chunks.append(block)
                continue
            for number, start in enumerate(range(0, len(offsets), 350), 1):
                end = min(start + 400, len(offsets))
                text = block.text[offsets[start][0] : offsets[end - 1][1]].strip()
                if text:
                    chunks.append(Block(f"{block.id}:c{number}", text, block.location))
                if end == len(offsets):
                    break
        return tuple(chunks)
