"""Hybrid retrieval. The document embedding matrix is built exactly once."""

import re

from .domain import Block
from .ports import Encoder

IDENTIFIER = re.compile(
    r"(?<!\w)(?:[\w]+[-./][\w./-]*\d[\w./-]*|\d+(?:\.\d+)+|[A-Za-zА-Яа-яЁё]+\d+[\w-]*)(?!\w)"
)


def identifiers(text: str) -> frozenset[str]:
    return frozenset(match.group().casefold() for match in IDENTIFIER.finditer(text))


class HybridIndex:
    def __init__(self, chunks: tuple[Block, ...], encoder: Encoder, limit: int = 8) -> None:
        if not chunks or limit < 1:
            raise ValueError("Индекс требует фрагментов и положительного лимита")
        self.chunks = chunks
        self.encoder = encoder
        self.limit = limit
        self.vectors = encoder.encode([chunk.text for chunk in chunks])
        self.terms = tuple(identifiers(chunk.text) for chunk in chunks)

    def retrieve(self, requirement: str) -> tuple[Block, ...]:
        vector = self.encoder.encode([requirement], query=True)[0]
        similarities = self.vectors @ vector
        terms = identifiers(requirement)
        scores = [
            (len(terms & self.terms[i]), float(similarities[i]), -i)
            for i in range(len(self.chunks))
        ]
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return tuple(self.chunks[i] for i in ranked[: self.limit])
