"""Поиск по смыслу, словам и обозначениям. Векторы документа считаем один раз за проверку."""

import re
from collections import Counter
from collections.abc import Sequence
from math import log1p
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool = False) -> NDArray[np.float32]: ...


IDENTIFIER = re.compile(
    r"(?<!\w)(?:[\w]+[-./][\w./-]*\d[\w./-]*|\d+(?:\.\d+)+|[A-Za-zА-Яа-яЁё]+\d+[\w-]*)(?!\w)"
)
STOP_WORDS = frozenset(
    "в и на к по из с а о об для что это или не должна должен должно должны "
    "быть содержать записке документе отчёте отчете требование".split()
)
RUSSIAN_WORD = re.compile(r"[а-яё]+")


def words(text: str) -> Counter[str]:
    # Начало длинного русского слова помогает сопоставить разные падежи.
    # Точные технические обозначения отдельно сравнивает identifiers().
    return Counter(
        word[:7] if RUSSIAN_WORD.fullmatch(word) else word
        for word in re.findall(r"\w+", text.casefold())
        if word not in STOP_WORDS
    )


def lexical_score(query: Counter, counts: Counter, idf: dict[str, float], average: float) -> float:
    length = sum(counts.values())
    normalization = 1.2 * (0.25 + 0.75 * length / average)
    return sum(
        idf.get(word, 0.0) * counts[word] * 2.2 / (counts[word] + normalization)
        for word in query
        if counts[word]
    )


def contents_penalty(text: str, query: Counter) -> float:
    if "оглавление" in query or "содержание" in query:
        return 0.0
    return 0.15 if re.search(r"\.{3,}\s*\d+\s*$", text) else 0.0


def identifiers(text: str) -> frozenset[str]:
    return frozenset(match.group().casefold() for match in IDENTIFIER.finditer(text))


def lexical_statistics(word_counts: tuple[Counter, ...]) -> tuple[dict[str, float], float]:
    frequencies = Counter(word for counts in word_counts for word in counts)
    size = len(word_counts)
    idf = {word: log1p((size - count + 0.5) / (count + 0.5)) for word, count in frequencies.items()}
    average = max(sum(sum(counts.values()) for counts in word_counts) / size, 1)
    return idf, average


class HybridIndex:
    def __init__(self, texts: tuple[str, ...], encoder: Encoder) -> None:
        self.texts = texts
        self.encoder = encoder
        self.vectors = encoder.encode(texts)
        self.terms = tuple(identifiers(text) for text in texts)
        frequencies = Counter(term for terms in self.terms for term in terms)
        # Редкое обозначение помогает найти нужный фрагмент. Частое имя объекта
        # не должно вытеснять полезный абзац, в котором это имя не повторили.
        size = len(texts)
        self.identifier_weights = {
            term: 3 * (log1p((size - count + 0.5) / (count + 0.5)) / log1p(size)) ** 3
            for term, count in frequencies.items()
        }
        self.word_counts = tuple(words(text) for text in texts)
        self.idf, self.average_length = lexical_statistics(self.word_counts)

    def rank(self, requirement: str) -> list[int]:
        vector = self.encoder.encode([requirement], query=True)[0]
        similarities = self.vectors @ vector
        terms = identifiers(requirement)
        query = words(requirement)
        scores = [
            self._score(i, float(similarity), query)
            + sum(self.identifier_weights[term] for term in terms & self.terms[i])
            for i, similarity in enumerate(similarities)
        ]
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

    def _score(self, index: int, semantic: float, query: Counter) -> float:
        lexical = lexical_score(query, self.word_counts[index], self.idf, self.average_length)
        return (
            semantic + 0.35 * lexical / (1 + lexical) - contents_penalty(self.texts[index], query)
        )
