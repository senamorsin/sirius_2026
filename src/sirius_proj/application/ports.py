"""Правила вызова поиска и модели, которые использует проверка документа."""

from collections.abc import Sequence
from typing import Protocol

from sirius_proj.domain.models import Block, ModelAnswer, Requirement


class SearchIndex(Protocol):
    def retrieve(self, requirement: str) -> tuple[Block, ...]: ...


class Judge(Protocol):
    def assess(self, requirement: Requirement, chunks: Sequence[Block]) -> ModelAnswer: ...
