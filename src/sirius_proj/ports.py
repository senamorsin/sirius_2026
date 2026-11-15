"""Only the boundaries used by application policy; no service locator."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Block, ModelAnswer, Requirement


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool = False) -> NDArray[np.float32]: ...


class SearchIndex(Protocol):
    def retrieve(self, requirement: str) -> tuple[Block, ...]: ...


class Judge(Protocol):
    def assess(self, requirement: Requirement, chunks: Sequence[Block]) -> ModelAnswer: ...
