"""Immutable domain values. No GUI, filesystem or model dependencies."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class Status(StrEnum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT = "insufficient"

    @property
    def label(self) -> str:
        return {
            Status.COMPLIANT: "Соответствует",
            Status.NON_COMPLIANT: "Не соответствует",
            Status.INSUFFICIENT: "Недостаточно данных",
        }[self]


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: требуется непустой текст")


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str

    def __post_init__(self) -> None:
        require_text(self.id, "Идентификатор")
        require_text(self.text, "Требование")


@dataclass(frozen=True)
class Location:
    kind: Literal["pdf", "docx"]
    page: int | None = None
    heading: str = ""
    paragraph: int | None = None
    table: int | None = None
    row: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "pdf":
            valid = self.page is not None and self.page > 0
            valid = valid and self.paragraph is None and self.table is None and self.row is None
        elif self.kind == "docx":
            paragraph = self.paragraph is not None and self.paragraph > 0
            table = self.table is not None and self.row is not None
            valid = (paragraph != table) and self.page is None
            if table:
                valid = valid and self.table > 0 and self.row > 0
            if paragraph:
                valid = valid and self.table is None and self.row is None
        else:
            valid = False
        if not valid:
            raise ValueError("Некорректный адрес источника")

    @property
    def label(self) -> str:
        if self.kind == "pdf":
            return f"Страница {self.page}"
        prefix = f"Раздел «{self.heading}», " if self.heading else ""
        if self.table is not None:
            return f"{prefix}таблица {self.table}, строка {self.row}"
        return f"{prefix}абзац {self.paragraph}"


@dataclass(frozen=True)
class Block:
    id: str
    text: str
    location: Location

    def __post_init__(self) -> None:
        require_text(self.id, "Фрагмент")
        require_text(self.text, "Текст фрагмента")


@dataclass(frozen=True)
class DocumentText:
    path: Path
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str

    def __post_init__(self) -> None:
        require_text(self.chunk_id, "Идентификатор цитаты")
        require_text(self.quote, "Цитата")


@dataclass(frozen=True)
class ModelAnswer:
    status: Status
    explanation: str
    evidence: tuple[Citation, ...]


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    quote: str
    location: Location


@dataclass(frozen=True)
class Finding:
    requirement: Requirement
    status: Status
    explanation: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        require_text(self.explanation, "Пояснение")
        if not isinstance(self.status, Status):
            raise ValueError("Неизвестный статус")
        if self.status is not Status.INSUFFICIENT and not self.evidence:
            raise ValueError("Определённый вывод требует проверенных доказательств")


@dataclass(frozen=True)
class CheckFailure:
    requirement: Requirement
    message: str


type CheckResult = Finding | CheckFailure


@dataclass(frozen=True)
class CheckReport:
    document: DocumentText
    results: tuple[CheckResult, ...]
    seconds: float
    cancelled: bool = False
