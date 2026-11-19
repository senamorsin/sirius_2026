"""Данные требований и результатов. Здесь нет работы с окном, файлами или моделями."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class Status(StrEnum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT = "insufficient"

    @property
    def label(self) -> str:
        return {
            Status.COMPLIANT: "Соответствует",
            Status.PARTIAL: "Частично соответствует",
            Status.NON_COMPLIANT: "Не соответствует",
            Status.INSUFFICIENT: "Недостаточно данных",
        }[self]


class CheckKind(StrEnum):
    CONTENT = "RAG"
    METADATA = "Метаданные"
    STRUCTURE = "Структура/оглавление"


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    comment: str = ""
    kind: CheckKind = CheckKind.CONTENT


@dataclass(frozen=True)
class Location:
    kind: Literal["pdf", "docx", "layout"]
    page: int | None = None
    heading: str = ""
    paragraph: int | None = None
    table: int | None = None
    row: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "layout":
            return self.heading
        if self.kind == "pdf":
            return f"Страница {self.page}"
        prefix = f"Раздел «{self.heading}», " if self.heading else ""
        if self.table is not None:
            return f"{prefix}таблица {self.table}, строка {self.row}"
        return f"{prefix}абзац {self.paragraph}"


@dataclass(frozen=True)
class TableRow:
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    caption: str = ""


@dataclass(frozen=True)
class Block:
    id: str
    text: str
    location: Location
    table: TableRow | None = None


@dataclass(frozen=True)
class DocxLayout:
    page_sizes: tuple[tuple[float, float] | None, ...]
    headings: tuple[Block, ...]
    table_numbers: tuple[str | None, ...]
    figure_numbers: tuple[str | None, ...]


@dataclass(frozen=True)
class DocumentText:
    path: Path
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...] = ()
    layout: DocxLayout | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str


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
    timings: tuple[tuple[str, float], ...] = ()
    checklist_path: Path | None = None
    context_mode: str = ""
