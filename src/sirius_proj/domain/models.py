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
        validators = {"pdf": self._valid_pdf, "docx": self._valid_docx}
        validate = validators.get(self.kind)
        if validate is None or not validate():
            raise ValueError("Некорректный адрес источника")

    def _valid_pdf(self) -> bool:
        return _positive_int(self.page) and all(
            value is None for value in (self.paragraph, self.table, self.row)
        )

    def _valid_docx(self) -> bool:
        if self.page is not None:
            return False
        if self.paragraph is not None:
            return _positive_int(self.paragraph) and self.table is None and self.row is None
        return _positive_int(self.table) and _positive_int(self.row)

    @property
    def label(self) -> str:
        if self.kind == "pdf":
            return f"Страница {self.page}"
        prefix = f"Раздел «{self.heading}», " if self.heading else ""
        if self.table is not None:
            return f"{prefix}таблица {self.table}, строка {self.row}"
        return f"{prefix}абзац {self.paragraph}"


def _positive_int(value: int | None) -> bool:
    return type(value) is int and value > 0


@dataclass(frozen=True)
class TableRow:
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    caption: str = ""

    def __post_init__(self):
        if not self.cells or len(self.headers) != len(self.cells):
            raise ValueError("Заголовки таблицы должны соответствовать ячейкам строки")

    def clipped(self, start: int, end: int) -> TableRow:
        cursor, cells = 0, []
        for cell in self.cells:
            left, right = max(0, start - cursor), min(len(cell), end - cursor)
            cells.append(cell[left:right] if right > left else "")
            cursor += len(cell) + 3  # source cells are separated by ' | '
        return TableRow(self.headers, tuple(cells), self.caption)


@dataclass(frozen=True)
class Block:
    id: str
    text: str
    location: Location
    table: TableRow | None = None

    def __post_init__(self) -> None:
        require_text(self.id, "Фрагмент")
        require_text(self.text, "Текст фрагмента")
        if self.table is not None and self.location.table is None:
            raise ValueError("Ячейки таблицы требуют адреса таблицы и строки")


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
    timings: tuple[tuple[str, float], ...] = ()
