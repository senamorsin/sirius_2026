"""Чтение шаблона Word и заполнение тех же строк результатами проверки."""

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Self

from docx import Document
from docx.document import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import _Cell

from sirius_proj.adapters.report_files import save_atomic
from sirius_proj.domain.models import CheckFailure, CheckKind, CheckReport, Requirement

RESULT = "результат и обоснование"
CONFIRMATION = "подтверждение"
REQUIRED = {"требование", RESULT, CONFIRMATION}


@dataclass(frozen=True)
class WordChecklist:
    document: WordDocument
    rows: list[tuple[Requirement, dict[str, _Cell]]]

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(requirement for requirement, _ in self.rows)

    @classmethod
    def read(cls, path: Path) -> Self:
        try:
            document = Document(path)
        except (ValueError, OSError, KeyError, PackageNotFoundError) as error:
            raise ValueError(f"Не удалось прочитать чек-лист {path.name}: {error}") from error
        tables = []
        for table in document.tables:
            if not table.rows:
                continue
            header, *body = table.rows
            columns = [""] * header.grid_cols_before + [
                " ".join(cell.text.casefold().split()) for cell in header.cells
            ]
            if REQUIRED.issubset(columns):
                tables.append((columns, body))
        if not tables:
            raise ValueError(
                "В DOCX не найдена таблица с требованиями, результатами и подтверждением"
            )

        rows = []
        for columns, body in tables:
            for row in body:
                cells = dict(zip(columns[row.grid_cols_before :], row.cells, strict=False))
                fields = {
                    name: cell.text.strip()
                    for name, cell in cells.items()
                    if name not in (RESULT, CONFIRMATION)
                }
                if criterion := fields.get("требование"):
                    identifier = fields.get("№ п/п", fields.get("№")) or str(len(rows) + 1)
                    comment = fields.get("комментарий", "")
                    kind = parse_kind(fields.get("тип проверки", fields.get("типа проверки", "")))
                    rows.append((Requirement(identifier, criterion, comment, kind), cells))
        return cls(document, rows)


def parse_kind(value: str) -> CheckKind:
    text = value.casefold().strip()
    for kind in (CheckKind.METADATA, CheckKind.STRUCTURE):
        if text.startswith(kind.value.casefold()):
            return kind
    return CheckKind.CONTENT


def write_docx_report(path: Path, report: CheckReport) -> None:
    if report.checklist_path is None or report.checklist_path.suffix.lower() != ".docx":
        raise ValueError("Для заполнения отчёта DOCX нужен исходный чек-лист DOCX")
    checklist = WordChecklist.read(report.checklist_path)
    checked = tuple(result.requirement for result in report.results)
    if checked != checklist.requirements[: len(checked)]:
        raise ValueError("Шаблон изменился после проверки. Запустите проверку заново.")
    for (_, cells), result in zip_longest(checklist.rows, report.results):
        state = evidence = ""
        if isinstance(result, CheckFailure):
            state = f"Техническая ошибка\n{result.message}"
        elif result is not None:
            state = f"{result.status.label}\n{result.explanation}"
            evidence = "\n".join(
                f"{item.location.label}\n{item.quote}"
                if item.location.kind == "layout"
                else f"{item.location.label}\n«{item.quote}»"
                for item in result.evidence
            )
        for column, text in ((RESULT, state), (CONFIRMATION, evidence)):
            if column in cells:
                cells[column].text = text
    if report.cancelled:
        checklist.document.add_paragraph("Проверка отменена. Заполнены только завершённые пункты.")
    save_atomic(path, checklist.document.save)
