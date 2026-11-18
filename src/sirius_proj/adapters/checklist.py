"""Чтение шаблона Word и заполнение тех же строк результатами проверки."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from docx import Document
from docx.document import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import _Row

from sirius_proj.adapters.report_files import save_atomic
from sirius_proj.domain.models import CheckFailure, CheckKind, CheckReport, Requirement

REQUIRED = {"требование", "результат и обоснование", "подтверждение"}


@dataclass(frozen=True)
class WordChecklist:
    document: WordDocument
    rows: dict[str, _Row]
    columns: dict[str, int]
    requirements: tuple[Requirement, ...]

    @classmethod
    def read(cls, path: Path) -> Self:
        try:
            document = Document(path)
        except (ValueError, OSError, KeyError, PackageNotFoundError) as error:
            raise ValueError(f"Не удалось прочитать чек-лист {path.name}: {error}") from error
        candidates = []
        for table in document.tables:
            if table.rows:
                columns = {
                    " ".join(cell.text.casefold().split()): i
                    for i, cell in enumerate(table.rows[0].cells)
                }
                if REQUIRED <= columns.keys():
                    candidates.append((table, columns))
        if len(candidates) != 1:
            raise ValueError(
                "В DOCX нужна одна таблица с требованиями, результатами и подтверждением"
            )
        table, columns = candidates[0]
        numbers = columns.get("№ п/п", columns.get("№"))
        method = columns.get("тип проверки", columns.get("типа проверки"))
        rows, requirements = {}, []
        width = len(table.rows[0].cells)
        for number, row in enumerate(table.rows[1:], 1):
            cells = row.cells
            if len(cells) != width:
                raise ValueError(f"Строка {number}: неверное число ячеек чек-листа")
            criterion = cells[columns["требование"]].text.strip()
            if not criterion and not any(cell.text.strip() for cell in cells):
                continue
            if any(cells[columns[name]].text.strip() for name in REQUIRED - {"требование"}):
                raise ValueError("Чек-лист уже заполнен. Выберите пустой шаблон для проверки.")
            identifier = (cells[numbers].text.strip() if numbers is not None else "") or str(number)
            if identifier in rows:
                raise ValueError(f"Повторяется номер требования: {identifier}")
            comment = cells[columns["комментарий"]].text.strip() if "комментарий" in columns else ""
            kind = parse_kind(cells[method].text) if method is not None else CheckKind.CONTENT
            requirements.append(Requirement(identifier, criterion, comment, kind))
            rows[identifier] = row
        if not requirements:
            raise ValueError("Чек-лист не содержит требований")
        return cls(document, rows, columns, tuple(requirements))


def parse_kind(value: str) -> CheckKind:
    text = value.casefold().strip()
    for kind in (CheckKind.METADATA, CheckKind.STRUCTURE):
        if text.startswith(kind.value.casefold()):
            return kind
    if text in ("", "rag"):
        return CheckKind.CONTENT
    raise ValueError(f"Неизвестный тип проверки: {value}")


def write_docx_report(path: Path, report: CheckReport) -> None:
    if report.checklist_path is None or report.checklist_path.suffix.lower() != ".docx":
        raise ValueError("Для заполнения отчёта DOCX нужен исходный чек-лист DOCX")
    checklist = WordChecklist.read(report.checklist_path)
    originals = {req.id: req for req in checklist.requirements}
    for result in report.results:
        if originals.get(result.requirement.id) != result.requirement:
            raise ValueError("Шаблон изменился после проверки. Запустите проверку заново.")
        cells = checklist.rows[result.requirement.id].cells
        if isinstance(result, CheckFailure):
            state = f"Техническая ошибка\n{result.message}"
            evidence = ""
        else:
            state = f"{result.status.label}\n{result.explanation}"
            evidence = "\n".join(
                f"{item.location.label}\n{item.quote}"
                if item.location.kind == "layout"
                else f"{item.location.label}\n«{item.quote}»"
                for item in result.evidence
            )
        cells[checklist.columns["результат и обоснование"]].text = state
        cells[checklist.columns["подтверждение"]].text = evidence
    if report.cancelled:
        checklist.document.add_paragraph("Проверка отменена. Заполнены только завершённые пункты.")
    save_atomic(path, checklist.document.save)
