"""Чтение шаблона Word и заполнение тех же строк результатами проверки."""

from dataclasses import dataclass
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
    rows: dict[str, dict[str, _Cell]]
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
                header = table.rows[0]
                columns = [""] * header.grid_cols_before + [
                    " ".join(cell.text.casefold().split()) for cell in header.cells
                ]
                if REQUIRED.issubset(columns):
                    candidates.append((table, columns))
        if len(candidates) != 1:
            raise ValueError(
                "В DOCX нужна одна таблица с требованиями, результатами и подтверждением"
            )
        table, columns = candidates[0]
        rows, requirements = {}, []
        for number, row in enumerate(table.rows[1:], 1):
            cells = dict(zip(columns[row.grid_cols_before :], row.cells, strict=False))
            fields = {
                name: cell.text.strip()
                for name, cell in cells.items()
                if name not in (RESULT, CONFIRMATION)
            }
            criterion = fields.get("требование")
            if not criterion:
                continue
            identifier = fields.get("№ п/п", fields.get("№")) or str(number)
            if identifier in rows:
                raise ValueError(f"Повторяется номер требования: {identifier}")
            comment = fields.get("комментарий", "")
            kind = parse_kind(fields.get("тип проверки", fields.get("типа проверки", "")))
            requirements.append(Requirement(identifier, criterion, comment, kind))
            rows[identifier] = cells
        if not requirements:
            raise ValueError("Чек-лист не содержит требований")
        return cls(document, rows, tuple(requirements))


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
    results = {result.requirement: result for result in report.results}
    if results.keys() - set(checklist.requirements):
        raise ValueError("Шаблон изменился после проверки. Запустите проверку заново.")
    for requirement in checklist.requirements:
        result = results.get(requirement)
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
        cells = checklist.rows[requirement.id]
        for column, text in ((RESULT, state), (CONFIRMATION, evidence)):
            if column in cells:
                cells[column].text = text
    if report.cancelled:
        checklist.document.add_paragraph("Проверка отменена. Заполнены только завершённые пункты.")
    save_atomic(path, checklist.document.save)
