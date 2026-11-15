"""XLSX adapter: safe text cells, all evidence, atomic destination replacement."""

import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from sirius_proj.domain.models import CheckFailure, CheckReport, CheckResult


def write_report(path: Path, report: CheckReport) -> None:
    warnings = report.document.warnings
    if report.cancelled:
        warnings += ("Проверка отменена: экспортированы только завершённые пункты",)
    export_xlsx(path, report.results, report.document.path, warnings)


def safe_text(value: str) -> str:
    value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", value)
    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        value = "'" + value
    return value


def export_xlsx(
    path: Path,
    results: Sequence[CheckResult],
    document: Path,
    warnings: Sequence[str],
) -> None:
    book = Workbook()
    summary = book.active
    summary.title = "Результаты"
    summary.append(
        ["ID", "Требование", "Статус", "Пояснение", "Цитаты", "Места", "Техническая ошибка"]
    )
    evidence_sheet = book.create_sheet("Доказательства")
    evidence_sheet.append(["ID требования", "ID фрагмента", "Цитата", "Место в документе"])
    for result in results:
        _append_result(summary, evidence_sheet, result)
    info = book.create_sheet("Сведения")
    info.append(["Параметр", "Значение"])
    info.append(["Документ", safe_text(str(document))])
    info.append(["Создано", datetime.now().astimezone().isoformat(timespec="seconds")])
    for warning in warnings:
        info.append(["Предупреждение", safe_text(warning)])
    info.append(
        ["Ограничение", "Проверяется доступный текст. Итоговое решение принимает специалист."]
    )
    for sheet in book:
        _format_sheet(sheet)
    _save_atomic(book, path)


def _append_result(summary, evidence_sheet, result: CheckResult) -> None:
    req = result.requirement
    if isinstance(result, CheckFailure):
        summary.append(
            [safe_text(req.id), safe_text(req.text), "", "", "", "", safe_text(result.message)]
        )
        return
    summary.append(
        [
            safe_text(req.id),
            safe_text(req.text),
            result.status.label,
            safe_text(result.explanation),
            safe_text("\n\n".join(item.quote for item in result.evidence)),
            safe_text("\n".join(item.location.label for item in result.evidence)),
            "",
        ]
    )
    for item in result.evidence:
        evidence_sheet.append(
            [
                safe_text(req.id),
                safe_text(item.chunk_id),
                safe_text(item.quote),
                safe_text(item.location.label),
            ]
        )


def _format_sheet(sheet) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="183B56")
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = 18 if letter == "A" else 55
        for cell in column:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _save_atomic(book: Workbook, path: Path) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as stream:
            temporary = Path(stream.name)
        book.save(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
