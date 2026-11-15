"""File boundaries: validate once and return domain values."""

import csv
import re
from pathlib import Path

from docx import Document
from docx.table import Table
from pypdf import PdfReader

from sirius_proj.domain.models import Block, DocumentText, Location, Requirement


def load_checklist(path: Path) -> tuple[Requirement, ...]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "requirement"} <= set(reader.fieldnames):
            raise ValueError(
                "CSV должен содержать заголовки id,requirement (UTF-8, разделитель — запятая)"
            )
        requirements = []
        seen = set()
        for number, row in enumerate(reader, 2):
            requirement = _parse_row(row, number)
            if requirement.id in seen:
                raise ValueError(f"Строка {number}: повторяется id «{requirement.id}»")
            seen.add(requirement.id)
            requirements.append(requirement)
    if not requirements:
        raise ValueError("Чек-лист не содержит требований")
    return tuple(requirements)


def _parse_row(row: dict, number: int) -> Requirement:
    if None in row or any(value is None for value in row.values()):
        raise ValueError(f"Строка {number}: неверное число колонок CSV")
    try:
        return Requirement(row["id"].strip(), row["requirement"].strip())
    except ValueError as error:
        raise ValueError(f"Строка {number}: {error}") from error


def read_document(path: Path) -> DocumentText:
    if not path.is_file():
        raise ValueError(f"Документ не найден: {path.name}")
    readers = {".docx": _read_docx, ".pdf": _read_pdf}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError("Поддерживаются только DOCX и PDF")
    try:
        document = reader(path)
    except Exception as error:
        raise ValueError(f"Не удалось прочитать {path.name}: {error}") from error
    if not document.blocks:
        raise ValueError(
            "В документе нет доступного текста. Для скана требуется OCR, он не входит в MVP."
        )
    return document


def _read_docx(path: Path) -> DocumentText:
    document = Document(path)
    blocks = []
    headings = {}
    paragraph_number = table_number = 0
    for item in document.iter_inner_content():
        heading = " / ".join(headings.values())
        if isinstance(item, Table):
            table_number += 1
            blocks.extend(_table_blocks(item, table_number, heading))
        else:
            paragraph_number += 1
            text = item.text.strip()
            if not text:
                continue
            headings = _update_headings(headings, item.style.style_id if item.style else "", text)
            heading = " / ".join(headings.values())
            location = Location("docx", heading=heading, paragraph=paragraph_number)
            blocks.append(Block(f"p{paragraph_number}", text, location))
    return DocumentText(path, tuple(blocks))


def _update_headings(headings: dict[int, str], style: str, text: str) -> dict[int, str]:
    match = re.fullmatch(r"Heading(\d+)", style)
    if style == "Title":
        return {0: text}
    if match:
        level = int(match[1])
        return {**{key: value for key, value in headings.items() if key < level}, level: text}
    return headings


def _table_blocks(table: Table, number: int, heading: str) -> list[Block]:
    blocks = []
    for row_number, row in enumerate(table.rows, 1):
        text = " | ".join(cell.text.strip() for cell in row.cells).strip(" |")
        if text:
            location = Location("docx", heading=heading, table=number, row=row_number)
            blocks.append(Block(f"t{number}r{row_number}", text, location))
    return blocks


def _read_pdf(path: Path) -> DocumentText:
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF защищён паролем; сохраните доступную копию")
    blocks, warnings = [], []
    for page_number, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            warnings.append(f"Страница {page_number}: не удалось извлечь текст")
            continue
        if text:
            blocks.append(Block(f"page{page_number}", text, Location("pdf", page=page_number)))
        else:
            warnings.append(f"Страница {page_number} без текста: возможен скан или пустая страница")
    return DocumentText(path, tuple(blocks), tuple(warnings))
