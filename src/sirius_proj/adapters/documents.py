"""Чтение DOCX, PDF и CSV. Проверяем данные и сохраняем текст вместе с адресами источников."""

import csv
import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from sirius_proj.adapters.checklist import WordChecklist
from sirius_proj.adapters.word_layout import read_layout, table_caption
from sirius_proj.domain.models import Block, DocumentText, Location, Requirement, TableRow


def load_checklist(path: Path) -> tuple[Requirement, ...]:
    if path.suffix.lower() == ".docx":
        return WordChecklist.read(path).requirements
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
    heading_blocks = []
    headings = {}
    paragraph_number = table_number = 0
    preceding = ""
    items = tuple(document.iter_inner_content())
    for item in items:
        heading = " / ".join(headings.values())
        if isinstance(item, Table):
            table_number += 1
            caption = preceding if table_caption(preceding) else ""
            blocks.extend(_table_blocks(item, table_number, heading, caption))
            preceding = ""
        else:
            paragraph_number += 1
            text = item.text.strip()
            if not text:
                continue
            updated = _update_headings(headings, _heading_level(item), text)
            is_heading = updated is not headings
            headings = updated
            heading = " / ".join(headings.values())
            location = Location("docx", heading=heading, paragraph=paragraph_number)
            block = Block(f"p{paragraph_number}", text, location)
            blocks.append(block)
            if is_heading:
                heading_blocks.append(block)
            preceding = text
    return DocumentText(path, tuple(blocks), layout=read_layout(document, items, heading_blocks))


def _heading_level(paragraph: Paragraph) -> int | None:
    direct = paragraph._p.xpath("./w:pPr/w:outlineLvl")
    style = paragraph.style
    outline = direct or (style._element.xpath("./w:pPr/w:outlineLvl") if style else [])
    if outline:
        return outline[0].val + 1 if 0 <= outline[0].val < 9 else None
    if style and style.name == "Title":
        return 0
    match = re.fullmatch(r"Heading\s*(\d+)", style.name if style else "")
    return int(match[1]) if match else None


def _update_headings(headings: dict[int, str], level: int | None, text: str) -> dict[int, str]:
    if level is None:
        return headings
    return {**{key: value for key, value in headings.items() if key < level}, level: text}


def _column_headers(table: Table) -> tuple[str, ...] | None:
    if not table.rows:
        return None
    first = table.rows[0]
    cells = tuple(cell.text.strip() for cell in first.cells)
    explicit = first._tr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}trPr")
    if (
        explicit is not None
        and explicit.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblHeader")
        is not None
    ):
        return cells
    names = {
        "параметр",
        "значение",
        "фактическое значение",
        "документ",
        "редакция",
        "состояние",
        "наименование",
        "обозначение",
        "дата",
        "показатель",
        "результат",
        "проверка",
        "условие",
        "наблюдаемый результат",
        "номер протокола",
        "основание",
        "единица измерения",
        "вариант",
        "предлагаемый состав",
        "преимущества и ограничения",
        "поток",
        "периодичность",
        "функция",
        "предлагаемая реакция",
    }
    return cells if sum(cell.casefold() in names for cell in cells) >= 2 else None


def _table_blocks(table: Table, number: int, heading: str, caption: str) -> list[Block]:
    blocks = []
    headers = _column_headers(table)
    for row_number, row in enumerate(table.rows, 1):
        if row_number == 1 and headers is not None:
            continue
        cells = tuple(cell.text.strip() for cell in row.cells)
        if any(cells):
            columns = (
                headers
                if headers and len(headers) == len(cells)
                else tuple(f"Столбец {i}" for i in range(1, len(cells) + 1))
            )
            location = Location("docx", heading=heading, table=number, row=row_number)
            blocks.append(
                Block(
                    f"t{number}r{row_number}",
                    " | ".join(cells),
                    location,
                    TableRow(columns, cells, caption),
                )
            )
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
