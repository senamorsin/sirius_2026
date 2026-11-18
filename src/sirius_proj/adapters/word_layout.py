"""Чтение размеров страниц и подписей объектов в основном тексте Word."""

import re
from collections.abc import Sequence

from docx.document import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from sirius_proj.domain.models import Block, DocxLayout

TABLE = re.compile(r"^таблица\s*(?:№\s*)?(\d+(?:\.\d+)*|[А-ЯA-Z]\.\d+)\b", re.I)
FIGURE = re.compile(r"^(?:рисунок|рис\.)\s*(?:№\s*)?(\d+(?:\.\d+)*|[А-ЯA-Z]\.\d+)\b", re.I)
UNNUMBERED_TABLE = re.compile(r"^таблица\s*[—:–-]", re.I)


def table_caption(text: str) -> bool:
    return bool(TABLE.match(text) or UNNUMBERED_TABLE.match(text))


def _number(text: str, pattern: re.Pattern) -> str | None:
    match = pattern.match(text)
    return match[1] if match else None


def read_layout(
    document: Document, items: Sequence[Paragraph | Table], headings: Sequence[Block]
) -> DocxLayout:
    sizes = tuple(
        (section.page_width.mm, section.page_height.mm)
        if section.page_width and section.page_height
        else None
        for section in document.sections
    )
    tables, figures = [], []
    preceding = ""
    for i, item in enumerate(items):
        if isinstance(item, Table):
            tables.append(_number(preceding, TABLE))
            preceding = ""
            continue
        pictures = item._p.xpath(".//pic:pic | .//w:pict | .//w:drawing[not(.//pic:pic)]")
        if pictures:
            following = next(
                (
                    p.text.strip() if isinstance(p, Paragraph) else ""
                    for p in items[i + 1 :]
                    if isinstance(p, Table) or p.text.strip()
                ),
                "",
            )
            number = _number(item.text.strip(), FIGURE) or _number(following, FIGURE)
            figures.extend([number, *([None] * (len(pictures) - 1))])
        if item.text.strip():
            preceding = item.text.strip()
    return DocxLayout(sizes, tuple(headings), tuple(tables), tuple(figures))
