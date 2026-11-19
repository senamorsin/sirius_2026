"""Проверки заголовков, ячеек и адресов строк таблиц."""

import json

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from sirius_proj.adapters.documents import read_document
from sirius_proj.adapters.llama_client import document_context
from sirius_proj.domain.models import Block, Location, TableRow


@pytest.mark.parametrize("omit_first", [False, True])
def test_docx_column_names_and_caption_stay_attached_to_values(tmp_path, omit_first):
    document = Document()
    document.add_heading("Шкаф ТК-101", 0)
    document.add_paragraph("Таблица 1 — Переданная документация")
    table = document.add_table(rows=1, cols=3)
    for cell, text in zip(table.rows[0].cells, ["Документ", "Редакция", "Состояние"], strict=True):
        cell.text = text
    row = table.add_row()
    for cell, text in zip(row.cells, ["Руководство по эксплуатации", "2", "Передано"], strict=True):
        cell.text = text
    start = int(omit_first)
    if omit_first:
        row._tr.remove(row.cells[0]._tc)
        omitted = OxmlElement("w:gridBefore")
        omitted.set(qn("w:val"), "1")
        row._tr.get_or_add_trPr().append(omitted)
    path = tmp_path / "report.docx"
    document.save(path)
    parsed = read_document(path)
    block = next(block for block in parsed.blocks if block.id == "t1r2")
    assert block.table.headers == ("Документ", "Редакция", "Состояние")[start:]
    assert block.table.cells == ("Руководство по эксплуатации", "2", "Передано")[start:]
    assert "Переданная документация" in block.table.caption
    assert block.location.row == 2
    context, _ = document_context([block])
    assert context["tables"][0]["columns"] == ["Документ", "Редакция", "Состояние"][start:]
    assert context["tables"][0]["rows"][0]["cells"][0]["cell_id"] == "t1r2:cell1"
    assert context["tables"][0]["rows"][0]["cells"][1 - start] == {
        "cell_id": f"t1r2:cell{2 - start}",
        "column": "Редакция",
        "text": "2",
    }


def test_selected_rows_are_grouped_by_table_and_keep_source_ids():
    rows = tuple(
        Block(
            f"t1r{i}",
            f"Параметр {i} | {i}",
            Location("docx", heading="Шкаф", table=1, row=i),
            TableRow(("Параметр", "Фактическое значение"), (f"Параметр {i}", str(i))),
        )
        for i in (2, 7)
    )
    payload, _ = document_context(rows)
    assert len(payload["tables"]) == 1
    assert [row["cells"][0]["cell_id"] for row in payload["tables"][0]["rows"]] == [
        "t1r2:cell1",
        "t1r7:cell1",
    ]
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
