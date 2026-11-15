import json

from docx import Document

from sirius_proj.adapters.documents import read_document
from sirius_proj.adapters.llama_client import document_context
from sirius_proj.domain.models import Block, Location, TableRow


def test_docx_column_names_and_caption_stay_attached_to_values(tmp_path):
    document = Document()
    document.add_heading("Шкаф ТК-101", 0)
    document.add_paragraph("Таблица 1 — Переданная документация")
    table = document.add_table(rows=1, cols=3)
    for cell, text in zip(table.rows[0].cells, ["Документ", "Редакция", "Состояние"], strict=True):
        cell.text = text
    row = table.add_row()
    for cell, text in zip(row.cells, ["Руководство по эксплуатации", "2", "Передано"], strict=True):
        cell.text = text
    path = tmp_path / "report.docx"
    document.save(path)
    parsed = read_document(path)
    block = next(block for block in parsed.blocks if block.id == "t1r2")
    assert block.table.headers == ("Документ", "Редакция", "Состояние")
    assert block.table.cells == ("Руководство по эксплуатации", "2", "Передано")
    assert "Переданная документация" in block.table.caption
    assert block.location.row == 2
    context = document_context([block])
    assert context["tables"][0]["columns"] == ["Документ", "Редакция", "Состояние"]
    assert context["tables"][0]["rows"][0]["cells"][0]["cell_id"] == "t1r2:cell1"
    assert context["tables"][0]["rows"][0]["cells"][1] == {
        "cell_id": "t1r2:cell2",
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
    payload = document_context(rows)
    assert len(payload["tables"]) == 1
    assert [row["cells"][0]["cell_id"] for row in payload["tables"][0]["rows"]] == [
        "t1r2:cell1",
        "t1r7:cell1",
    ]
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_long_row_sends_only_cells_present_in_selected_fragment():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from sirius_proj.adapters.embeddings import E5Encoder

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    encoder = E5Encoder.__new__(E5Encoder)
    encoder.raw_tokenizer = tokenizer
    value = "измерение " * 500 + "ЗНАЧЕНИЕ_В_КОНЦЕ"
    row = Block(
        "row",
        "Протокол | " + value,
        Location("docx", table=1, row=2),
        TableRow(("Документ", "Содержание"), ("Протокол", value)),
    )
    first = encoder.split([row])[0]
    sent = json.dumps(document_context([first]), ensure_ascii=False)
    assert "ЗНАЧЕНИЕ_В_КОНЦЕ" not in sent
    assert first.table.cells[1] in first.text
    assert first.location == row.location
    assert first.table.headers == row.table.headers
