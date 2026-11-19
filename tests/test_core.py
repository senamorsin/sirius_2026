"""Проверки статусов, цитат, чтения документов и отчёта Excel."""

import json
from pathlib import Path

import pytest
from docx import Document
from openpyxl import load_workbook

from sirius_proj.adapters.documents import load_checklist, read_document
from sirius_proj.adapters.llama_client import parse_answer
from sirius_proj.adapters.xlsx import export_xlsx
from sirius_proj.domain.evidence import normalize, verify_answer
from sirius_proj.domain.models import Block, Evidence, Finding, Location, Requirement, Status


@pytest.fixture
def block():
    return Block(
        "b1", "Напряжение питания составляет 220 В.\nМощность — 5 кВт.", Location("pdf", page=2)
    )


def answer(status="compliant", quote="Напряжение питания составляет 220 В.", chunk_id="b1"):
    return parse_answer(
        json.dumps(
            {
                "status": status,
                "explanation": "Питание указано.",
                "evidence": [{"chunk_id": chunk_id, "quote": quote}],
            }
        )
    )


def test_grounded_finding_keeps_program_source(block):
    result = verify_answer(Requirement("1", "Питание 220 В"), answer(), {"b1": block})
    assert result.status is Status.COMPLIANT
    assert result.requirement == Requirement("1", "Питание 220 В")
    assert result.evidence[0].chunk_id == "b1"
    assert result.evidence[0].location.page == 2
    assert result.evidence[0].quote in block.text


@pytest.mark.parametrize("status", ["compliant", "non_compliant"])
@pytest.mark.parametrize(
    "quote,chunk_id",
    [("Выдуманный текст", "b1"), ("Напряжение питания составляет 220 В.", "alien")],
)
def test_ungrounded_decision_is_unknown(block, status, quote, chunk_id):
    result = verify_answer(
        Requirement("1", "Питание 220 В"), answer(status, quote, chunk_id), {"b1": block}
    )
    assert result.status is Status.INSUFFICIENT
    assert result.evidence == ()
    assert result.requirement == Requirement("1", "Питание 220 В")
    assert "цитат" in result.explanation.lower() or "фрагмент" in result.explanation.lower()


def test_whitespace_is_normalized_but_words_are_not(block):
    assert normalize("  220\u00a0В.\n  Мощность — 5 кВт.  ") == "220 В. Мощность — 5 кВт."
    result = verify_answer(
        Requirement("1", "Мощность 5 кВт"), answer(quote="220 В. Мощность — 5 кВт."), {"b1": block}
    )
    assert result.status is Status.COMPLIANT
    bad = verify_answer(
        Requirement("1", "Мощность 7 кВт"), answer(quote="Мощность — 7 кВт."), {"b1": block}
    )
    assert bad.status is Status.INSUFFICIENT


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "не JSON",
        '{"status":"unknown"}',
        '{"status":"compliant","explanation":"x","evidence":[]}',
    ],
)
def test_malformed_model_answer_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)


def test_csv_bom_and_duplicate_ids(tmp_path):
    path = tmp_path / "list.csv"
    path.write_text("\ufeffid,requirement\nA,Питание 220 В\n", encoding="utf-8")
    assert load_checklist(path) == (Requirement("A", "Питание 220 В"),)
    path.write_text("id,requirement\nA,Первое\nA,Второе\n", encoding="utf-8")
    assert load_checklist(path) == (Requirement("A", "Первое"), Requirement("A", "Второе"))


def test_docx_retains_order_and_locations(tmp_path):
    doc = Document()
    doc.add_heading("Шкаф ТК-101", 0)
    doc.add_heading("Питание", level=1)
    doc.add_paragraph("Вводный абзац")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Напряжение"
    table.cell(0, 1).text = "220 В"
    doc.add_paragraph("Заключение")
    path = tmp_path / "report.docx"
    doc.save(path)
    parsed = read_document(path)
    assert [b.text for b in parsed.blocks][1:] == [
        "Питание",
        "Вводный абзац",
        "Напряжение | 220 В",
        "Заключение",
    ]
    assert parsed.blocks[3].location.table == 1
    assert parsed.blocks[3].location.row == 1
    assert "Питание" in parsed.blocks[3].location.heading
    assert "ТК-101" in parsed.blocks[3].location.heading
    assert parsed.blocks[4].location.paragraph == 4


def test_export_preserves_evidence_and_neutralizes_formulas(tmp_path):
    req = Requirement("=1+1", '=HYPERLINK("http://example.test")')
    location = Location("pdf", page=2)
    result = Finding(req, Status.COMPLIANT, "Указано", (Evidence("b1", "220 В", location),))
    path = tmp_path / "result.xlsx"
    export_xlsx(path, (result,), Path("report.pdf"), ("Страница 4 без текста",))
    book = load_workbook(path)
    assert book["Результаты"].cell(2, 1).data_type != "f"
    assert book["Результаты"].cell(2, 3).value == "Соответствует"
    assert book["Доказательства"].cell(2, 3).value == "220 В"
    assert "Страница 4" in book["Сведения"].cell(4, 2).value
    export_xlsx(path, (result,), Path("report.pdf"), ())
    assert path.is_file()
