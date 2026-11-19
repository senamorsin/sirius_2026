"""Проверки маршрутизации требований Word, целых таблиц и заполнения шаблона."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm

from sirius_proj.adapters.documents import load_checklist, read_document
from sirius_proj.adapters.retrieval import search_groups
from sirius_proj.application.review import ReviewService
from sirius_proj.bootstrap import write_report
from sirius_proj.domain.models import (
    Block,
    CheckKind,
    Citation,
    DocumentText,
    Location,
    ModelAnswer,
    Requirement,
    Status,
)


def test_word_template_routes_numbered_rows_and_preserves_sources(tmp_path):
    source = tmp_path / "отчёт.docx"
    document = Document()
    document.sections[0].page_width = Mm(210)
    document.sections[0].page_height = Mm(297)
    heading = document.add_heading("Список определений", 1)
    heading_style = heading.style
    document.add_paragraph("НИР – научно-исследовательская работа.")
    document.add_paragraph("Калибровка – сравнение с эталоном.")
    document.add_paragraph("Цель исследования описана в основном тексте.")
    conclusion = document.add_heading("10 Заключение и дальнейшие работы", 1)
    document.add_paragraph("По результатам исследования выбран вариант В.")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Таблица без номера"
    heading_style.style_id = "1"
    heading.style = conclusion.style = heading_style
    document.save(source)

    checklist = tmp_path / "чек-лист.docx"
    template = Document()
    table = template.add_table(rows=1, cols=6)
    data = [
        (
            "№ п/п",
            "Требование",
            "Результат и обоснование",
            "Подтверждение",
            "Комментарий",
            "Тип проверки",
        ),
        (
            "",
            "Файл отчета имеет формат A4",
            "Старый вывод",
            "Старая цитата",
            "Свойства Word",
            "Метаданные",
        ),
        ("", "Отчёт содержит обязательный раздел «Введение»", "", "", "", "Структура/оглавление"),
        ("", "Отчёт содержит обязательный раздел «Заключение»", "", "", "", "Структура/оглавление"),
        (
            "",
            "Отчёт содержит раздел «Список сокращений и определений»",
            "",
            "",
            "",
            "Структура/оглавление",
        ),
        (
            "",
            "Таблицы и рисунки имеют сквозную нумерацию",
            "",
            "",
            "",
            "Структура/оглавление (?) / RAG (?)",
        ),
    ]
    for i, values in enumerate(data):
        row = table.rows[0] if i == 0 else table.add_row()
        for cell, value in zip(row.cells, values, strict=True):
            cell.text = value
    first = table.rows[1]
    first._tr.remove(first.cells[0]._tc)
    omitted = OxmlElement("w:gridBefore")
    omitted.set(qn("w:val"), "1")
    first._tr.get_or_add_trPr().append(omitted)
    template.save(checklist)

    def unused(*args):
        pytest.fail("Программная проверка не должна запускать поиск и модель")

    report = ReviewService(read_document, load_checklist, unused, unused).run(source, checklist)
    assert report.context_mode == "layout"
    assert [item.requirement.id for item in report.results] == ["1", "2", "3", "4", "5"]
    assert report.results[0].requirement.comment == "Свойства Word"
    assert [item.status for item in report.results] == [
        Status.COMPLIANT,
        Status.NON_COMPLIANT,
        Status.COMPLIANT,
        Status.PARTIAL,
        Status.NON_COMPLIANT,
    ]
    output = tmp_path / "результат.docx"
    initial = checklist.read_bytes()
    write_report(output, report)
    assert checklist.read_bytes() == initial
    rows = Document(output).tables[0].rows
    assert "Частично соответствует" in rows[4].cells[2].text
    assert "Калибровка – сравнение с эталоном." in rows[4].cells[3].text
    assert rows[1].cells[3].text == "Свойства Word"
    assert rows[1].cells[1].text.startswith("Соответствует\n")
    assert "Старая цитата" not in rows[1].cells[2].text
    assert load_checklist(output) == load_checklist(checklist)
    with pytest.raises(ValueError, match="исходные файлы"):
        write_report(checklist, report)

    partial_path = tmp_path / "частичный отчёт.docx"
    partial = replace(report, checklist_path=output, results=report.results[:1], cancelled=True)
    write_report(partial_path, partial)
    rows = Document(partial_path).tables[0].rows
    assert rows[1].cells[1].text.startswith("Соответствует\n")
    assert all(not cell.text for row in rows[2:] for cell in row.cells[2:4])


@pytest.mark.parametrize(
    ("length", "force", "mode"),
    [(200, False, "full"), (15_000, False, "rag"), (200, True, "rag")],
)
def test_context_mode_uses_index_only_for_rag(length, force, mode):
    block = Block("p1", "Ц" * length, Location("docx", paragraph=1))
    document = DocumentText(Path("report.docx"), (block,))
    requirement = Requirement("1", "Указать цель", "Источник только в отчёте", CheckKind.CONTENT)
    builds = []

    def build(blocks, cancelled):
        builds.append(blocks)
        return SimpleNamespace(retrieve=lambda text: blocks)

    def assess(req, chunks):
        assert req.comment == requirement.comment
        assert chunks == document.blocks
        return ModelAnswer(Status.PARTIAL, "Цель описана частично.", (Citation("p1", "Ц"),))

    service = ReviewService(
        lambda path: document,
        lambda path: (requirement,),
        build,
        lambda cancelled: nullcontext(SimpleNamespace(assess=assess)),
    )
    report = replace(service, force_rag=force).run(document.path, Path("checklist.csv"))
    assert report.context_mode == mode
    assert len(builds) == (mode == "rag")
    assert report.results[0].status is Status.PARTIAL


def test_search_keeps_large_table_together_and_overlaps_long_paragraph(tmp_path):
    path = tmp_path / "report.docx"
    document = Document()
    document.add_heading("Результаты", 1)
    document.add_paragraph(" ".join(f"фрагмент{i}" for i in range(300)))
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Параметр", "Значение"
    for value in ("Первый", "Второй", "Третий"):
        row = table.add_row()
        row.cells[0].text, row.cells[1].text = value, value * 120
    document.save(path)

    groups = tuple(search_groups(read_document(path).blocks))
    assert [b.id for b in groups[-1]] == ["t1r2", "t1r3", "t1r4"]
    parts = [b for group in groups[:-1] for b in group if b.id.startswith("p2:")]
    assert len(parts) > 1
    assert all(len(b.text) <= 1400 and b.location.paragraph == 2 for b in parts)
    assert parts[0].text[-190:-10] in parts[1].text


def test_layout_checks_all_sections_and_reads_numbered_caption(tmp_path):
    from docx.enum.section import WD_SECTION

    from sirius_proj.application.structure import check_layout

    path = tmp_path / "report.docx"
    doc = Document()
    doc.sections[0].page_width = Mm(210)
    doc.sections[0].page_height = Mm(297)
    doc.add_paragraph("Таблица 1 — Результат")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "Факт"
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.page_width = Mm(216)
    section.page_height = Mm(279)
    doc.add_paragraph("Таблица 2 — Наблюдения")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "Данные"
    doc.save(path)
    document = read_document(path)
    a4 = Requirement("1", "Файл имеет формат A4", kind=CheckKind.METADATA)
    numbering = Requirement("2", "Таблицы имеют сквозную нумерацию", kind=CheckKind.STRUCTURE)
    size = check_layout(a4, document)
    assert size.status is Status.NON_COMPLIANT and len(size.evidence) == 2
    assert check_layout(numbering, document).status is Status.COMPLIANT
    assert document.layout.table_numbers == ("1", "2")
