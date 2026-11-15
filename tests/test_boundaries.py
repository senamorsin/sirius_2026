from dataclasses import replace
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from sirius_proj.adapters.documents import load_checklist, read_document
from sirius_proj.adapters.llama_client import parse_answer
from sirius_proj.adapters.xlsx import export_xlsx
from sirius_proj.config import Settings
from sirius_proj.domain.models import Block, CheckFailure, Location, Requirement


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "pdf", "page": 0},
        {"kind": "pdf", "page": True},
        {"kind": "pdf", "page": 2, "paragraph": 1},
        {"kind": "docx"},
        {"kind": "docx", "paragraph": -1},
        {"kind": "docx", "table": 1},
        {"kind": "docx", "row": 1},
        {"kind": "docx", "table": 0, "row": 1},
        {"kind": "docx", "table": 1, "row": 1, "paragraph": 1},
        {"kind": "docx", "paragraph": 1, "page": 1},
        {"kind": "other", "page": 1},
    ],
)
def test_invalid_source_addresses_are_rejected(values):
    with pytest.raises(ValueError):
        Location(**values)


def test_valid_address_labels():
    assert Location("pdf", page=1).label == "Страница 1"
    assert "абзац 2" in Location("docx", paragraph=2).label
    assert "строка 3" in Location("docx", table=2, row=3).label


@pytest.mark.parametrize(
    "text",
    [
        "wrong,headers\n1,abc\n",
        "id,requirement\n",
        "id,requirement\n1,\n",
        "id,requirement\n1,a,b\n",
        "id,requirement\n1\n",
    ],
)
def test_bad_checklist_has_actionable_error(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_checklist(path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"insufficient","explanation":"","evidence":[]}',
        '{"status":"insufficient","explanation":"x","evidence":{}}',
        '{"status":"insufficient","explanation":"x","evidence":[{"chunk_id":"x","quote":" "}]}',
    ],
)
def test_invalid_values_at_model_boundary(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)


def test_pdf_pages_and_empty_page_warning(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.append(PdfReader("examples/report.pdf"))
    path = tmp_path / "partial.pdf"
    writer.write(path)
    result = read_document(path)
    assert result.blocks[0].location.page == 2
    assert "220" in result.blocks[0].text
    assert "Страница 1" in result.warnings[0]


def test_scan_and_password_protected_pdf_fail_clearly(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    path = tmp_path / "scan.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="OCR"):
        read_document(path)
    writer.encrypt("private-password")
    writer.write(path)
    with pytest.raises(ValueError, match="паролем"):
        read_document(path)


def test_missing_unsupported_and_corrupt_documents(tmp_path):
    with pytest.raises(ValueError, match="не найден"):
        read_document(tmp_path / "missing.pdf")
    path = tmp_path / "file.txt"
    path.write_text("text")
    with pytest.raises(ValueError, match="DOCX"):
        read_document(path)
    path = tmp_path / "bad.docx"
    path.write_text("not a zip")
    with pytest.raises(ValueError, match="Не удалось"):
        read_document(path)


def test_atomic_export_failure_preserves_old_destination(tmp_path, monkeypatch):
    path = tmp_path / "report.xlsx"
    path.write_bytes(b"previous")
    from openpyxl import Workbook

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(Workbook, "save", fail)
    with pytest.raises(OSError):
        export_xlsx(path, (), Path("input.pdf"), ())
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_export_failure_has_no_substantive_status(tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "report.xlsx"
    export_xlsx(
        path, (CheckFailure(Requirement("1", "Питание"), "Сбой модели"),), Path("report.pdf"), ()
    )
    book = load_workbook(path)
    assert book["Результаты"].cell(2, 3).value is None
    assert book["Результаты"].cell(2, 7).value == "Сбой модели"


def test_settings_validate_paths_and_limits(tmp_path):
    settings = Settings(tmp_path / "server", tmp_path / "model", tmp_path / "e5")
    with pytest.raises(ValueError, match="llama-server"):
        settings.validate()
    settings.server_binary.touch()
    with pytest.raises(ValueError, match="GGUF"):
        settings.validate()
    settings.model_file.touch()
    settings.embedding_dir.mkdir()
    for name in ["model.onnx", "tokenizer.json"]:
        with pytest.raises(ValueError, match=name):
            settings.validate()
        (settings.embedding_dir / name).touch()
    settings.validate()
    for invalid in [
        replace(settings, request_timeout=0),
        replace(settings, context_size=10),
        replace(settings, output_tokens=9000),
    ]:
        with pytest.raises(ValueError):
            invalid.validate()


def test_empty_domain_text_is_rejected():
    with pytest.raises(ValueError):
        Requirement("", "text")
    with pytest.raises(ValueError):
        Block("x", " ", Location("pdf", page=1))
