from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import sirius_proj.ui.window as gui
from sirius_proj.adapters.documents import read_document
from sirius_proj.adapters.xlsx import write_report
from sirius_proj.config import Settings
from sirius_proj.domain.models import (
    CheckFailure,
    CheckReport,
    Evidence,
    Finding,
    Requirement,
    Status,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    monkeypatch.setattr(
        gui,
        "QSettings",
        lambda *args: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
    )
    settings = Settings(tmp_path / "server", tmp_path / "model", tmp_path / "e5")
    settings.server_binary.touch()
    settings.model_file.touch()
    settings.embedding_dir.mkdir()
    for name in ("model.onnx", "tokenizer.json"):
        (settings.embedding_dir / name).touch()
    widget = gui.MainWindow(settings, lambda settings: None, write_report)
    widget.server_edit.setText(str(settings.server_binary))
    widget.model_edit.setText(str(settings.model_file))
    widget.embedding_edit.setText(str(settings.embedding_dir))
    widget.show()
    app.processEvents()
    yield widget
    assert widget.worker is None
    widget.close()


def sample_report():
    doc = read_document(Path("examples/report.pdf"))
    evidence = Evidence("page1", "220 В", doc.blocks[0].location)
    results = (
        Finding(Requirement("1", "Питание 220 В"), Status.COMPLIANT, "Указано", (evidence,)),
        Finding(
            Requirement("2", "Питание 380 В"), Status.NON_COMPLIANT, "Другое питание", (evidence,)
        ),
        Finding(Requirement("3", "Подпись"), Status.INSUFFICIENT, "Подпись не найдена", ()),
        CheckFailure(Requirement("4", "Гарантия"), "Тайм-аут"),
    )
    return CheckReport(doc, results, 1.0)


def test_results_sources_filtering_and_export(window, monkeypatch, tmp_path):
    report = sample_report()
    for result in report.results:
        window._on_result(result)
    window._on_report(report)
    window._set_busy(False)
    assert window.table.rowCount() == 4
    assert "Страница 1" in window.details.toPlainText()
    assert "220 В" in window.details.toPlainText()
    window.filter.setCurrentIndex(2)
    assert window.table.isRowHidden(0)
    assert not window.table.isRowHidden(1)
    window.table.selectRow(2)
    assert "Подтверждённых цитат нет" in window.details.toPlainText()
    window.table.selectRow(3)
    assert "Тайм-аут" in window.details.toPlainText()
    path = tmp_path / "export"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.save_report()
    assert path.with_suffix(".xlsx").is_file()
    assert window.export_button.isEnabled()


def test_run_uses_worker_and_clears_previous_results(window, app, monkeypatch):
    report = sample_report()

    def run(document, checklist, **callbacks):
        callbacks["on_progress"](0, 4, "Работа")
        for result in report.results:
            callbacks["on_result"](result)
        return report

    window.service_factory = lambda settings: SimpleNamespace(run=run)
    window.document_edit.setText(str(Path("examples/report.pdf").resolve()))
    window.checklist_edit.setText(str(Path("examples/checklist.csv").resolve()))
    window.start_check()
    assert not window.run_button.isEnabled()
    deadline = monotonic() + 5
    while window.worker is not None and monotonic() < deadline:
        app.processEvents()
    assert window.worker is None
    assert window.run_button.isEnabled()
    assert window.table.rowCount() == 4
    assert window.report is report


def test_invalid_paths_do_not_start_worker(window):
    window.document_edit.setText("missing.pdf")
    window.start_check()
    assert window.worker is None


def test_service_configuration_failure_is_shown_without_starting_worker(window):
    window.document_edit.setText(str(Path("examples/report.pdf").resolve()))
    window.checklist_edit.setText(str(Path("examples/checklist.csv").resolve()))

    def fail(settings):
        raise ValueError("Недоступна конфигурация модели")

    window.service_factory = fail
    window.start_check()
    assert window.worker is None
    assert window.run_button.isEnabled()


def test_settings_and_file_selection(window, monkeypatch, tmp_path):
    window.settings_toggle.setChecked(True)
    assert window.settings_panel.isVisible()
    window.settings_toggle.setChecked(False)
    assert not window.settings_panel.isVisible()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("report.pdf", ""))
    window._choose(window.document_edit, "*")
    assert window.document_edit.text() == "report.pdf"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window._choose_embedding()
    assert window.embedding_edit.text() == str(tmp_path)
