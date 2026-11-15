"""Qt presentation. The worker owns adapters; widgets receive immutable values."""

import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .application import run_job
from .domain import CheckFailure, CheckReport, CheckResult, Status
from .export import export_xlsx
from .settings import Settings

COLORS = {
    Status.COMPLIANT: "#15785d",
    Status.NON_COMPLIANT: "#b23a3a",
    Status.INSUFFICIENT: "#946518",
}

STYLE = """
QMainWindow { background: #f3f6fa; }
QWidget { font-family: 'DejaVu Sans'; font-size: 13px; color: #1d3449; }
QFrame#card { background: white; border: 1px solid #dce4ed; border-radius: 10px; }
QLabel#title { font-size: 26px; font-weight: 700; }
QLabel#subtitle { color: #607588; }
QLabel#eyebrow { color: #14836c; font-size: 11px; font-weight: 700; }
QPushButton { background: #e8eef5; border: 0; border-radius: 6px; padding: 10px 16px; }
QPushButton:hover { background: #d8e5ef; }
QPushButton#primary { background: #176f63; color: white; font-weight: 700; }
QPushButton#primary:hover { background: #105a50; }
QPushButton:disabled { background: #e3e8ee; color: #8996a3; }
QLineEdit, QComboBox { background: white; border: 1px solid #ccd7e3; border-radius: 5px;
                      padding: 8px; selection-background-color: #176f63; }
QTableWidget { background: white; border: 1px solid #dce4ed; gridline-color: #e9eef4;
               selection-background-color: #e1f2ed; selection-color: #123f38; }
QHeaderView::section { background: #edf2f7; border: 0; padding: 10px; font-weight: 700; }
QPlainTextEdit { background: white; border: 1px solid #dce4ed; border-radius: 6px; padding: 8px; }
QProgressBar { border: 0; background: #e1e8f0; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #218b77; border-radius: 4px; }
"""


class CheckWorker(QThread):
    progress = Signal(int, int, str)
    result = Signal(object)
    report_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, document: Path, checklist: Path, settings: Settings):
        super().__init__()
        self.document = document
        self.checklist = checklist
        self.settings = settings
        self.cancel_event = threading.Event()

    def run(self):
        try:
            report = run_job(
                self.document,
                self.checklist,
                self.settings,
                on_result=self.result.emit,
                on_progress=self.progress.emit,
                cancelled=self.cancel_event.is_set,
            )
            self.report_ready.emit(report)
        except Exception as error:
            self.failed.emit(str(error))


def path_control(edit: QLineEdit, button: QPushButton) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return widget


class MainWindow(QMainWindow):
    def __init__(self, defaults: Settings):
        super().__init__()
        self.defaults = defaults
        self.preferences = QSettings("Sirius2026", "DocumentReview")
        self.worker = None
        self.report: CheckReport | None = None
        self.results: list[CheckResult] = []
        self.pending_close = False
        self.setWindowTitle("Контроль документации · Сириус")
        self.resize(1280, 900)
        self.setMinimumSize(960, 720)
        self.setStyleSheet(STYLE)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)
        eyebrow = QLabel("СИРИУС / ЛОКАЛЬНЫЙ АНАЛИЗ")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)
        title = QLabel("Контроль технической документации")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(
            "Проверка по чек-листу с цитатами и адресами источников. Решение принимает специалист."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        layout.addWidget(self._input_card())
        layout.addLayout(self._toolbar())
        layout.addWidget(self._result_area(), 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.message = QLabel(
            "Выберите документ и чек-лист. После подготовки моделей интернет не нужен."
        )
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.setCentralWidget(root)

    def _input_card(self):
        card = QFrame()
        card.setObjectName("card")
        form = QFormLayout(card)
        form.setContentsMargins(18, 16, 18, 16)
        self.document_edit = QLineEdit()
        self.document_edit.setPlaceholderText("Отчёт подрядчика · DOCX или PDF с текстовым слоем")
        self.checklist_edit = QLineEdit()
        self.checklist_edit.setPlaceholderText("CSV в UTF-8 · колонки id,requirement")
        doc_button = QPushButton("Выбрать…")
        doc_button.clicked.connect(
            lambda: self._choose(self.document_edit, "Документы (*.docx *.pdf)")
        )
        list_button = QPushButton("Выбрать…")
        list_button.clicked.connect(lambda: self._choose(self.checklist_edit, "Чек-лист (*.csv)"))
        form.addRow("Документ", path_control(self.document_edit, doc_button))
        form.addRow("Чек-лист", path_control(self.checklist_edit, list_button))
        self.settings_toggle = QPushButton("Настройки локальных моделей ▸")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.toggled.connect(self._toggle_settings)
        form.addRow("", self.settings_toggle)
        self.settings_panel = self._settings_panel()
        self.settings_panel.hide()
        form.addRow(self.settings_panel)
        self.input_card = card
        return card

    def _settings_panel(self):
        panel = QWidget()
        form = QFormLayout(panel)
        self.server_edit = QLineEdit(
            str(self.preferences.value("server", str(self.defaults.server_binary)))
        )
        self.model_edit = QLineEdit(
            str(self.preferences.value("model", str(self.defaults.model_file)))
        )
        self.embedding_edit = QLineEdit(
            str(self.preferences.value("e5", str(self.defaults.embedding_dir)))
        )
        for label, edit in [("llama-server", self.server_edit), ("GGUF-модель", self.model_edit)]:
            button = QPushButton("…")
            button.clicked.connect(
                lambda checked=False, target=edit: self._choose(target, "Все файлы (*)")
            )
            form.addRow(label, path_control(edit, button))
        button = QPushButton("…")
        button.clicked.connect(self._choose_embedding)
        form.addRow("Каталог E5", path_control(self.embedding_edit, button))
        self.gpu_check = QCheckBox("Использовать GPU (снимите для режима CPU)")
        self.gpu_check.setChecked(True)
        form.addRow("", self.gpu_check)
        return panel

    def _toggle_settings(self, checked):
        self.settings_panel.setVisible(checked)
        self.settings_toggle.setText("Настройки локальных моделей " + ("▾" if checked else "▸"))

    def _choose(self, edit, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл", "", file_filter)
        if path:
            edit.setText(path)

    def _choose_embedding(self):
        path = QFileDialog.getExistingDirectory(self, "Каталог с model.onnx и tokenizer.json")
        if path:
            self.embedding_edit.setText(path)

    def _toolbar(self):
        bar = QHBoxLayout()
        self.run_button = QPushButton("Запустить проверку")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.start_check)
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_check)
        self.summary = QLabel("Проверено: 0")
        self.filter = QComboBox()
        self.filter.addItem("Все результаты", "all")
        for status in Status:
            self.filter.addItem(status.label, status.value)
        self.filter.addItem("Технические ошибки", "error")
        self.filter.currentIndexChanged.connect(self._filter_rows)
        self.export_button = QPushButton("Сохранить Excel")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.save_report)
        for widget in (self.run_button, self.cancel_button, self.summary):
            bar.addWidget(widget)
        bar.addStretch()
        bar.addWidget(self.filter)
        bar.addWidget(self.export_button)
        return bar

    def _result_area(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Требование", "Статус", "Пояснение"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(2, 190)
        self.table.itemSelectionChanged.connect(self._show_evidence)
        splitter.addWidget(self.table)
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 0, 0, 0)
        label = QLabel("Доказательства и источники")
        label.setFont(QFont("DejaVu Sans", 12, QFont.Weight.Bold))
        layout.addWidget(label)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText(
            "Выберите строку результата: здесь появятся точные цитаты, "
            "их адреса и пояснение проверки."
        )
        layout.addWidget(self.details)
        splitter.addWidget(pane)
        splitter.setSizes([820, 360])
        return splitter

    def start_check(self):
        settings = replace(
            self.defaults,
            server_binary=Path(self.server_edit.text()),
            model_file=Path(self.model_edit.text()),
            embedding_dir=Path(self.embedding_edit.text()),
            use_gpu=self.gpu_check.isChecked(),
        )
        document = Path(self.document_edit.text())
        checklist = Path(self.checklist_edit.text())
        try:
            if not document.is_file() or not checklist.is_file():
                raise ValueError("Выберите существующие документ и CSV-чек-лист")
            settings.validate()
        except ValueError as error:
            QMessageBox.warning(self, "Проверка не запущена", str(error))
            return
        for key, value in [
            ("server", settings.server_binary),
            ("model", settings.model_file),
            ("e5", settings.embedding_dir),
        ]:
            self.preferences.setValue(key, str(value))
        self.report = None
        self.results.clear()
        self.table.setRowCount(0)
        self.details.clear()
        self.summary.setText("Проверено: 0")
        self._set_busy(True)
        self.worker = CheckWorker(document, checklist, settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.report_ready.connect(self._on_report)
        self.worker.failed.connect(self._on_failure)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _set_busy(self, busy):
        self.input_card.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.report is not None and bool(self.results))

    def _on_progress(self, done, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.message.setText(message)

    def _on_result(self, result):
        row = self.table.rowCount()
        self.results.append(result)
        self.table.insertRow(row)
        failure = isinstance(result, CheckFailure)
        status = "Техническая ошибка" if failure else result.status.label
        explanation = result.message if failure else result.explanation
        values = [result.requirement.id, result.requirement.text, status, explanation]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 2:
                item.setForeground(QColor("#b23a3a" if failure else COLORS[result.status]))
            self.table.setItem(row, column, item)
        self.table.resizeRowToContents(row)
        self.summary.setText(f"Проверено: {len(self.results)}")
        self._filter_rows()
        if row == 0:
            self.table.selectRow(0)

    def _on_report(self, report):
        self.report = report
        warning = "\n".join(report.document.warnings)
        if warning:
            QMessageBox.warning(self, "Проверен только доступный текст", warning)
        state = "Остановлено" if report.cancelled else "Готово"
        failures = sum(isinstance(result, CheckFailure) for result in report.results)
        self.message.setText(
            f"{state} · {len(report.results)} пунктов · "
            f"{report.seconds:.1f} с · технических ошибок: {failures}. "
            "Окончательное решение принимает специалист."
        )

    def _on_failure(self, message):
        self.message.setText(message)
        if not self.pending_close and message != "Проверка отменена":
            QMessageBox.warning(self, "Проверка не выполнена", message)

    def _on_finished(self):
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.report and not self.report.cancelled else 0)
        self._set_busy(False)
        self.worker.deleteLater()
        self.worker = None
        if self.pending_close:
            self.close()

    def _filter_rows(self):
        selected = self.filter.currentData()
        for row, result in enumerate(self.results):
            value = "error" if isinstance(result, CheckFailure) else result.status.value
            self.table.setRowHidden(row, selected not in ("all", value))

    def _show_evidence(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.results):
            return
        result = self.results[row]
        if isinstance(result, CheckFailure):
            self.details.setPlainText("Техническая ошибка\n\n" + result.message)
            return
        parts = [result.requirement.text, result.status.label, result.explanation]
        for number, item in enumerate(result.evidence, 1):
            parts.append(
                f"{number}. {item.location.label}\nФрагмент: {item.chunk_id}\n«{item.quote}»"
            )
        if not result.evidence:
            parts.append("Подтверждённых цитат нет.")
        self.details.setPlainText("\n\n".join(parts))

    def save_report(self):
        if self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "Проверка.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".xlsx":
            destination = destination.with_suffix(".xlsx")
        warnings = self.report.document.warnings
        if self.report.cancelled:
            warnings += ("Проверка отменена: экспортированы только завершённые пункты",)
        try:
            export_xlsx(destination, self.report.results, self.report.document.path, warnings)
            self.message.setText(f"Отчёт сохранён: {destination}")
        except Exception as error:
            QMessageBox.warning(self, "Не удалось сохранить отчёт", str(error))

    def cancel_check(self):
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.message.setText("Остановка после текущего запроса к модели…")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.pending_close = True
            self.cancel_check()
            event.ignore()
        else:
            event.accept()


def launch(settings: Settings) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings)
    window.show()
    return app.exec()
