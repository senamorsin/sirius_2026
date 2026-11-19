"""Главное окно. Рабочий поток управляет моделями и передаёт окну готовые результаты."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont
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
    QVBoxLayout,
    QWidget,
)

from sirius_proj.application.review import ReviewService
from sirius_proj.config import Settings
from sirius_proj.domain.models import CheckFailure, CheckReport, CheckResult, Status
from sirius_proj.ui.results import ResultTable, describe_result
from sirius_proj.ui.style import STYLE
from sirius_proj.ui.worker import CheckWorker, Installer, SetupWorker


def path_control(edit: QLineEdit, button: QPushButton) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return widget


class MainWindow(QMainWindow):
    def __init__(
        self,
        defaults: Settings,
        service_factory: Callable[[Settings], ReviewService],
        report_writer: Callable[[Path, CheckReport], None],
        installer: Installer | None = None,
    ):
        super().__init__()
        self.defaults = defaults
        self.service_factory = service_factory
        self.report_writer = report_writer
        self.installer = installer
        self.preferences = QSettings("Sirius2026", "DocumentReview")
        self.worker: CheckWorker | SetupWorker | None = None
        self.report: CheckReport | None = None
        self.pending_close = False
        self.setWindowTitle("Контроль документации · Сириус")
        self.resize(1280, 900)
        self.setMinimumSize(960, 720)
        self.setStyleSheet(STYLE)
        self._build_ui()
        if installer is not None:
            QTimer.singleShot(0, self._initial_setup)

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
        self.checklist_edit.setPlaceholderText("Пустой шаблон DOCX или CSV с требованиями")
        doc_button = QPushButton("Выбрать…")
        doc_button.clicked.connect(
            lambda: self._choose(self.document_edit, "Документы (*.docx *.pdf)")
        )
        list_button = QPushButton("Выбрать…")
        list_button.clicked.connect(
            lambda: self._choose(self.checklist_edit, "Чек-лист (*.docx *.csv)")
        )
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
        self.rag_check = QCheckBox("Использовать поиск фрагментов даже для коротких отчётов (RAG)")
        self.rag_check.setChecked(self.defaults.force_rag)
        form.addRow("", self.rag_check)
        self.prepare_button = QPushButton("Подготовить компоненты")
        self.prepare_button.setEnabled(self.installer is not None)
        self.prepare_button.clicked.connect(self.prepare_components)
        form.addRow("", self.prepare_button)
        return panel

    def _settings(self) -> Settings:
        return replace(
            self.defaults,
            server_binary=Path(self.server_edit.text()),
            model_file=Path(self.model_edit.text()),
            embedding_dir=Path(self.embedding_edit.text()),
            use_gpu=self.gpu_check.isChecked(),
            force_rag=self.rag_check.isChecked(),
        )

    def _initial_setup(self):
        if self._settings().missing_components:
            self.prepare_components()

    def prepare_components(self):
        if self.installer is None or self.worker is not None:
            return
        self.progress.setRange(0, 0)
        worker = SetupWorker(self.installer)
        worker.progress.connect(self.message.setText)
        worker.prepared.connect(self._components_ready)
        self._start_worker(worker)

    def _components_ready(self, settings):
        self.defaults = settings
        self.server_edit.setText(str(settings.server_binary))
        self.model_edit.setText(str(settings.model_file))
        self.embedding_edit.setText(str(settings.embedding_dir))

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
        self.export_button = QPushButton("Сохранить отчёт")
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
        self.table = ResultTable()
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
        if self.worker is not None:
            return
        settings = self._settings()
        document = Path(self.document_edit.text())
        checklist = Path(self.checklist_edit.text())
        try:
            if not document.is_file() or not checklist.is_file():
                raise ValueError("Выберите существующие документ и чек-лист")
            service = self.service_factory(settings)
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
        self.table.clear_results()
        self.details.clear()
        self.summary.setText("Проверено: 0")
        worker = CheckWorker(document, checklist, service)
        worker.progress.connect(self._on_progress)
        worker.result.connect(self._on_result)
        worker.report_ready.connect(self._on_report)
        self._start_worker(worker)

    def _start_worker(self, worker: CheckWorker | SetupWorker):
        self.worker = worker
        self._set_busy(True)
        worker.failed.connect(self._on_failure)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _set_busy(self, busy):
        self.input_card.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.report is not None and bool(self.results))

    def _on_progress(self, done, total, message):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.message.setText(message)

    @property
    def results(self) -> tuple[CheckResult, ...]:
        return self.table.results

    def _on_result(self, result):
        self.table.append_result(result)
        self.summary.setText(f"Проверено: {len(self.results)}")
        self._filter_rows()

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
        if not self.pending_close and not message.endswith("отменена"):
            QMessageBox.warning(self, "Проверка не выполнена", message)

    def _on_finished(self):
        self.progress.setRange(0, 100)
        completed = (
            isinstance(self.worker, CheckWorker) and self.report and not self.report.cancelled
        )
        self.progress.setValue(100 if completed else 0)
        self._set_busy(False)
        self.worker.deleteLater()
        self.worker = None
        if self.pending_close:
            self.close()

    def _filter_rows(self):
        self.table.filter_status(self.filter.currentData())

    def _show_evidence(self):
        result = self.table.selected_result()
        if result is not None:
            self.details.setPlainText(describe_result(result))

    def save_report(self):
        if self.report is None:
            return
        word = self.report.checklist_path and self.report.checklist_path.suffix.lower() == ".docx"
        filters = "Word (*.docx);;Excel (*.xlsx)" if word else "Excel (*.xlsx)"
        default = "Проверка.docx" if word else "Проверка.xlsx"
        path, selected = QFileDialog.getSaveFileName(self, "Сохранить отчёт", default, filters)
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() not in (".docx", ".xlsx"):
            destination = destination.with_suffix(
                ".docx" if selected.startswith("Word") else ".xlsx"
            )
        try:
            self.report_writer(destination, self.report)
            self.message.setText(f"Отчёт сохранён: {destination}")
        except Exception as error:
            QMessageBox.warning(self, "Не удалось сохранить отчёт", str(error))

    def cancel_check(self):
        active = self.worker
        if active is not None:
            active.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.message.setText("Остановка текущей операции…")

    def closeEvent(self, event):
        active = self.worker
        if active is not None and active.isRunning():
            self.pending_close = True
            self.cancel_check()
            event.ignore()
        else:
            event.accept()


def launch(settings: Settings, service_factory, report_writer, installer=None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings, service_factory, report_writer, installer)
    window.show()
    return app.exec()
