"""Фоновые потоки установки и проверки: передача результатов в окно и отмена работы."""

import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from sirius_proj.application.review import ReviewService
from sirius_proj.config import Settings

type Installer = Callable[[Callable[[str], None], Callable[[], bool]], Settings]


class SetupWorker(QThread):
    progress = Signal(str)
    prepared = Signal(object)
    failed = Signal(str)

    def __init__(self, installer: Installer):
        super().__init__()
        self.installer = installer
        self.cancel_event = threading.Event()

    def run(self):
        try:
            self.prepared.emit(self.installer(self.progress.emit, self.cancel_event.is_set))
        except Exception as error:
            self.failed.emit(str(error))


class CheckWorker(QThread):
    progress = Signal(int, int, str)
    result = Signal(object)
    report_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, document: Path, checklist: Path, service: ReviewService):
        super().__init__()
        self.document = document
        self.checklist = checklist
        self.service = service
        self.cancel_event = threading.Event()

    def run(self):
        try:
            report = self.service.run(
                self.document,
                self.checklist,
                on_result=self.result.emit,
                on_progress=self.progress.emit,
                cancelled=self.cancel_event.is_set,
            )
            self.report_ready.emit(report)
        except Exception as error:
            self.failed.emit(str(error))
