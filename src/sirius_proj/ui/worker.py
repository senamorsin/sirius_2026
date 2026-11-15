"""Only Qt threading and cancellation; no concrete model or file adapters."""

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from sirius_proj.application.review import ReviewService


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
