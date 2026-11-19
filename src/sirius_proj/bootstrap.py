"""Сборка приложения: подключаем чтение файлов, поиск, модель, окно и экспорт."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from sirius_proj.application.ports import SearchIndex
from sirius_proj.application.review import ReviewService
from sirius_proj.config import Settings, default_data_dir
from sirius_proj.domain.models import Block, CheckReport


def create_service(settings: Settings) -> ReviewService:
    from sirius_proj.adapters.documents import load_checklist, read_document
    from sirius_proj.adapters.llama_process import open_llama

    def build_index(blocks: tuple[Block, ...], cancelled: Callable[[], bool]) -> SearchIndex:
        from sirius_proj.adapters.embeddings import E5Encoder
        from sirius_proj.adapters.retrieval import SectionIndex

        return SectionIndex(blocks, E5Encoder(settings.embedding_dir), cancelled)

    return ReviewService(
        read_document,
        load_checklist,
        build_index,
        partial(open_llama, settings),
        force_rag=settings.force_rag,
    )


def write_report(path: Path, report: CheckReport) -> None:
    from sirius_proj.adapters.checklist import write_docx_report
    from sirius_proj.adapters.xlsx import write_report as write_xlsx_report

    inputs = (report.document.path, report.checklist_path)
    if any(source and path.resolve() == source.resolve() for source in inputs):
        raise ValueError("Выберите другое имя, чтобы сохранить исходные файлы.")
    if path.suffix.lower() == ".docx":
        write_docx_report(path, report)
    elif path.suffix.lower() == ".xlsx":
        write_xlsx_report(path, report)
    else:
        raise ValueError("Отчёт можно сохранить в формате XLSX или DOCX.")


def launch_gui(settings: Settings) -> int:
    from sirius_proj.installation import install_components
    from sirius_proj.ui.window import launch

    def installer(progress, cancelled):
        return install_components(settings.data_dir or default_data_dir(), progress, cancelled)

    return launch(settings, create_service, write_report, installer)
