"""The single composition root for concrete infrastructure and entry points."""

from collections.abc import Callable
from pathlib import Path

from sirius_proj.application.ports import SearchIndex
from sirius_proj.application.review import ReviewService
from sirius_proj.config import Settings
from sirius_proj.domain.models import Block, CheckReport


def create_service(settings: Settings) -> ReviewService:
    from sirius_proj.adapters.documents import load_checklist, read_document
    from sirius_proj.adapters.llama_process import LlamaServer

    settings.validate()

    def build_index(blocks: tuple[Block, ...], cancelled: Callable[[], bool]) -> SearchIndex:
        from sirius_proj.adapters.embeddings import E5Encoder
        from sirius_proj.adapters.search import HybridIndex

        encoder = E5Encoder(settings.embedding_dir)
        chunks = encoder.split(blocks, cancelled)
        if cancelled():
            raise InterruptedError("Проверка отменена")
        return HybridIndex(chunks, encoder)

    return ReviewService(
        read_document,
        load_checklist,
        build_index,
        lambda cancelled: LlamaServer(settings, cancelled),
    )


def write_report(path: Path, report: CheckReport) -> None:
    from sirius_proj.adapters.xlsx import write_report as save

    save(path, report)


def launch_gui(settings: Settings) -> int:
    from sirius_proj.ui.window import launch

    return launch(settings, create_service, write_report)
