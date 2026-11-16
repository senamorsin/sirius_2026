"""Проверка цитат по исходному тексту. Адреса источников берём из документа."""

import re
from collections.abc import Mapping

from sirius_proj.domain.models import Block, Evidence, Finding, ModelAnswer, Requirement, Status


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def verify_answer(
    requirement: Requirement,
    answer: ModelAnswer,
    blocks: Mapping[str, Block],
) -> Finding:
    """Сверяет цитаты с переданными фрагментами и добавляет адреса источников."""
    evidence = []
    for citation in answer.evidence:
        block = blocks.get(citation.chunk_id)
        if block is None or normalize(citation.quote) not in normalize(block.text):
            return Finding(
                requirement,
                Status.INSUFFICIENT,
                "Цитата или идентификатор фрагмента не подтверждены исходным текстом. "
                "Требуется проверка специалистом.",
                (),
            )
        item = Evidence(citation.chunk_id, citation.quote, block.location)
        if item not in evidence:
            evidence.append(item)
    return Finding(requirement, answer.status, answer.explanation, tuple(evidence))
