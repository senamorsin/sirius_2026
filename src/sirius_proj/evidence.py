"""Pure evidence policy. A model never supplies source locations."""

import json
import re
from collections.abc import Mapping

from .domain import Block, Citation, Evidence, Finding, ModelAnswer, Requirement, Status


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_answer(raw: str) -> ModelAnswer:
    try:
        value = json.loads(raw)
        status = Status(value["status"])
        explanation = value["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Нет пояснения")
        if not isinstance(value["evidence"], list):
            raise ValueError("evidence должен быть массивом")
        citations = tuple(Citation(item["chunk_id"], item["quote"]) for item in value["evidence"])
        if status is not Status.INSUFFICIENT and not citations:
            raise ValueError("Для определённого вывода не переданы цитаты")
        return ModelAnswer(status, explanation.strip(), citations)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Некорректный структурированный ответ модели: {error}") from error


def verify_answer(
    requirement: Requirement, answer: ModelAnswer, blocks: Mapping[str, Block],
) -> Finding:
    evidence = []
    for citation in answer.evidence:
        block = blocks.get(citation.chunk_id)
        if block is None or normalize(citation.quote) not in normalize(block.text):
            return Finding(
                requirement, Status.INSUFFICIENT,
                "Цитата или идентификатор фрагмента не подтверждены исходным текстом. "
                "Требуется проверка специалистом.", (),
            )
        item = Evidence(citation.chunk_id, citation.quote, block.location)
        if item not in evidence:
            evidence.append(item)
    return Finding(requirement, answer.status, answer.explanation, tuple(evidence))
