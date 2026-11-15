import threading
from pathlib import Path

import numpy as np

from sirius_proj.domain import (
    Block,
    CheckFailure,
    DocumentText,
    Location,
    Requirement,
    Status,
)
from sirius_proj.evidence import parse_answer
from sirius_proj.search import HybridIndex
from sirius_proj.service import check_requirements


class Encoder:
    def encode(self, texts, *, query=False):
        if query:
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)
        return np.array([[1.0, 0.0], [0.1, 0.99], [0.9, 0.1]], dtype=np.float32)


def blocks():
    return tuple(
        Block(f"b{i}", text, Location("pdf", page=i + 1))
        for i, text in enumerate(
            [
                "Общие сведения о тяговом оборудовании",
                "Контроллер КТ-1234 снят с производства",
                "Оборудование проверено",
            ]
        )
    )


def test_exact_identifier_is_not_lost_to_semantic_ranking():
    index = HybridIndex(blocks(), Encoder(), limit=2)
    found = index.retrieve("Применён контроллер КТ-1234")
    assert found[0].id == "b1"
    assert len(found) == 2


def test_index_encodes_document_only_once():
    class CountingEncoder(Encoder):
        calls = []

        def encode(self, texts, *, query=False):
            self.calls.append(query)
            return super().encode(texts, query=query)

    encoder = CountingEncoder()
    index = HybridIndex(blocks(), encoder)
    index.retrieve("Первое")
    index.retrieve("Второе")
    assert encoder.calls == [False, True, True]


class Index:
    def retrieve(self, requirement):
        return (Block("b1", "Питание 220 В", Location("pdf", page=1)),)


class Judge:
    def assess(self, requirement, chunks):
        return parse_answer(
            '{"status":"compliant","explanation":"Указано питание",'
            '"evidence":[{"chunk_id":"b1","quote":"Питание 220 В"}]}'
        )


def test_service_reports_each_requirement_and_separates_failures():
    class BrokenJudge(Judge):
        def assess(self, requirement, chunks):
            if requirement.id == "2":
                raise TimeoutError("Модель не ответила")
            return super().assess(requirement, chunks)

    requirements = (Requirement("1", "Питание"), Requirement("2", "Мощность"))
    seen = []
    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        requirements,
        Index(),
        BrokenJudge(),
        on_result=seen.append,
    )
    assert len(report.results) == 2
    assert report.results[0].status is Status.COMPLIANT
    assert isinstance(report.results[1], CheckFailure)
    assert "не ответила" in report.results[1].message
    assert seen == list(report.results)


def test_service_cancels_between_requirements():
    event = threading.Event()
    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        (Requirement("1", "Питание"), Requirement("2", "Другое")),
        Index(),
        Judge(),
        on_result=lambda _: event.set(),
        cancelled=event.is_set,
    )
    assert report.cancelled
    assert len(report.results) == 1


def test_model_cannot_cite_unretrieved_document_block():
    class OtherIndex(Index):
        def retrieve(self, requirement):
            return (Block("b2", "Нет данных", Location("pdf", page=2)),)

    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        (Requirement("1", "Питание"),),
        OtherIndex(),
        Judge(),
    )
    assert report.results[0].status is Status.INSUFFICIENT
