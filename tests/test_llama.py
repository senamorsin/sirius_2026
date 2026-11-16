"""Проверки запросов к модели, формата ответов, языка и ссылок на источники."""

import httpx
import pytest

from sirius_proj.adapters.llama_client import LlamaJudge
from sirius_proj.application.errors import CheckError
from sirius_proj.domain.models import Block, Location, Requirement, Status


def runtime(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:12345")
    return LlamaJudge(client)


def test_model_receives_table_context_without_owning_source_addresses():
    import json

    def handler(request):
        payload = json.loads(request.content)
        assert list(payload["response_format"]["schema"]["properties"])[-1] == "status"
        user = json.loads(payload["messages"][-1]["content"])
        assert "ТК-101" in user["fragments"][0]["section"]
        assert user["fragments"][0]["text"] == "Напряжение | 220 В"
        assert "page" not in user["fragments"][0]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"status":"compliant","explanation":"Указано",'
                            '"evidence":[{"chunk_id":"t1","quote":"Напряжение | 220 В"}]}',
                        },
                    }
                ]
            },
        )

    llm = runtime(handler)
    answer = llm.assess(
        Requirement("1", "Питание ТК-101 220 В"),
        [Block("t1", "Напряжение | 220 В", Location("docx", heading="ТК-101", table=1, row=1))],
    )
    assert answer.status is Status.COMPLIANT
    llm.client.close()


@pytest.mark.parametrize("code", [400, 500])
def test_http_error_is_technical_failure(code):
    llm = runtime(lambda request: httpx.Response(code, json={"error": "bad"}))
    with pytest.raises(RuntimeError, match="локальной модели"):
        llm.assess(Requirement("1", "Питание"), [])
    llm.client.close()


def test_truncated_generation_is_not_a_document_status():
    llm = runtime(
        lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})
    )
    with pytest.raises(CheckError, match="обрезан"):
        llm.assess(Requirement("1", "Питание"), [])
    llm.client.close()


def test_foreign_explanation_is_translated_without_changing_decision_or_citations():
    import json

    calls = []
    answers = [
        {
            "status": "non_compliant",
            "explanation": "版本是2，不是4。",
            "evidence": [{"chunk_id": "row", "quote": "2"}],
        },
        {"explanation": "Редакция равна 2, а требуется 4."},
    ]

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(answers[len(calls) - 1], ensure_ascii=False)
                        },
                    }
                ]
            },
        )

    llm = runtime(handler)
    answer = llm.assess(Requirement("R", "Редакция 4"), [])
    assert answer.status is Status.NON_COMPLIANT
    assert answer.evidence[0].chunk_id == "row"
    assert answer.evidence[0].quote == "2"
    assert answer.explanation == "Редакция равна 2, а требуется 4."
    assert len(calls) == 2
    assert "СТРОГО НА РУССКОМ" in calls[0]["messages"][0]["content"]
    assert "status" not in calls[1]["response_format"]["schema"]["properties"]
    llm.client.close()


def test_explanation_still_in_foreign_language_is_a_technical_error():
    import json

    count = 0

    def handler(request):
        nonlocal count
        count += 1
        answer = {"explanation": "没有数据。"}
        if count == 1:
            answer.update(status="insufficient", evidence=[])
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]
            },
        )

    llm = runtime(handler)
    with pytest.raises(CheckError, match="на русском языке"):
        llm.assess(Requirement("R", "Параметр"), [])
    assert count == 2
    llm.client.close()


def test_table_cell_reference_resolves_to_original_text_and_row():
    import json

    from sirius_proj.domain.evidence import verify_answer
    from sirius_proj.domain.models import TableRow

    block = Block(
        "t3r2",
        "Руководство | 2",
        Location("docx", table=3, row=2),
        TableRow(("Документ", "Редакция"), ("Руководство", "2")),
    )

    def handler(request):
        data = json.loads(json.loads(request.content)["messages"][-1]["content"])
        cell = data["tables"][0]["rows"][0]["cells"][1]
        assert cell["cell_id"] == "t3r2:cell2"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "non_compliant",
                                    "explanation": "Передана редакция 2 вместо 4.",
                                    "evidence": [{"chunk_id": cell["cell_id"], "quote": ""}],
                                },
                                ensure_ascii=False,
                            )
                        },
                    }
                ]
            },
        )

    llm = runtime(handler)
    requirement = Requirement("N", "Редакция 4")
    answer = llm.assess(requirement, [block])
    result = verify_answer(requirement, answer, {block.id: block})
    assert result.status is Status.NON_COMPLIANT
    assert result.evidence[0].quote == "2"
    assert result.evidence[0].location.row == 2
    llm.client.close()


def test_generation_can_only_reference_sources_present_in_the_request():
    import json

    from sirius_proj.domain.models import TableRow

    def handler(request):
        schema = json.loads(request.content)["response_format"]["schema"]
        allowed = schema["properties"]["evidence"]["items"]["properties"]["chunk_id"]["enum"]
        assert set(allowed) == {"p1", "row:cell1", "row:cell2"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"status":"insufficient",'
                            '"explanation":"Недостаточно данных","evidence":[]}'
                        },
                    }
                ]
            },
        )

    llm = runtime(handler)
    llm.assess(
        Requirement("R", "Проверка"),
        [
            Block("p1", "Вводный текст", Location("docx", paragraph=1)),
            Block(
                "row",
                "Параметр | 2",
                Location("docx", table=1, row=2),
                TableRow(("Параметр", "Значение"), ("Параметр", "2")),
            ),
        ],
    )
    llm.client.close()
