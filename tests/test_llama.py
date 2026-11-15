from pathlib import Path

import httpx
import pytest

from sirius_proj.domain import Block, Location, Requirement, Status
from sirius_proj.llama import LlamaRuntime
from sirius_proj.settings import Settings


def runtime(handler):
    instance = LlamaRuntime(Settings(Path("server"), Path("model"), Path("e5")))
    instance.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:12345"
    )
    return instance


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
    llm.close()


@pytest.mark.parametrize("code", [400, 500])
def test_http_error_is_technical_failure(code):
    llm = runtime(lambda request: httpx.Response(code, json={"error": "bad"}))
    with pytest.raises(RuntimeError, match="локальной модели"):
        llm.assess(Requirement("1", "Питание"), [])
    llm.close()


def test_truncated_generation_is_not_a_document_status():
    llm = runtime(
        lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})
    )
    with pytest.raises(ValueError, match="обрезан"):
        llm.assess(Requirement("1", "Питание"), [])
    llm.close()
