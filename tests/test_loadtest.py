import asyncio
import functools

import httpx
import pytest

from ko_judge import loadtest
from ko_judge.generation import OpenAIBackend


def test_percentile_nearest_rank():
    v = [float(i) for i in range(1, 101)]
    assert loadtest.percentile(v, 95) == 95.0
    assert loadtest.percentile([3.0, 1.0, 2.0], 50) == 2.0
    with pytest.raises(ValueError):
        loadtest.percentile([], 95)


def test_payloads():
    b = OpenAIBackend("http://x/v1/", "m")
    s = b.score_payload([{"role": "user", "content": "q"}])
    assert b.url == "http://x/v1/chat/completions"
    assert s["max_tokens"] == 1 and s["logprobs"] and s["top_logprobs"] == 20
    j = b.judge_payload([], seed=7)
    assert j["temperature"] == 0.3 and j["max_tokens"] == 512 and j["seed"] == 7


def _mock(monkeypatch, module, handler):
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )


def test_run_against_mock_server(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(500) if len(seen) == 7 else httpx.Response(200, json={})

    _mock(monkeypatch, loadtest, handler)
    msgs = [[{"role": "user", "content": str(i)}] for i in range(25)]
    res = asyncio.run(loadtest.run(OpenAIBackend("http://mock/v1", "m"), msgs, 4, 5))
    assert len(seen) == 25 and res["completed"] + res["errors"] == 20 and res["errors"] == 1


def test_score_and_judge_against_mock(monkeypatch):
    from ko_judge import generation

    calls = {"n": 0}

    def handler(request):
        import json

        body = json.loads(request.content)
        if body["max_tokens"] == 1:
            top = [{"token": "4", "logprob": -0.1}, {"token": "5", "logprob": -2.4}]
            return httpx.Response(
                200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
            )
        calls["n"] += 1
        text = "형식 없음" if calls["n"] == 1 else "Feedback: 무난 [RESULT] 3"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    _mock(monkeypatch, generation, handler)
    b = OpenAIBackend("http://mock/v1", "m", concurrency=1)
    s = b.score([[{"role": "user", "content": "q"}]])[0]
    assert max(s["dist"], key=s["dist"].get) == 4
    j = b.judge([[{"role": "user", "content": "q"}]], seed=0)[0]
    assert j["score"] == 3 and j["attempts"] == 2  # 첫 응답은 형식 실패 → 재시도
