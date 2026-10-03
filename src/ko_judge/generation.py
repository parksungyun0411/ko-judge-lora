"""판정 백엔드.

- OpenAIBackend: vLLM `vllm serve`의 OpenAI 호환 /v1/chat/completions.
  - judge(): 생성형 판정(제로샷). 논문 저장소 src/config.py처럼 temperature 0.3, max_tokens 512,
    형식 실패 시 최대 3회 재시도(요청마다 다른 seed).
  - score(): 점수만. max_tokens 1, 상위 20개 logprob에서 '1'~'5' 분포를 만든다.
- HFBackend.score(): transformers로 같은 분포를 계산한다(스모크·CPU 확인용).
"""

from __future__ import annotations

import asyncio
import time

import httpx

from ko_judge.parsing import SCORES, extract_result, score_distribution


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, concurrency: int = 32, timeout: float = 300.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

    def judge_payload(self, messages: list[dict], seed: int) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
            "seed": seed,
        }

    def score_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
        }

    async def _post(self, client, sem, body) -> tuple[dict, float]:
        async with sem:
            t0 = time.perf_counter()
            r = await client.post(self.url, json=body)
            r.raise_for_status()
            return r.json(), time.perf_counter() - t0

    async def _judge_one(self, client, sem, messages, seed: int, retries: int) -> dict:
        texts = []
        for attempt in range(retries + 1):
            resp, _ = await self._post(client, sem, self.judge_payload(messages, seed + attempt))
            text = resp["choices"][0]["message"]["content"] or ""
            texts.append(text)
            score = extract_result(text)
            if score is not None:
                break
        return {"score": score, "attempts": len(texts), "output": texts[-1]}

    async def _score_one(self, client, sem, messages) -> dict:
        resp, latency = await self._post(client, sem, self.score_payload(messages))
        content = resp["choices"][0]["logprobs"]["content"]
        top = {t["token"]: t["logprob"] for t in content[0]["top_logprobs"]} if content else {}
        return {"dist": score_distribution(top), "latency_s": latency}

    async def _run(self, coros_fn, batch):
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await asyncio.gather(*(coros_fn(client, sem, m) for m in batch))

    def judge(self, batch: list[list[dict]], seed: int = 0, retries: int = 3) -> list[dict]:
        return asyncio.run(
            self._run(lambda c, s, m: self._judge_one(c, s, m, seed, retries), batch)
        )

    def score(self, batch: list[list[dict]]) -> list[dict]:
        return asyncio.run(self._run(self._score_one, batch))


class HFBackend:
    def __init__(self, model_path: str, adapter: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ko_judge import compat

        compat.apply()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.to(device).eval()
        self.device = device
        self.score_ids = {s: self.tok.convert_tokens_to_ids(str(s)) for s in SCORES}

    def score(self, batch: list[list[dict]]) -> list[dict]:
        out = []
        for messages in batch:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits[0, -1].float()
            lp = self.torch.log_softmax(logits, dim=-1)
            top = {str(s): float(lp[i]) for s, i in self.score_ids.items()}
            out.append({"dist": score_distribution(top), "latency_s": None})
        return out
