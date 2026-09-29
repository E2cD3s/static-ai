"""OpenAI-compatible LLM client (Ollama /v1, LM Studio, llama.cpp, vLLM, OpenAI, OpenRouter, ...)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI, BadRequestError

from .text_utils import ThinkFilter

log = logging.getLogger("voicebot.llm")


def _join_content(a: str | list, b: str | list) -> str | list:
    """Content is a string, or a list of parts when images are attached."""
    if isinstance(a, str) and isinstance(b, str):
        return a + "\n" + b
    def parts(c):
        return [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
    return parts(a) + parts(b)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Many chat templates (Mistral, Gemma and lots of RP fine-tunes) reject histories that don't
    strictly alternate user/assistant, or that start with an assistant turn. Merge consecutive
    same-role messages and drop leading assistant turns left over from history trimming."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and not any(x["role"] == "user" for x in out):
            continue
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": _join_content(out[-1]["content"], m["content"])}
        else:
            out.append(dict(m))
    return out


@dataclass
class Tally:
    requests: int = 0
    errors: int = 0
    cancelled: int = 0          # stopped early (barge-in, speaker guard, speculative reply dropped)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0         # summed over `timed` requests
    gen_s: float = 0.0          # first -> last token, summed
    gen_tokens: int = 0         # tokens generated inside gen_s (for tok/s)
    timed: int = 0

    def add(self, other: "Tally") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


class LLMUsage:
    """Token and speed accounting for every LLM call, by purpose (voice, text, search, ...). Stats only."""

    def __init__(self):
        self.by_purpose: dict[str, Tally] = {}
        self.last_prompt: dict[str, int] = {}   # purpose -> prompt tokens of its latest request (context fill)
        self.recent: deque[tuple[float, str, float]] = deque(maxlen=50)  # (when, purpose, tok/s)

    def record(self, purpose: str, t0: float, first: float | None, last: float | None, chunks: int,
               usage, status: str) -> None:
        t = self.by_purpose.setdefault(purpose, Tally())
        t.requests += 1
        t.errors += status == "error"
        t.cancelled += status == "cancelled"
        completion = getattr(usage, "completion_tokens", None) or chunks  # no usage when cut short: ~1 token/chunk
        prompt = getattr(usage, "prompt_tokens", None) or 0
        t.prompt_tokens += prompt
        t.completion_tokens += completion
        if prompt:
            self.last_prompt[purpose] = prompt
        if first is not None:
            t.timed += 1
            t.ttft_s += first - t0
            if last is not None and last > first and completion > 1:
                t.gen_s += last - first
                t.gen_tokens += completion - 1
                self.recent.append((time.time(), purpose, (completion - 1) / (last - first)))

    def total(self) -> Tally:
        out = Tally()
        for t in self.by_purpose.values():
            out.add(t)
        return out


class LLMRouter:
    def __init__(self, cfg):
        self.endpoints = cfg.llm.endpoints
        self.text_endpoint: str = cfg.llm.default
        self.voice_endpoint: str = cfg.llm.voice_endpoint or cfg.llm.default
        self.model_overrides: dict[str, str] = {}
        self._clients: dict[str, AsyncOpenAI] = {}
        self.usage = LLMUsage()
        self._no_usage: set[str] = set()  # endpoints that reject stream_options

    def client(self, name: str) -> AsyncOpenAI:
        if name not in self._clients:
            ep = self.endpoints[name]
            self._clients[name] = AsyncOpenAI(
                base_url=ep.base_url, api_key=ep.api_key or "not-needed", timeout=float(ep.timeout)
            )
        return self._clients[name]

    def model_for(self, name: str) -> str:
        return self.model_overrides.get(name) or self.endpoints[name].model

    def describe(self, voice: bool) -> str:
        name = self.voice_endpoint if voice else self.text_endpoint
        return f"{name} ({self.model_for(name)} @ {self.endpoints[name].base_url})"

    async def stream(self, messages: list[dict], voice: bool = False, endpoint: str | None = None,
                     temperature: float | None = None, max_tokens: int | None = None,
                     purpose: str | None = None) -> AsyncIterator[str]:
        name = endpoint or (self.voice_endpoint if voice else self.text_endpoint)
        ep = self.endpoints[name]
        params = {
            "model": self.model_for(name),
            "messages": normalize_messages(messages),
            "stream": True,
            "temperature": float(ep.temperature if temperature is None else temperature),
            "max_tokens": int(max_tokens or (ep.voice_max_tokens if voice else ep.max_tokens)),
        }
        for key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
            if ep.get(key) is not None:
                params[key] = ep[key]
        if ep.extra_body:
            params["extra_body"] = dict(ep.extra_body)
        if name not in self._no_usage:
            params["stream_options"] = {"include_usage": True}  # token counts in the last chunk (for /stats)
        t0, first, last, chunks, usage, status = time.perf_counter(), None, None, 0, None, "ok"
        try:
            try:
                stream = await self.client(name).chat.completions.create(**params)
            except BadRequestError:
                if "stream_options" not in params:
                    raise
                self._no_usage.add(name)
                del params["stream_options"]
                stream = await self.client(name).chat.completions.create(**params)
            try:
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        last = time.perf_counter()
                        first = first or last
                        chunks += 1
                        yield delta
            finally:
                # Closing the HTTP stream makes the server stop generating (important on barge-in).
                await stream.close()
        except BaseException as e:
            status = "cancelled" if isinstance(e, (GeneratorExit, asyncio.CancelledError)) else "error"
            raise
        finally:
            self.usage.record(purpose or ("voice" if voice else "text"), t0, first, last, chunks, usage, status)

    async def complete(self, messages: list[dict], voice: bool = False, endpoint: str | None = None,
                       temperature: float | None = None, max_tokens: int | None = None,
                       purpose: str | None = None) -> str:
        tf = ThinkFilter()
        raw: list[str] = []
        parts = []
        async for tok in self.stream(messages, voice, endpoint, temperature, max_tokens, purpose):
            raw.append(tok)
            parts.append(tf.feed(tok))
        parts.append(tf.flush())
        out = "".join(parts).strip()
        if not out and "".join(raw).strip():
            log.warning("LLM reply was all <think> (nothing left to say): %.300r", "".join(raw))
        return out

    async def list_models(self, name: str) -> list[str]:
        page = await self.client(name).models.list()
        return sorted(m.id for m in page.data)

    async def warmup(self, attempts: int = 15) -> None:
        """Forces the model to load into VRAM so the first real reply isn't slow.
        Retries so the bot can start at boot while Ollama is still coming up."""
        for name in {self.text_endpoint, self.voice_endpoint}:
            ep = self.endpoints[name]
            for attempt in range(1, attempts + 1):
                try:
                    await self.client(name).chat.completions.create(
                        model=self.model_for(name),
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                        **({"extra_body": dict(ep.extra_body)} if ep.extra_body else {}),
                    )
                    log.info("LLM endpoint '%s' ready (%s)", name, self.model_for(name))
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == attempts:
                        log.warning("LLM warmup failed for '%s': %s", name, e)
                    else:
                        await asyncio.sleep(2)
