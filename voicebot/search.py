"""Web search: your SearXNG first, DuckDuckGo (keyless, via the ddgs library) as the fallback.

Before each reply a tiny, neutral LLM call looks at the last few lines and answers with a search query
or NO. (Asking the in-character model to call a search tool itself is unreliable at 4B - it roleplays
"let me check" and then makes something up.) Results are added to that one turn's prompt as short plain
text, placed before the person's message (after it, the 4B model tends to echo their question back),
and never stored in history - the reply carries whatever it used.
~120ms per reply when no search is needed; the chat's cached prompt prefix survives the extra call.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

import httpx
from ddgs import DDGS

from .text_utils import is_backchannel, now_note

log = logging.getLogger("voicebot.search")
for _noisy in ("ddgs", "primp"):  # they log every HTTP request at INFO
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_TAGS = re.compile(r"<[^>]+>")
# Pure reactions never need a search: skip the ~120ms check for them.
_REACTION = re.compile(
    r"(?:lol|lmao|lmfao|rofl|ha(?:ha)+|he(?:he)+|bruh|bro|dude|same|nice|damn|dang|fr|true|facts|bet|word|"
    r"no way|yo|hey|hi|hello|sup|thanks?(?: you)?|thx|ty|gg|gn|good ?night|bye|see ya|cya|oh (?:no|my god|wow)|"
    r"that'?s (?:crazy|wild|insane|funny|hilarious|cool|awesome|sick|nuts)|i know|me too|for real|let'?s go)"
    r"[\s.!?,]*", re.I)
_REACTIONS = re.compile(f"(?:{_REACTION.pattern})+", re.I)  # "Nice. Thank you."


def _clean(text: str, limit: int) -> str:
    text = " ".join(html.unescape(_TAGS.sub("", text or "")).split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


class WebSearch:
    def __init__(self, cfg, llm):
        self.cfg = cfg.search
        self.llm = llm
        self.bot_name = cfg.bot.name
        self.backends = [name for name, ok in (("searxng", self.cfg.searxng_url),
                                               ("duckduckgo", self.cfg.duckduckgo)) if ok]
        self.enabled = bool(self.cfg.enabled and self.backends)
        self.stats: Counter[str] = Counter()  # for /stats: checks, searches, per-backend ok/empty/fail + ms
        self._http = httpx.AsyncClient(timeout=float(self.cfg.timeout), follow_redirects=True,
                                       headers={"User-Agent": "static-bot/1.0", **dict(self.cfg.searxng_headers or {})})

    async def decide(self, history: list[dict], context: str = "") -> str | None:
        """Search query for the last message of this conversation, or None if no search is needed.
        `context`: what people are doing right now, so "tell me about the game" searches the actual game."""
        lines = [m["content"] if m["role"] == "user" else f"{self.bot_name}: {m['content']}"
                 for m in history[-int(self.cfg.context_lines):]]
        convo = "\n".join(" ".join(line.split())[:300] for line in lines)
        if context and context not in convo:
            convo = f"{context}\n{convo}"
        answer = await self.llm.complete([
            {"role": "system", "content": self.cfg.decide_prompt.strip()},
            {"role": "user", "content": f"{now_note()}\n\nConversation:\n{convo}\n\n"
                                        "Search query or NO?"},
        ], endpoint=self.cfg.endpoint or None, temperature=0, max_tokens=32, purpose="search check")
        lines_out = answer.strip().splitlines()
        query = lines_out[0].strip(" \"'`.") if lines_out else ""
        if not query or query.upper().startswith("NO") or len(query) > 150:
            return None
        return query

    async def lookup(self, history: list[dict], before_search=None, context: str = "") -> str | None:
        """Decide, then search. Returns text to add to the prompt, or None. `before_search` is called
        once a search is going to happen (voice uses it to say "let me look that up")."""
        if not self.enabled or not history:
            return None
        trailing = []  # the unanswered turn(s): "what's new in X?" then "thank you" still needs a search
        for m in reversed(history):
            if m["role"] != "user":
                break
            trailing.append(m["content"])
        if all(self._is_chatter(c) for c in trailing or [history[-1]["content"]]):
            self.stats["skipped (chatter)"] += 1
            return None
        self.stats["checks"] += 1
        try:
            query = await self.decide(history, context)
        except Exception as e:  # noqa: BLE001
            log.warning("Search decision failed: %s", e)
            return None
        if not query:
            return None
        if before_search:
            before_search()
        return await self.run(query)

    @staticmethod
    def _is_chatter(content: str) -> bool:
        """The message is only reactions ("lol", "no way", "that's crazy") - nothing to look up.
        [Bracketed] lines (time, presence, notes) are ignored; every spoken line has to be a reaction."""
        spoken = [line.split(": ", 1)[-1].strip() for line in content.strip().splitlines()
                  if line.strip() and not line.lstrip().startswith("[")]
        return bool(spoken) and all(is_backchannel(t) or bool(_REACTIONS.fullmatch(t)) for t in spoken)

    async def run(self, query: str) -> str:
        """Results as prompt text. Never raises: a failure becomes a note the model can relay."""
        self.stats["searches"] += 1
        for backend in self.backends:
            t0 = time.perf_counter()
            try:
                results, answers = await getattr(self, f"_{backend}")(query)
            except Exception as e:  # noqa: BLE001
                log.warning("🔎 %s failed for %r: %r", backend, query, e)
                self.stats[f"{backend} fail"] += 1
                continue
            ms = (time.perf_counter() - t0) * 1000
            self.stats[f"{backend} ok" if results or answers else f"{backend} empty"] += 1
            self.stats[f"{backend} ms"] += ms
            if results or answers:
                log.info("🔎 [%s %.0fms] %r -> %d results", backend, ms, query, len(results))
                return self._format(query, results, answers)
            log.info("🔎 [%s %.0fms] %r -> nothing", backend, ms, query)
        return (f'[You tried to search the web for "{query}" but it didn\'t work. '
                "If you don't know the answer, say you couldn't look it up - don't guess.]")

    @staticmethod
    def _date(value) -> str:
        """'Sep 22' from an ISO date, '' if there isn't one. Tells the model how fresh a result is."""
        try:
            return f"{datetime.fromisoformat(str(value).replace('Z', '+00:00')):%b %d, %Y}"
        except ValueError:
            return ""

    def _format(self, query: str, results: list[dict], answers: list[str]) -> str:
        lines = [f'[Web search for "{query}" - results below. Base your answer ONLY on facts stated in them. '
                 "Read them carefully: check which result actually answers the question, and don't mix up names "
                 "from different results. If they don't answer it, say you couldn't find it. "
                 "Answer in character and short, like you just know it - don't talk about 'results', "
                 "'snippets' or 'searches', and no links.]"]
        lines += [f"Quick answer: {_clean(a, 400)}" for a in answers[:2]]
        for i, r in enumerate(results[: int(self.cfg.max_results)], 1):
            site = urlparse(r.get("url") or "").netloc.removeprefix("www.")
            date = self._date(r.get("date")) if r.get("date") else ""
            lines.append(f"{i}. {_clean(r.get('title', ''), 120)} ({site}{', ' + date if date else ''}): "
                         f"{_clean(r.get('snippet', ''), 300)}")
        return "\n".join(lines)

    async def _searxng(self, query: str) -> tuple[list[dict], list[str]]:
        params = {"q": query, "format": "json", "safesearch": 0}
        if self.cfg.searxng_timeout_limit:
            params["timeout_limit"] = self.cfg.searxng_timeout_limit
        if self.cfg.language:
            params["language"] = self.cfg.language
        r = await self._http.get(self.cfg.searxng_url.rstrip("/") + "/search", params=params)
        r.raise_for_status()  # 403 = "json" isn't in search.formats in SearXNG's settings.yml
        data = r.json()
        results = [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content"),
                    "date": x.get("publishedDate")} for x in data.get("results", [])]
        answers = [a if isinstance(a, str) else a.get("answer", "") for a in data.get("answers", [])]
        answers += [f"{b.get('infobox', '')}: {b.get('content', '')}" for b in data.get("infoboxes", []) if b.get("content")]
        return results, [a for a in answers if a]

    async def _duckduckgo(self, query: str) -> tuple[list[dict], list[str]]:
        """Web + news results at once: plain web results are often stale, news ones are dated and recent."""
        n = int(self.cfg.max_results)
        timeout = int(self.cfg.timeout)

        def web():
            return DDGS(timeout=timeout).text(query, region="us-en", max_results=n)

        def news():
            return DDGS(timeout=timeout).news(query, region="us-en", max_results=2)

        loop = asyncio.get_running_loop()
        got = await asyncio.gather(loop.run_in_executor(None, web), loop.run_in_executor(None, news),
                                   return_exceptions=True)
        if all(isinstance(g, Exception) for g in got):
            raise got[0]
        web_r, news_r = (g if isinstance(g, list) else [] for g in got)
        results = [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("body"), "date": x.get("date")}
                   for x in news_r]
        results += [{"title": x.get("title"), "url": x.get("href"), "snippet": x.get("body")} for x in web_r]
        return results[:n], []

    async def close(self) -> None:
        await self._http.aclose()
