"""Replay test set: run real conversations through the current prompts + model and check what comes out.

There's no test suite, and a 4B model is tuned through prompts - which is guesswork without a fixed set of
cases to compare against. This runs the cases in evals/cases.yaml against the live Ollama (same prompts, notes
and filters as the bot) and reports pass/fail. Run it before and after changing a prompt.

    python -m voicebot.replay                       # everything
    python -m voicebot.replay -k roast -v           # cases whose id contains "roast", printing every reply
    python -m voicebot.replay --extract --since "2026-09-25 00:40" --until "2026-09-25 01:00"
                                                    # turn logged voice replies into draft cases (stdout)

Case kinds (see evals/cases.yaml):
  addressee - is a follow-up line for the bot?           {recent, speaker, line, for_me}
  calc      - calculator note for a line                 {line, expect (substring) | none: true}
  search    - search query (or NO) for a line            {line, context?, expect (regex) | none: true}
  reply     - the voice reply to a conversation          {history, people?, playing?, target?, expect: {...}}
              expect: include / exclude (regexes, case-insensitive), max_words, min_words
Reply cases generate `runs` times (default 2: sampling varies) and pass only if every run passes.
The bot doesn't need to be stopped - it shares Ollama with the live bot, so run it when nobody's in a call.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import discord
import yaml

from . import calc, members
from .addressee import AddresseeCheck
from .config import load_config
from .llm import LLMRouter
from .search import WebSearch
from .text_utils import ActionFilter, EchoGuard, SpeakerGuard, ThinkFilter, clean_for_speech, now_note

CASES = Path("evals/cases.yaml")


def _person(name: str, playing: str = ""):
    """Enough of a discord.Member for the roast/rizz notes."""
    acts = [discord.Game(playing)] if playing else []
    return SimpleNamespace(display_name=name, name=name, id=hash(name) & 0xFFFFFFFF, bot=False, activities=acts)


class Runner:
    def __init__(self, cfg, verbose: bool):
        self.cfg = cfg
        self.verbose = verbose
        self.llm = LLMRouter(cfg)
        self.addressee = AddresseeCheck(cfg, self.llm)
        self.search = WebSearch(cfg, self.llm)
        self.bot_name = cfg.bot.name

    # ------------------------------------------------------------------ kinds

    async def addressee_case(self, c) -> tuple[bool, str]:
        got = await self.addressee.is_for_me(c["speaker"], c["line"], c.get("recent", []), c.get("people", []))
        return got == bool(c["for_me"]), f"for_me={got}"

    async def calc_case(self, c) -> tuple[bool, str]:
        got = calc.note(c["line"]) or ""
        found = got.split(": ", 1)[1].split(". Say", 1)[0] if got else "-"
        ok = (not got) if c.get("none") else bool(got) and c["expect"] in got
        return ok, found

    async def search_case(self, c) -> tuple[bool, str]:
        q = await self.search.decide([{"role": "user", "content": c["line"]}], c.get("context", ""))
        ok = (q is None) if c.get("none") else q is not None and re.search(c["expect"], q, re.I) is not None
        return ok, repr(q)

    async def reply_case(self, c) -> tuple[bool, str]:
        people = c.get("people") or sorted({ln.split(": ", 1)[0] for ln in c["history"] if ": " in ln} - {self.bot_name})
        playing = c.get("playing", {})
        history: list[dict] = []
        for ln in c["history"]:
            role = "assistant" if ln.startswith(f"{self.bot_name}: ") else "user"
            text = ln.split(": ", 1)[1] if role == "assistant" else ln
            if history and history[-1]["role"] == role == "user":
                history[-1]["content"] += "\n" + text
            else:
                history.append({"role": role, "content": text})
        last = [ln for ln in history[-1]["content"].splitlines()]
        said = " ".join(ln.split(": ", 1)[-1] for ln in last)
        requester = last[-1].split(": ", 1)[0]
        note = calc.note(said)
        if note is None and (mode := members.roast_request(said)) and c.get("target"):
            tgt = c["target"]
            note = members.roast_note(requester, _person(tgt, playing.get(tgt, "")), c.get("target_notes", ""),
                                      [ln.split(": ", 1)[1] for ln in c["history"] if ln.startswith(f"{tgt}: ")], "",
                                      mode)
        roster = "\n".join(f"- {p}" for p in people)
        system = "\n\n".join([
            self.cfg.bot.system_prompt.replace("{name}", self.bot_name).strip(),
            self.cfg.voice.system_prompt_suffix.replace("{name}", self.bot_name).strip(),
            self.cfg.bot.time_prompt.strip(),
            f"People in the voice channel right now (display name, @username, roles):\n{roster}",
        ])
        extra = [now_note()] + ([f"[What's been going on in this call (your own notes - use them to follow along): "
                                 f"{c['scene']}]"] if c.get("scene") else [])
        if playing:
            extra.insert(0, "[Discord shows what people are up to right now - "
                         + " | ".join(f"{p}: playing {g}" for p, g in playing.items()) + "]")
        body = history[-1]["content"]
        history[-1] = {"role": "user", "content": (f"{note}\n\n" if note else "") + "\n".join(extra) + "\n" + body}
        runs, fails, shown = int(c.get("runs", 2)), [], []
        for _ in range(runs):
            reply = await self._generate([{"role": "system", "content": system}] + history, people,
                                         [ln.split(": ", 1)[1] for ln in last if ": " in ln])
            shown.append(reply)
            fails += [f"{why}: {reply!r}" for why in self._check(reply, c.get("expect", {}))]
        return not fails, " || ".join(shown) if not fails else "; ".join(fails)

    async def _generate(self, messages, people, said: list[str]) -> str:
        """Same filters as the live bot: <think>, *actions*, name labels, a copy of what they just said."""
        guard = SpeakerGuard(self.bot_name, set(people))
        think, actions, out = ThinkFilter(), ActionFilter(), []
        async for tok in self.llm.stream(messages, voice=True, purpose="replay"):
            out.append(actions.feed(think.feed(tok)))
        out.append(actions.feed(think.flush()) + actions.flush())
        text = EchoGuard(said).strip("".join(out), guard.strip_label)
        return clean_for_speech(guard.clean(text)).strip()

    @staticmethod
    def _check(reply: str, expect: dict) -> list[str]:
        why = []
        words = len(reply.split())
        for pat in expect.get("include", []):
            if not re.search(pat, reply, re.I):
                why.append(f"missing /{pat}/")
        for pat in expect.get("exclude", []):
            if re.search(pat, reply, re.I):
                why.append(f"has /{pat}/")
        if "max_words" in expect and words > int(expect["max_words"]):
            why.append(f"{words} words > {expect['max_words']}")
        if words < int(expect.get("min_words", 1)):
            why.append("empty" if not words else f"{words} words < {expect['min_words']}")
        return why

    # ------------------------------------------------------------------ run

    async def run(self, cases: list[dict], pattern: str | None) -> int:
        cases = [c for c in cases if not pattern or pattern.lower() in c["id"].lower()]
        passed, t0, by_kind = 0, time.perf_counter(), {}
        for c in cases:
            fn = getattr(self, f"{c['kind']}_case")
            try:
                ok, detail = await fn(c)
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"error: {e!r}"
            passed += ok
            k = by_kind.setdefault(c["kind"], [0, 0])
            k[0] += ok
            k[1] += 1
            if self.verbose or not ok:
                print(f"{'PASS' if ok else 'FAIL'}  {c['id']:34} {detail}")
            else:
                print(f"PASS  {c['id']}")
        summary = ", ".join(f"{k} {a}/{b}" for k, (a, b) in by_kind.items())
        print(f"\n{passed}/{len(cases)} passed ({summary}) in {time.perf_counter() - t0:.0f}s")
        return 0 if passed == len(cases) else 1


def extract(since: str, until: str | None, bot_name: str) -> None:
    """Logged voice turns -> draft reply cases (review them, add expectations, paste into cases.yaml)."""
    cmd = ["journalctl", "-u", "discord-voicebot", "--no-pager", "-o", "cat", "--since", since]
    if until:
        cmd += ["--until", until]
    lines, n = [], 0
    for raw in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines():
        if m := re.search(r"🎙 (.+)$", raw):
            lines.append(m[1])
        elif m := re.search(r"🤖 (.+)$", raw):
            if not lines:
                continue
            n += 1
            print(yaml.safe_dump([{"id": f"logged-{n}", "kind": "reply", "history": lines[-8:],
                                   "logged_reply": m[1], "expect": {"max_words": 45}}],
                                 sort_keys=False, allow_unicode=True, width=120))
            lines.append(f"{bot_name}: {m[1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("-k", help="only cases whose id contains this")
    ap.add_argument("-v", action="store_true", help="print every result, not just failures")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--extract", action="store_true", help="print draft cases from the bot's log")
    ap.add_argument("--since", default="-1d")
    ap.add_argument("--until")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.extract:
        extract(a.since, a.until, cfg.bot.name)
        return
    cases = yaml.safe_load(Path(a.cases).read_text())
    sys.exit(asyncio.run(Runner(cfg, a.v).run(cases, a.k)))


if __name__ == "__main__":
    main()
