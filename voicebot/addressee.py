"""Is a follow-up line actually for the bot?

In wake-word mode, anything its conversation partner says inside the follow-up window used to count as talking
to the bot - so "All right.", "What?" to a friend, or "is she roasting you or hyping you up?" got an answer
nobody wanted. Lines that say its name never come here; only follow-ups do. A tiny neutral LLM call reads the
last few lines and answers YES/NO (~60-100ms; the reply to a real follow-up waits for it, a line it rejects
costs nothing more). Obvious cases skip the call: a line that starts by naming someone else is for them.
Any error or timeout falls back to the old behaviour (it's for the bot).
"""
from __future__ import annotations

import asyncio
import logging
import re

log = logging.getLogger("voicebot.addressee")

_WORD = re.compile(r"[\w']+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def names_someone_else(text: str, others: list[str]) -> bool:
    """ "Mitch, look at this" / "yo jordan, ..." - opens by naming another person in the call."""
    words = _WORD.findall(text.lower())[:3]
    if words and words[0] in ("yo", "hey", "oi", "dude", "bro", "okay", "ok", "so"):
        words = words[1:]
    first = _norm(words[0]) if words else ""
    return len(first) >= 3 and any(first == _norm(o) or (len(first) >= 4 and _norm(o).startswith(first))
                                   for o in others)


class AddresseeCheck:
    def __init__(self, cfg, llm):
        self.cfg = cfg.voice.addressee
        self.llm = llm
        self.bot_name = cfg.bot.name
        self.enabled = bool(self.cfg.enabled)
        self.stats = {"checked": 0, "not for me": 0, "errors": 0}

    async def is_for_me(self, name: str, text: str, recent: list[str], others: list[str], asked_them: bool = True) -> bool:
        """recent: the last few conversation lines ("name: text", the bot's included), oldest first.
        asked_them: this person is who the bot's last reply was for (only then does "it asked a question, so this is
        the answer" apply - the bot's rhetorical "huh?" at the group made everyone's next line count)."""
        if not self.enabled:
            return True
        if names_someone_else(text, [o for o in others if o != name]):
            self.stats["not for me"] += 1
            log.info("↪ not for me (names someone else): %s: %s", name, text)
            return False
        last = recent[-1] if recent else ""
        if asked_them and last.startswith(f"{self.bot_name}:") and "?" in last.split(":", 1)[1]:
            # She just asked something ("You?", "what have you been up to?"): the next line is the answer. The small
            # model kept calling plain answers "NO" (they don't sound like they're *to* anyone), so no call here.
            self.stats["answered her"] = self.stats.get("answered her", 0) + 1
            return True
        self.stats["checked"] += 1
        convo = "\n".join(recent[-int(self.cfg.context_lines):])
        prompt = self.cfg.prompt.replace("{name}", self.bot_name)
        try:
            answer = await asyncio.wait_for(self.llm.complete([
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Conversation so far:\n{convo}\n\nNew line - {name}: {text}\n\n"
                                            f"Is {name}'s new line meant for {self.bot_name}? YES or NO."},
            ], endpoint=self.cfg.endpoint or None, temperature=0, max_tokens=3, purpose="addressee"),
                timeout=float(self.cfg.timeout_s))
        except Exception as e:  # noqa: BLE001 - never go deaf over this
            self.stats["errors"] += 1
            log.warning("Addressee check failed (%s) - treating it as for me", e or type(e).__name__)
            return True
        yes = not answer.strip().upper().startswith("NO")
        if not yes:
            self.stats["not for me"] += 1
            log.info("↪ not for me: %s: %s", name, text)
        return yes
