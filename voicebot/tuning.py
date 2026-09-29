"""Self-tuning: the bot adjusts how it talks in voice from how people react, per server.

Two knobs, both nudged a step at a time and slowly relaxed back, persisted in a small JSON file:
  * brevity level 0-3: a one-line hint added to the newest turn (never the system prompt - the cached
    prompt prefix has to stay identical). Goes up when people cut a long reply short or say "too long" /
    "shut up"; comes back down after a run of replies nobody interrupted. "Tell me more" lowers it and
    allows one longer answer.
  * follow-up window (wake-word mode): how long after a reply it still answers without its name.
    Shrinks when a reply it gave without being named gets talked over (people were talking to each other);
    grows when someone says its name just after the window closed (they were still talking to it).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("voicebot.tuning")

HINTS = {
    1: "[Keep this reply short: one or two sentences.]",
    2: "[One short sentence only.]",
    3: "[Answer in just a few words.]",
}
MORE_HINT = "[They asked for more - you can go a bit longer this time, still spoken and casual, no lists.]"
LEVEL_NAMES = {0: "normal", 1: "short", 2: "one sentence", 3: "few words"}

_SHORTER = re.compile(r"\b(?:too long|shut up|stop talking|talk(?:s|ing)? too much|get to the point|tl;?dr|"
                      r"shorter|stop rambling|you'?re rambling|stop yapping|yapping|less talking|be brief)\b", re.I)
_LONGER = re.compile(r"\b(?:tell me more|go on|keep going|elaborate|more detail|explain (?:that )?more|"
                     r"say more|what else|go deeper|longer answer)\b", re.I)

RELAX_AFTER = 6          # uninterrupted replies in a row before brevity steps back down
LONG_ENOUGH_WORDS = 15   # talked over after this many spoken words = "too long", not just "wanted to jump in"


class GuildTuning:
    def __init__(self, store: "TuningStore", data: dict):
        self.store = store
        self.level = int(data.get("level", 0))
        self.followup_s = max(store.min_followup, float(data.get("followup_s", store.base_followup)))
        self.streak = int(data.get("streak", 0))
        self.more_next = bool(data.get("more_next", False))
        self.log: list[tuple[float, str]] = [tuple(x) for x in data.get("log", [])][-10:]

    def to_json(self) -> dict:
        return {"level": self.level, "followup_s": self.followup_s, "streak": self.streak,
                "more_next": self.more_next, "log": self.log}

    def _change(self, why: str) -> None:
        self.log = (self.log + [(time.time(), why)])[-10:]
        log.info("🎛 %s (brevity %s, follow-up %.0fs)", why, LEVEL_NAMES[self.level], self.followup_s)
        self.store.save()

    # ------------------------------------------------------------------ what the voice loop reads

    def hint(self) -> str | None:
        if not self.store.enabled:
            return None
        return MORE_HINT if self.more_next else HINTS.get(self.level)

    def followup(self) -> float:
        return self.followup_s if self.store.enabled else self.store.base_followup

    # ------------------------------------------------------------------ signals

    def heard(self, text: str, named: bool, since_reply_s: float | None) -> None:
        """Someone said something to the bot (by name, or inside the follow-up window)."""
        if not self.store.enabled:
            return
        if _SHORTER.search(text):
            self.streak = 0
            if self.level < 3:
                self.level += 1
                self._change(f"told \"{text[:40]}\" → shorter replies")
        elif _LONGER.search(text):
            self.more_next = True
            if self.level > 0:
                self.level -= 1
                self._change("asked for more → longer replies")
            else:
                self.store.save()
        if (named and since_reply_s is not None and self.followup_s < self.store.max_followup
                and self.followup_s < since_reply_s <= self.followup_s + 10):
            self.followup_s = min(self.store.max_followup, self.followup_s + 2)
            self._change(f"called by name {since_reply_s:.0f}s after replying → longer follow-up window")

    def finished(self) -> None:
        """A reply played to the end without anyone taking over."""
        if not self.store.enabled:
            return
        self.more_next = False
        self.streak += 1
        if self.streak >= RELAX_AFTER and self.level > 0:
            self.level -= 1
            self.streak = 0
            self._change(f"{RELAX_AFTER} replies in a row without interruption → a bit longer again")
        else:
            self.store.save()

    def interrupted(self, words_spoken: int, by_followup: bool) -> None:
        """Someone talked over a reply until it stopped (barge-in)."""
        if not self.store.enabled:
            return
        self.streak = 0
        self.more_next = False
        why = []
        if words_spoken >= LONG_ENOUGH_WORDS and self.level < 3:
            self.level += 1
            why.append("shorter replies")
        # Only when anyone in the window gets an answer: then being talked over without the name suggests people
        # were talking among themselves. With speaker-scoped follow-ups that can't happen, and it's usually the
        # person the bot is talking with cutting in - shrinking the window then just makes them repeat its name.
        if by_followup and self.store.shrink_on_interrupt and self.followup_s > self.store.min_followup:
            self.followup_s = max(self.store.min_followup, self.followup_s - 3)
            why.append("shorter follow-up window")
        if why:
            self._change(f"talked over after {words_spoken} words" + (" (not called by name)" if by_followup else "")
                         + " → " + " + ".join(why))
        else:
            self.store.save()

    def reset(self) -> None:
        self.level, self.followup_s, self.streak, self.more_next = 0, self.store.base_followup, 0, False
        self._change("reset")


class TuningStore:
    def __init__(self, cfg):
        self.enabled = bool(cfg.tuning.enabled)
        self.path = Path(cfg.tuning.path)
        self.base_followup = float(cfg.voice.wake_word_followup_s)
        self.min_followup = max(10.0, self.base_followup * 0.6)
        self.shrink_on_interrupt = cfg.voice.get("followup_scope", "speaker") == "anyone"
        self.max_followup = self.base_followup * 1.5
        self._guilds: dict[int, GuildTuning] = {}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            raw = {}
        for gid, data in raw.items():
            self._guilds[int(gid)] = GuildTuning(self, data)

    def for_guild(self, guild_id: int) -> GuildTuning:
        if guild_id not in self._guilds:
            self._guilds[guild_id] = GuildTuning(self, {})
        return self._guilds[guild_id]

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({str(g): t.to_json() for g, t in self._guilds.items()}, indent=1))
            tmp.replace(self.path)
        except OSError as e:
            log.warning("Couldn't save tuning state: %s", e)
