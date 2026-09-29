"""Reminders and polls, from voice ("Static, remind everyone in 20 minutes to...") or text/slash commands.

Natural language: a cheap keyword check (`wants`), then a tiny neutral LLM call turns the message into
JSON (same idea as the search check - the in-character 4B model can't be trusted to call tools). The
code does the action and returns a one-turn note, so the bot confirms it in its own words.
Times are parsed by dateparser from the phrase the person used; the model never does date math.

Jobs live in SQLite so they survive restarts; one scheduler task sleeps until the next one is due.
  * reminder: DM (personal) or a channel post pinging everyone it's for (group). If those people are
    in voice with the bot, it also says it out loud.
  * poll: a native Discord poll. Discord's shortest poll is 1 hour, so shorter ones are ended early
    by the scheduler, which then announces the result in voice.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import dateparser
import discord

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.reminders")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,                  -- 'reminder' | 'poll'
    guild_id INTEGER NOT NULL DEFAULT 0,
    channel_id INTEGER NOT NULL DEFAULT 0,
    message_id INTEGER NOT NULL DEFAULT 0,  -- polls: the poll message
    creator_id INTEGER NOT NULL,
    creator_name TEXT NOT NULL DEFAULT '',
    targets TEXT NOT NULL DEFAULT '[]',  -- JSON list of user ids
    deliver TEXT NOT NULL DEFAULT 'dm',  -- reminders: 'dm' | 'channel'
    text TEXT NOT NULL,                  -- reminder text / poll question
    created REAL NOT NULL,
    due REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs(due);
"""

# Cheap pre-check so the LLM extraction only runs when a message might be a command.
_COMMAND_RE = re.compile(r"\b(?:remind(?:er|ers)?|poll|polls|vote|timer|alarm|ping (?:me|us|everyone|everybody))\b",
                         re.I)

# ------------------------------------------------------------------ time parsing

_NORMALIZE = [
    (r"\b(?:at )?(?:around|about|approximately|roughly|by)\b", "at"),
    (r"\b(?:an? |one )?hour and a half\b", "90 minutes"),
    (r"\bhalf (?:an? )?hour\b", "30 minutes"),
    (r"\ba quarter (?:of an )?hour\b", "15 minutes"),
    (r"\ba couple(?: of)? (second|minute|hour|day|week)s?\b", r"2 \1s"),
    (r"\ba few (second|minute|hour|day|week)s?\b", r"3 \1s"),
    (r"\b(?:a|an|one) (second|minute|hour|day|week)\b", r"1 \1"),
    (r"\b(\d+) ?(?:secs?|s)\b", r"\1 seconds"),
    (r"\b(\d+) ?(?:mins?|m)\b", r"\1 minutes"),
    (r"\b(\d+) ?(?:hrs?|h)\b", r"\1 hours"),
    (r"\bnoon\b", "12pm"),
    (r"\bmidnight\b", "12am"),
    (r"\b(\d)\s*([ap])\.?m\.?", r"\1\2m"),
    (r"\b(at) (\d{1,2})(?![:\d]| ?[ap]m| ?(?:second|minute|hour|day|week)s?\b)", r"\1 \2:00"),  # "at 8" isn't August
]
_PM_HINT = re.compile(r"\b(tonight|this evening|this afternoon|in the evening|in the afternoon)\b")
_AMPM = re.compile(r"\d\s*[ap]m\b")
_RELATIVE = re.compile(r"\b(?:in|from now|later|ago)\b")
_DATE_WORDS = re.compile(r"\b(?:today|tomorrow|mon|tue|wed|thu|fri|sat|sun|next|jan|feb|mar|apr|may|jun|jul|aug|"
                         r"sep|oct|nov|dec|\d{1,2}/\d{1,2})", re.I)
_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\b")


def parse_when(phrase: str, now: datetime | None = None) -> datetime | None:
    """'in 20 minutes', 'at 5', 'tomorrow at 9am', 'friday 6pm', 'tonight at 8' -> an aware datetime in the
    future, or None. A bare '5:30' means the next 5:30 (am or pm), like a person would take it."""
    now = now or datetime.now().astimezone()
    p = " ".join(phrase.lower().replace(",", " ").split())
    if not p:
        return None
    for pat, rep in _NORMALIZE:
        p = re.sub(pat, rep, p)
    pm = bool(_PM_HINT.search(p))
    p = _PM_HINT.sub(lambda m: "today" if m.group(1) in ("tonight", "this evening", "this afternoon") else "", p)
    p = " ".join(p.split())
    dt = dateparser.parse(p, languages=["en"], settings={
        "PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": True, "RELATIVE_BASE": now.replace(tzinfo=None)})
    if dt is None:
        return None
    dt = dt.astimezone(now.tzinfo) if dt.tzinfo else dt.replace(tzinfo=now.tzinfo)
    has_date = bool(_DATE_WORDS.search(p))
    if not _AMPM.search(p) and not _RELATIVE.search(p) and _CLOCK.search(p) and 1 <= dt.hour <= 11:
        if pm or (has_date and dt.hour <= 6):  # "tonight at 8", "tomorrow at 5" -> pm
            dt += timedelta(hours=12)
        elif not has_date:  # bare clock time: whichever am/pm comes next
            dt = min((c for c in (dt - timedelta(hours=12), dt, dt + timedelta(hours=12)) if c > now), default=dt)
    if dt <= now and not has_date and not _RELATIVE.search(p):
        dt += timedelta(days=1)  # "at 9am" when it's already 10am -> tomorrow
    return dt if dt > now else None


def parse_duration(phrase: str, default_s: float) -> float | None:
    """'5 minutes', 'an hour' -> seconds; '' -> default_s; unreadable -> None."""
    phrase = (phrase or "").strip()
    if not phrase:
        return default_s
    if not re.match(r"(?i)\s*(?:in|for)\b", phrase):
        phrase = f"in {phrase}"
    dt = parse_when(re.sub(r"(?i)^\s*for\b", "in", phrase))
    return (dt - datetime.now().astimezone()).total_seconds() if dt else None


def _local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts).astimezone()


def spoken_time(ts: float) -> str:
    """'5:30 PM (in 20 minutes)' / 'tomorrow at 9:00 AM' - for the LLM to relay."""
    dt, now = _local(ts), datetime.now().astimezone()
    clock = f"{dt.hour % 12 or 12}:{dt:%M %p}"
    secs = ts - time.time()
    if secs < 3600:
        mins = max(1, round(secs / 60))
        n, unit = (mins, "minute") if secs >= 60 else (max(1, round(secs)), "second")
        rel = f"in {n} {unit}{'s' * (n != 1)}"
        return f"{clock} ({rel})"
    if dt.date() == now.date():
        return f"{clock} today"
    if dt.date() == (now + timedelta(days=1)).date():
        return f"tomorrow at {clock}"
    return f"{dt:%A, %B} {dt.day} at {clock}"


# ------------------------------------------------------------------ storage


class JobStore:
    """Tiny synchronous SQLite wrapper, used from the event-loop thread only."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def add(self, **row) -> int:
        row.setdefault("created", time.time())
        if "targets" in row:
            row["targets"] = json.dumps(list(row["targets"]))
        cols = ", ".join(row)
        cur = self.db.execute(f"INSERT INTO jobs ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
        self.db.commit()
        return cur.lastrowid

    def delete(self, job_id: int) -> None:
        self.db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self.db.commit()

    def get(self, job_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def next_due(self) -> float | None:
        row = self.db.execute("SELECT MIN(due) AS due FROM jobs").fetchone()
        return row["due"] if row and row["due"] is not None else None

    def due(self, now: float) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM jobs WHERE due <= ? ORDER BY due", (now,)).fetchall()

    def reminders_for(self, user_id: int) -> list[sqlite3.Row]:
        """Reminders this person set, or that are for them."""
        rows = self.db.execute("SELECT * FROM jobs WHERE kind = 'reminder' ORDER BY due").fetchall()
        return [r for r in rows if r["creator_id"] == user_id or user_id in json.loads(r["targets"])]

    def pending(self) -> dict[str, int]:
        return {r["kind"]: r["n"] for r in self.db.execute("SELECT kind, COUNT(*) AS n FROM jobs GROUP BY kind")}

    def count_by(self, user_id: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM jobs WHERE creator_id = ?", (user_id,)).fetchone()[0]


# ------------------------------------------------------------------ the feature


class Planner:
    def __init__(self, bot: "VoiceBot"):
        self.bot = bot
        self.cfg = bot.cfg.reminders
        self.enabled = bool(self.cfg.enabled)
        self.store = JobStore(self.cfg.db_path)
        self._wake = asyncio.Event()
        self.stats: Counter[str] = Counter()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self.enabled:
            self._task = asyncio.create_task(self._loop(), name="reminders")

    def wants(self, text: str) -> bool:
        """Might this message be a reminder/poll command? (Cheap; the LLM makes the real call.)"""
        return self.enabled and bool(_COMMAND_RE.search(text or ""))

    # ------------------------------------------------------------------ creating

    def add_reminder(self, creator: discord.abc.User, targets: list[int], text: str, due: float, deliver: str,
                     channel) -> int:
        job_id = self.store.add(kind="reminder", guild_id=getattr(getattr(channel, "guild", None), "id", 0),
                                channel_id=getattr(channel, "id", 0), creator_id=creator.id,
                                creator_name=creator.display_name, targets=targets, deliver=deliver,
                                text=text[:500], due=due)
        self._wake.set()
        self.stats["reminders set"] += 1
        log.info("⏰ reminder #%d by %s for %s at %s (%s): %s", job_id, creator.display_name, targets,
                 _local(due).isoformat(timespec="minutes"), deliver, text)
        return job_id

    def check_quota(self, user_id: int) -> str | None:
        if self.store.count_by(user_id) >= int(self.cfg.max_per_user):
            return f"you already have {self.cfg.max_per_user} reminders/polls pending - cancel some with /reminders"
        return None

    def build_poll(self, question: str, options: list[str], seconds: float,
                   multiple: bool = False) -> tuple[discord.Poll, float]:
        """A native poll lasting at least an hour (Discord's minimum) and the time we'll actually close it."""
        seconds = min(max(float(seconds), 60.0), 32 * 86400.0)
        hours = min(max(1, math.ceil(seconds / 3600)), 768)
        poll = discord.Poll(question=question[:300], duration=timedelta(hours=hours), multiple=multiple)
        for opt in options[:10]:
            poll.add_answer(text=opt[:55])
        return poll, time.time() + seconds

    def track_poll(self, message: discord.Message, creator: discord.abc.User, question: str, ends: float) -> None:
        job_id = self.store.add(kind="poll", guild_id=message.guild.id if message.guild else 0,
                                channel_id=message.channel.id, message_id=message.id, creator_id=creator.id,
                                creator_name=creator.display_name, text=question, due=ends)
        self._wake.set()
        self.stats["polls started"] += 1
        log.info("📊 poll #%d by %s until %s: %s", job_id, creator.display_name,
                 _local(ends).isoformat(timespec="minutes"), question)

    def cancel(self, job_id: int) -> None:
        self.store.delete(job_id)
        self._wake.set()

    # ------------------------------------------------------------------ natural language

    async def extract(self, text: str, context: list[str]) -> dict | None:
        convo = "\n".join(" ".join(line.split())[:300] for line in context[-3:])
        answer = await self.bot.llm.complete([
            {"role": "system", "content": self.cfg.extract_prompt.replace("{name}", self.bot.cfg.bot.name).strip()},
            {"role": "user", "content": (f"Earlier:\n{convo}\n\n" if convo else "")
                                        + f"Message:\n{' '.join(text.split())[:500]}\n\nJSON:"},
        ], endpoint=self.cfg.endpoint or None, temperature=0, max_tokens=160, purpose="commands")
        m = re.search(r"\{.*\}", answer, re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict) or data.get("action") in (None, "", "none"):
            return None
        return data

    async def handle(self, text: str, author: discord.abc.User, channel, context: list[str],
                     voice_members: list[discord.Member]) -> str | None:
        """Run a reminder/poll command said in chat or voice. Returns a note for this turn's prompt (so the
        bot confirms or asks for what's missing), or None if the message wasn't a command."""
        try:
            data = await self.extract(text, context)
        except Exception as e:  # noqa: BLE001
            log.warning("Command extraction failed: %s", e)
            return None
        if data is None:
            return None
        log.info("🗒 %s: %s", author.display_name, data)
        who = author.display_name
        try:
            action = data.get("action")
            if action == "remind":
                return await self._nl_remind(data, author, channel, voice_members)
            if action == "poll":
                return await self._nl_poll(data, author, channel)
            if action == "list":
                return self._nl_list(author)
            if action == "cancel":
                return self._nl_cancel(data, author)
        except discord.HTTPException as e:
            log.warning("Command failed: %s", e)
            return f"[{who} asked you to do something in the chat, but Discord refused ({e.text or e}). Tell them.]"
        return None

    async def _nl_remind(self, data: dict, author, channel, voice_members) -> str:
        who = author.display_name
        what = str(data.get("what") or "").strip() or "(no details)"
        when = str(data.get("when") or "").strip()
        due = parse_when(when) if when else None
        if due is None:
            return (f"[{who} wants a reminder about \"{what}\" but it's not clear when (\"{when}\"). "
                    "Ask them when - don't pretend you set it.]")
        if err := self.check_quota(author.id):
            return f"[You can't set that reminder: {err}. Tell {who}.]"
        targets, missing = await self._resolve(data.get("who"), author, voice_members)
        if not targets:
            return f"[{who} wants a reminder for {', '.join(missing)} but you don't know who that is. Ask them.]"
        deliver = "dm" if targets == [author.id] else "channel"
        self.add_reminder(author, targets, what, due.timestamp(), deliver, channel)
        names = self._names(targets, author, voice_members)
        how = "as a DM" if deliver == "dm" else "in the chat, pinging them"
        extra = f" You couldn't find {', '.join(missing)}, so they're not included - mention that." if missing else ""
        return (f"[Done - you set a reminder for {names} at {spoken_time(due.timestamp())}: \"{what}\". "
                f"You'll send it {how}.{extra} Confirm it in a few words.]")

    async def _nl_poll(self, data: dict, author, channel) -> str:
        who = author.display_name
        question = str(data.get("question") or "").strip()
        options = list(dict.fromkeys(str(o).strip() for o in data.get("options") or [] if str(o).strip()))
        if not question or len(options) < 2:
            return f"[{who} wants a poll but the question or the choices aren't clear. Ask them.]"
        if err := self.check_quota(author.id):
            return f"[You can't make that poll: {err}. Tell {who}.]"
        default = float(self.cfg.poll_minutes) * 60
        seconds = parse_duration(str(data.get("duration") or ""), default) or default
        poll, ends = self.build_poll(question, options, seconds)
        msg = await channel.send(f"📊 Poll for {author.mention}", poll=poll,
                                 allowed_mentions=discord.AllowedMentions.none())
        self.track_poll(msg, author, question, ends)
        return (f"[Done - you posted a poll in the text chat: \"{question}\" - options: {', '.join(options[:10])}. "
                f"It closes at {spoken_time(ends)} and then you'll announce the winner. "
                "Tell everyone to vote in the chat, in a few words.]")

    def _nl_list(self, author) -> str:
        rows = self.store.reminders_for(author.id)
        if not rows:
            return f"[{author.display_name} has no reminders pending. Tell them.]"
        items = "; ".join(f"\"{r['text']}\" at {spoken_time(r['due'])}" for r in rows[:8])
        return f"[{author.display_name}'s pending reminders: {items}. Tell them briefly.]"

    def _nl_cancel(self, data: dict, author) -> str:
        rows = [r for r in self.store.reminders_for(author.id) if r["creator_id"] == author.id]
        if not rows:
            return f"[{author.display_name} asked to cancel a reminder but they haven't set any. Tell them.]"
        what = set(re.findall(r"\w+", str(data.get("what") or "").lower()))
        if what:
            row = max(rows, key=lambda r: (len(what & set(re.findall(r"\w+", r["text"].lower()))), r["created"]))
        else:
            row = max(rows, key=lambda r: r["created"])  # "cancel that" -> the latest one
        self.cancel(row["id"])
        log.info("⏰ reminder #%d cancelled by %s", row["id"], author.display_name)
        return f"[Done - you cancelled {author.display_name}'s reminder \"{row['text']}\". Confirm briefly.]"

    async def _resolve(self, who, author, voice_members: list[discord.Member]) -> tuple[list[int], list[str]]:
        """'me' / 'everyone' / ['Bob', 'me'] -> user ids, and the names that matched nobody."""
        names = who if isinstance(who, list) else [who or "me"]
        pool = [m for m in voice_members if not m.bot]
        ids: list[int] = []
        missing: list[str] = []
        for raw in names:
            name = str(raw).strip().lstrip("@").lower()
            if name in ("", "me", "myself", "i") or name in (n.lower() for n in (author.display_name, author.name)):
                ids.append(author.id)
            elif name in ("everyone", "everybody", "all", "us", "we", "the group", "the channel", "here"):
                ids += [m.id for m in pool] or [author.id]
            elif member := await self._find_member(name, author, pool):
                ids.append(member.id)
            else:
                missing.append(str(raw))
        return list(dict.fromkeys(ids)), missing

    @staticmethod
    async def _find_member(name: str, author, pool: list[discord.Member]):
        def names_of(m):
            return [n.lower() for n in (m.display_name, m.name, getattr(m, "global_name", None)) if n]
        for test in (lambda n: n == name, lambda n: n.startswith(name), lambda n: name in n):
            for m in pool:
                if any(test(n) for n in names_of(m)):
                    return m
        guild = getattr(author, "guild", None)
        if guild is not None:
            try:
                found = await guild.query_members(query=name, limit=1)
                return found[0] if found else None
            except (discord.HTTPException, asyncio.TimeoutError):
                return None
        return None

    def _names(self, ids: list[int], author, pool) -> str:
        known = {m.id: m.display_name for m in pool}
        known[author.id] = author.display_name
        out = []
        for uid in ids:
            user = self.bot.get_user(uid)
            out.append(known.get(uid) or (user.display_name if user else f"<@{uid}>"))
        return ", ".join(out)

    # ------------------------------------------------------------------ firing

    async def _loop(self) -> None:
        await self.bot.wait_until_ready()
        while True:
            self._wake.clear()
            now = time.time()
            rows = self.store.due(now)
            for row in rows:
                self.store.delete(row["id"])  # first, so a crash in delivery can't make it fire forever
                try:
                    await (self._fire_poll(row) if row["kind"] == "poll" else self._fire_reminder(row))
                except Exception:  # noqa: BLE001
                    log.exception("Job #%d failed", row["id"])
            if rows:
                continue
            nxt = self.store.next_due()
            timeout = 300.0 if nxt is None else min(300.0, max(0.0, nxt - time.time()))
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except asyncio.TimeoutError:
                pass

    async def _channel(self, channel_id: int):
        if not channel_id:
            return None
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except discord.HTTPException:
                return None
        return channel

    def _voice_listeners(self, guild_id: int) -> tuple[object, dict[int, str]]:
        """The guild's voice session (if the bot is in voice there) and who's in that channel."""
        session = self.bot.sessions.get(guild_id)
        channel = session.vc.channel if session else None
        if not self.cfg.announce_in_voice or channel is None:
            return None, {}
        return session, {m.id: m.display_name for m in channel.members if not m.bot}

    async def _fire_reminder(self, row: sqlite3.Row) -> None:
        targets = json.loads(row["targets"])
        late = time.time() - row["due"] > 120
        by = "" if targets == [row["creator_id"]] else f" from {row['creator_name']}"
        body = (f"⏰ **Reminder**{by}: {row['text']}\n-# set <t:{int(row['created'])}:R>"
                + (f" · sorry, I was offline when it was due (<t:{int(row['due'])}:R>)" if late else ""))
        failed = []
        if row["deliver"] == "dm":
            for uid in targets:
                try:
                    user = self.bot.get_user(uid) or await self.bot.fetch_user(uid)
                    await user.send(body)
                except discord.HTTPException as e:
                    log.warning("⏰ DM to %s failed: %s %s (code %s)", uid, e.status, e.text, e.code)
                    failed.append(uid)  # DMs closed: ping them in the channel instead
        ping = targets if row["deliver"] == "channel" else failed
        if ping and (channel := await self._channel(row["channel_id"])):
            await channel.send(" ".join(f"<@{u}>" for u in ping) + " " + body,
                               allowed_mentions=discord.AllowedMentions(
                                   users=[discord.Object(u) for u in ping], everyone=False, roles=False))
        self.stats["reminders delivered"] += 1
        log.info("⏰ reminder #%d delivered (%s%s): %s", row["id"], row["deliver"],
                 f", {len(failed)} DM(s) failed" if failed else "", row["text"])

        session, here = self._voice_listeners(row["guild_id"])
        names = [here[u] for u in targets if u in here]
        if session and names:
            set_by = "" if targets == [row["creator_id"]] else f", set by {row['creator_name']}"
            session.post_event(f"A reminder is due right now for {', '.join(names)}{set_by}: \"{row['text']}\". "
                               "Tell them, in a few words", respond=True)

    async def _fire_poll(self, row: sqlite3.Row) -> None:
        channel = await self._channel(row["channel_id"])
        if channel is None:
            return
        try:
            msg = await channel.fetch_message(row["message_id"])
        except discord.HTTPException:
            return  # deleted
        if msg.poll is None:
            return
        if not msg.poll.is_finalized():
            try:
                await msg.end_poll()
            except discord.HTTPException:
                pass  # already over (it ran its full native duration)
            await asyncio.sleep(2)  # vote counts are finalized shortly after it ends
            msg = await channel.fetch_message(row["message_id"])
        answers = sorted(msg.poll.answers, key=lambda a: -a.vote_count)
        total = sum(a.vote_count for a in answers)
        summary = ", ".join(f"{a.text}: {a.vote_count}" for a in answers)
        self.stats["polls closed"] += 1
        log.info("📊 poll #%d closed: %s -> %s", row["id"], row["text"], summary)
        session, here = self._voice_listeners(row["guild_id"])
        if session and here:
            result = (f"Results: {summary} ({total} votes)." if total
                      else "Nobody voted.")
            session.post_event(f"The poll \"{row['text']}\" just closed. {result} Announce the result in a few words",
                               respond=True)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
        self.store.db.close()
