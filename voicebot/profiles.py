"""Who the bot is talking to (identity + roles) and long-term profiles of the people it talks with.

Profiles live in a local SQLite file. Each line a person says (plus the bot's replies to them) is
queued as 'pending'. Once someone has said enough - or went quiet a while ago - and nobody is
talking, the LLM rewrites that person's profile from their old notes + the pending lines.
Updates never compete with a live conversation: any new activity cancels an in-flight update
(the pending lines are kept and it's retried later).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.profiles")

MAX_PENDING_PER_USER = 80

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL DEFAULT '',
    display_name    TEXT    NOT NULL DEFAULT '',
    roles           TEXT    NOT NULL DEFAULT '[]',
    first_seen      REAL    NOT NULL,
    last_seen       REAL    NOT NULL,
    messages        INTEGER NOT NULL DEFAULT 0,
    profile         TEXT    NOT NULL DEFAULT '',
    profile_updated REAL,
    opted_out       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pending (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ts      REAL    NOT NULL,
    line    TEXT    NOT NULL,
    own     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS pending_user ON pending(user_id, id);
"""

UPDATE_PROMPT = """You keep private memory notes for {bot}, a member of a Discord server, about the people {bot} talks with.
Update the notes about {who} using the new conversation.

Rules:
- Keep everything in the current notes that is still true. Fix or drop things the conversation shows are outdated.
- Add what {short} said or revealed: who they are, what they do, interests, opinions, likes and dislikes, what's going on in their life, people they mention, their vibe and sense of humor, running jokes with {bot}, how they like to be talked to.
- Only use what the conversation supports. Don't guess. Things other people say count only if they're clearly about {short}.
- Never store passwords, addresses, phone numbers, financial details or other secrets.
- Short bullet points starting with "- ", most important first, at most {max_chars} characters total.
- Output only the bullet points."""


class ProfileStore:
    """Tiny synchronous SQLite wrapper. Only used from the event-loop thread; queries take ~ms."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(users)")}
        for col in ("avatar_key", "avatar_desc"):  # added after the first release
            if col not in cols:
                self.db.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        self.db.commit()

    def upsert(self, user_id: int, username: str, display_name: str, roles: list[str], own: bool) -> None:
        now = time.time()
        self.db.execute(
            """INSERT INTO users (user_id, username, display_name, roles, first_seen, last_seen, messages)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 username     = CASE WHEN excluded.username != '' THEN excluded.username ELSE users.username END,
                 display_name = CASE WHEN excluded.display_name != '' THEN excluded.display_name ELSE users.display_name END,
                 roles        = CASE WHEN excluded.roles != '[]' THEN excluded.roles ELSE users.roles END,
                 last_seen    = excluded.last_seen,
                 messages     = users.messages + excluded.messages""",
            (user_id, username, display_name, json.dumps(roles), now, now, 1 if own else 0),
        )
        self.db.commit()

    def get(self, user_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()

    def add_pending(self, user_id: int, line: str, own: bool) -> None:
        self.db.execute("INSERT INTO pending (user_id, ts, line, own) VALUES (?, ?, ?, ?)",
                        (user_id, time.time(), line, int(own)))
        self.db.execute(
            """DELETE FROM pending WHERE user_id = ? AND id NOT IN
               (SELECT id FROM pending WHERE user_id = ? ORDER BY id DESC LIMIT ?)""",
            (user_id, user_id, MAX_PENDING_PER_USER),
        )
        self.db.commit()

    def pending(self, user_id: int) -> list[sqlite3.Row]:
        return self.db.execute("SELECT id, line FROM pending WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()

    def pending_lines(self, user_id: int) -> list[sqlite3.Row]:
        """With timestamps and whose line it is, for the dashboard."""
        return self.db.execute("SELECT ts, line, own FROM pending WHERE user_id = ? ORDER BY id",
                               (user_id,)).fetchall()

    def all_users(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT u.*, (SELECT COUNT(*) FROM pending p WHERE p.user_id = u.user_id) AS pending
               FROM users u ORDER BY u.last_seen DESC""").fetchall()

    def pending_count(self, user_id: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM pending WHERE user_id = ?", (user_id,)).fetchone()[0]

    def clear_pending(self, user_id: int, up_to_id: int) -> None:
        self.db.execute("DELETE FROM pending WHERE user_id = ? AND id <= ?", (user_id, up_to_id))
        self.db.commit()

    def set_profile(self, user_id: int, text: str) -> None:
        self.db.execute("UPDATE users SET profile = ?, profile_updated = ? WHERE user_id = ?",
                        (text, time.time(), user_id))
        self.db.commit()

    def set_avatar(self, user_id: int, key: str, desc: str) -> None:
        now = time.time()
        self.db.execute(
            """INSERT INTO users (user_id, first_seen, last_seen, avatar_key, avatar_desc) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET avatar_key = excluded.avatar_key, avatar_desc = excluded.avatar_desc""",
            (user_id, now, now, key, desc),
        )
        self.db.commit()

    def forget(self, user_id: int) -> None:
        self.db.execute("DELETE FROM pending WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self.db.commit()

    def set_opted_out(self, user_id: int, opted_out: bool) -> None:
        now = time.time()
        self.db.execute(
            """INSERT INTO users (user_id, first_seen, last_seen, opted_out) VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET opted_out = excluded.opted_out""",
            (user_id, now, now, int(opted_out)),
        )
        if opted_out:  # opting out also wipes what was collected
            self.db.execute("UPDATE users SET profile = '', profile_updated = NULL WHERE user_id = ?", (user_id,))
            self.db.execute("DELETE FROM pending WHERE user_id = ?", (user_id,))
        self.db.commit()

    def due(self, min_own: int, stale_before: float) -> list[int]:
        """Users with enough new lines, or with any new lines from a conversation that ended a while ago."""
        rows = self.db.execute(
            """SELECT p.user_id, SUM(p.own) AS own_count, MAX(p.ts) AS last_ts
               FROM pending p JOIN users u ON u.user_id = p.user_id
               WHERE u.opted_out = 0
               GROUP BY p.user_id
               HAVING own_count >= ? OR (own_count >= 1 AND last_ts < ?)
               ORDER BY own_count DESC""",
            (min_own, stale_before),
        ).fetchall()
        return [r["user_id"] for r in rows]


# ---------------------------------------------------------------------- identity / prompt text

def member_roles(member) -> list[str]:
    """Role names, highest first, without @everyone. Empty for DMs / plain Users."""
    roles = getattr(member, "roles", None) or []
    return [r.name for r in reversed(roles) if not r.is_default()]


_VERBS = {
    discord.ActivityType.playing: "playing",
    discord.ActivityType.streaming: "streaming",
    discord.ActivityType.listening: "listening to",
    discord.ActivityType.watching: "watching",
    discord.ActivityType.competing: "competing in",
}


def member_presence(member) -> list[str]:
    """What someone is doing right now: games, music, streams, custom status, and camera/screen share in voice.
    Needs the presences intent; without it activities are just empty."""
    out = []
    for a in getattr(member, "activities", None) or ():
        if isinstance(a, discord.Spotify):
            out.append(f'listening to "{a.title}" by {a.artist} on Spotify')
        elif isinstance(a, discord.CustomActivity):
            if a.name:
                out.append(f'custom status "{a.name}"')
        elif isinstance(a, discord.Streaming):
            out.append(f"streaming {a.game or a.name or 'something'}" + (f" on {a.platform}" if a.platform else ""))
        elif isinstance(a, discord.Game):
            out.append(f"playing {a.name}")
        elif isinstance(a, discord.Activity) and a.name:
            detail = ", ".join(x for x in (a.details, a.state) if x)
            out.append(f"{_VERBS.get(a.type, 'using')} {a.name}" + (f" - {detail}" if detail else ""))
    voice = getattr(member, "voice", None)
    if voice:
        if voice.self_stream:
            out.append("sharing their screen in the call")
        if voice.self_video:
            out.append("has their camera on")
        if voice.self_deaf:
            out.append("deafened (can't hear you)")
        elif voice.self_mute:
            out.append("muted")
    return out


def _familiarity(messages: int) -> str:
    # Coarse buckets on purpose: this text sits in the system prompt, and changing it every turn
    # would defeat the LLM server's prompt cache (slower replies).
    if messages <= 0:
        return "you haven't talked with them before"
    if messages < 10:
        return "you've only just met"
    if messages < 100:
        return "you're getting to know them"
    return "a regular you know well"


def _compact(profile: str, max_chars: int) -> str:
    items = [line.strip().lstrip("-•*").strip() for line in profile.splitlines() if line.strip()]
    text = "; ".join(i for i in items if i)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(";", 1)[0] + " …"


def _clean_notes(text: str, max_chars: int) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    bullets = [line for line in lines if line[:1] in "-•*"]
    if bullets:  # drop any "Here are the updated notes:" chatter
        lines = bullets
    out, total = [], 0
    for line in lines:
        line = "- " + line.lstrip("-•*").strip()
        if total + len(line) + 1 > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


class ProfileManager:
    def __init__(self, bot: "VoiceBot"):
        self.bot = bot
        self.cfg = bot.cfg.profiles
        self.enabled = bool(self.cfg.enabled)
        self.store = ProfileStore(self.cfg.db_path)
        self.linked = None  # callable(user id) -> str: the linked account on the other platform (links.py), or None
        self._task: asyncio.Task | None = None
        self._runner: asyncio.Task | None = None
        self._last_activity = 0.0

    def start(self) -> None:
        if self.enabled:
            self._runner = asyncio.create_task(self._loop(), name="profile-updater")

    @property
    def last_activity(self) -> float:
        return self._last_activity

    def activity(self, cancel: bool = True) -> None:
        """Called on live conversation. Pauses background updates so the LLM is free for replies."""
        self._last_activity = time.monotonic()
        if cancel:
            self.bot.lore.pause()  # lore extraction waits for idle time too
        if cancel and self._task and not self._task.done():
            self._task.cancel()
            log.debug("Profile update paused for live conversation")

    # ------------------------------------------------------------------ recording

    def observe(self, user_id: int, name: str, line: str, member=None) -> None:
        """Something this person said to/around the bot."""
        if not self.enabled:
            return
        row = self.store.get(user_id)
        if row and row["opted_out"]:
            return
        self.store.upsert(user_id, getattr(member, "name", "") or "", name, member_roles(member), own=True)
        self.store.add_pending(user_id, line, own=True)

    def observe_reply(self, user_ids, line: str) -> None:
        """The bot's reply, as context for the people it was answering."""
        if not self.enabled:
            return
        for uid in set(user_ids):
            row = self.store.get(uid)
            if row and not row["opted_out"]:
                self.store.add_pending(uid, line, own=False)

    # ------------------------------------------------------------------ prompt text

    def self_note(self, guild) -> str:
        """The bot's own roles in this server, so "what roles do you have?" gets a real answer."""
        me = getattr(guild, "me", None)
        roles = member_roles(me)[: int(self.cfg.max_roles)] if me is not None and self.cfg.include_roles else []
        return f"Your own roles in this server: {', '.join(roles)}." if roles else ""

    def describe(self, member) -> str:
        """One roster entry: identity, roles, familiarity and (if any) what the bot remembers."""
        row = self.store.get(member.id)
        head = f"- {member.name}"
        extras = ["your creator"] if member.id in self.bot.cfg.bot.creator_ids else []
        guild = getattr(member, "guild", None)
        if guild is not None and guild.owner_id == member.id:
            extras.append("server owner")
        elif getattr(getattr(member, "guild_permissions", None), "administrator", False):
            extras.append("admin")
        if self.cfg.include_roles:
            roles = member_roles(member)[: int(self.cfg.max_roles)]
            if roles:
                extras.append("roles: " + ", ".join(roles))
        if self.enabled:
            extras.append(_familiarity(row["messages"] if row else 0))
        text = head + (" - " + "; ".join(extras) if extras else "")
        avatar = self.bot.vision.avatar_note(member, row)
        if avatar:
            text += "\n  Their profile picture: " + avatar
        if self.enabled and row and row["profile"] and not row["opted_out"]:
            text += "\n  What you remember about them: " + _compact(row["profile"], int(self.cfg.max_prompt_chars))
        if self.linked is not None and (also := self.linked(member.id)):
            text += "\n  " + also
        return text

    def roster(self, members) -> str:
        members = sorted(members, key=lambda m: getattr(m, "display_name", m.name).lower())
        return "\n".join(self.describe(m) for m in members[: int(self.cfg.max_people)])

    def presence_note(self, members) -> str:
        """What people are doing right now, as a [bracketed] line. Added to the conversation only when it
        changes, so no timers in it (it would change every minute)."""
        if not self.cfg.include_activities:
            return ""
        members = sorted(members, key=lambda m: getattr(m, "display_name", m.name).lower())
        lines = [f"{getattr(m, 'display_name', m.name)}: {'; '.join(p)}"
                 for m in members[: int(self.cfg.max_people)] if (p := member_presence(m))]
        if not lines:
            return ""
        return "[Discord shows what people are up to right now - " + " | ".join(lines) + "]"

    # ------------------------------------------------------------------ background updates

    async def _loop(self) -> None:
        idle_s = float(self.cfg.idle_s)
        while True:
            await asyncio.sleep(10)
            if time.monotonic() - self._last_activity < idle_s:
                continue
            due = self.store.due(int(self.cfg.update_after_messages), time.time() - float(self.cfg.stale_after_s))
            for user_id in due:
                if time.monotonic() - self._last_activity < idle_s:
                    break
                task = self._task = asyncio.create_task(self._update(user_id))
                await asyncio.wait({task})
                if task.cancelled():
                    break
                if task.exception():
                    log.warning("Profile update failed: %r - retrying later", task.exception())
                    await asyncio.sleep(120)
                    break

    async def _update(self, user_id: int) -> None:
        row = self.store.get(user_id)
        pending = self.store.pending(user_id)
        if not row or not pending:
            return
        max_chars = int(self.cfg.max_profile_chars)
        short = row["display_name"] or row["username"] or "this person"
        who = f"{short} (@{row['username']})" if row["username"] else short
        roles = json.loads(row["roles"] or "[]")

        system = UPDATE_PROMPT.format(bot=self.bot.cfg.bot.name, who=who, short=short, max_chars=max_chars)
        user = (
            f"About: {who}" + (f" - roles: {', '.join(roles)}" if roles else "") + "\n\n"
            f"Current notes:\n{row['profile'] or '(nothing yet)'}\n\n"
            "New conversation:\n" + "\n".join(p["line"] for p in pending)
        )
        t0 = time.perf_counter()
        text = await self.bot.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            endpoint=self.cfg.endpoint or None, temperature=0.3, purpose="profiles",
        )
        notes = _clean_notes(text, max_chars)
        if notes:
            self.store.set_profile(user_id, notes)
        self.store.clear_pending(user_id, pending[-1]["id"])
        log.info("Updated profile for %s from %d new lines (%.1fs)", short, len(pending), time.perf_counter() - t0)

    # ------------------------------------------------------------------ for /profile

    def summary(self, user_id: int) -> str:
        row = self.store.get(user_id)
        if not row:
            return "Nothing stored."
        fmt = "%b %d, %Y"
        lines = [
            f"**{row['username'] or row['display_name']}**",
            f"Roles: {', '.join(json.loads(row['roles'] or '[]')) or '-'}",
            f"First seen {datetime.fromtimestamp(row['first_seen']):{fmt}} · last seen "
            f"{datetime.fromtimestamp(row['last_seen']):{fmt}} · {row['messages']} messages",
            f"Profiling: {'**off**' if row['opted_out'] else 'on'} · {self.store.pending_count(user_id)} lines not yet summarized",
            f"Profile picture: {row['avatar_desc'] or '-'}",
            "",
            "**What I remember:**",
            row["profile"] or "_nothing yet_",
        ]
        return "\n".join(lines)
