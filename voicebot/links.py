"""Opt-in account linking between Discord and Fluxer: "this Discord account and this Fluxer account are the same
person", so each bot also knows what the other remembers about them. Everything else stays separate.

Flow: /link on Discord (or !link on Fluxer) gives a 6-digit code, valid 10 minutes; entering it on the *other*
platform (!link 123456 / /link code:123456) links the two - that proves the same person controls both accounts.
/unlink or !unlink (either side) undoes it; /forget on either side unlinks too.

data/links.db is the one file both bots share on purpose, and it only holds the pair of ids and names.
While linked, each bot's roster line for that person gets the other bot's notes on them (read-only), unless they
opted out of profiling on either platform.
"""
from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from pathlib import Path

PLATFORMS = ("discord", "fluxer")
CODE_TTL = 600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS links (
    discord_id INTEGER NOT NULL UNIQUE,
    fluxer_id INTEGER NOT NULL UNIQUE,
    discord_name TEXT NOT NULL DEFAULT '',
    fluxer_name TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL
);
"""


def other(platform: str) -> str:
    return "fluxer" if platform == "discord" else "discord"


class Links:
    def __init__(self, cfg):
        c = cfg.get("links") or {}
        self.enabled = bool(c.get("enabled", True))
        path = str(c.get("db_path") or "data/links.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self._lock = threading.Lock()
        self._codes: dict[str, tuple[str, int, str, float]] = {}  # code -> (platform, user id, name, expires)
        self.profile_stores: dict[str, object] = {}  # platform -> ProfileStore (registered by each bot)

    # ------------------------------------------------------------------ linking

    def start(self, platform: str, user_id: int, name: str) -> str:
        """A fresh code for this person (any earlier code of theirs stops working)."""
        now = time.time()
        with self._lock:
            self._codes = {c: v for c, v in self._codes.items() if v[3] > now and not (v[0] == platform and v[1] == user_id)}
            code = f"{secrets.randbelow(10**6):06d}"
            while code in self._codes:
                code = f"{secrets.randbelow(10**6):06d}"
            self._codes[code] = (platform, user_id, name, now + CODE_TTL)
        return code

    def complete(self, platform: str, user_id: int, name: str, code: str) -> tuple[bool, str]:
        """Enter a code made on the other platform. (ok, message for the user)."""
        code = "".join(ch for ch in code if ch.isdigit())
        with self._lock:
            got = self._codes.get(code)
            if got is None or got[3] < time.time():
                self._codes.pop(code, None)
                return False, "That code didn't work (wrong, or older than 10 minutes). Make a new one."
            if got[0] == platform:
                return False, f"That code is from {platform.title()} too - enter it on {other(platform).title()}."
            del self._codes[code]
        ids = {platform: (user_id, name), got[0]: (got[1], got[2])}
        (did, dname), (fid, fname) = ids["discord"], ids["fluxer"]
        with self._lock:
            self.db.execute("DELETE FROM links WHERE discord_id = ? OR fluxer_id = ?", (did, fid))
            self.db.execute("INSERT INTO links (discord_id, fluxer_id, discord_name, fluxer_name, created) "
                            "VALUES (?, ?, ?, ?, ?)", (did, fid, dname, fname, time.time()))
            self.db.commit()
        return True, f"Linked: **{dname}** on Discord = **{fname}** on Fluxer. Unlink any time with unlink."

    def linked(self, platform: str, user_id: int) -> tuple[int, str] | None:
        """(the other account's id, its name), or None."""
        me, you = ("discord_id", "fluxer") if platform == "discord" else ("fluxer_id", "discord")
        with self._lock:
            row = self.db.execute(f"SELECT * FROM links WHERE {me} = ?", (user_id,)).fetchone()
        return (row[f"{you}_id"], row[f"{you}_name"]) if row else None

    def unlink(self, platform: str, user_id: int) -> bool:
        col = "discord_id" if platform == "discord" else "fluxer_id"
        with self._lock:
            cur = self.db.execute(f"DELETE FROM links WHERE {col} = ?", (user_id,))
            self.db.commit()
        return cur.rowcount > 0

    def all(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute("SELECT * FROM links ORDER BY created DESC").fetchall()

    # ------------------------------------------------------------------ prompt text

    def note(self, platform: str, user_id: int, max_chars: int = 400) -> str:
        """The extra roster line for a linked person: who they are over there + what that bot remembers."""
        if not self.enabled:
            return ""
        got = self.linked(platform, user_id)
        if got is None:
            return ""
        other_id, other_name = got
        there = other(platform)
        mine = self.profile_stores.get(platform)
        theirs = self.profile_stores.get(there)
        row_here = mine.get(user_id) if mine is not None else None
        row = theirs.get(other_id) if theirs is not None else None
        head = f"Same person as {other_name} on {there.title()}."
        if (row_here is not None and row_here["opted_out"]) or row is None or row["opted_out"] or not row["profile"]:
            return head
        items = [ln.strip().lstrip("-•*").strip() for ln in row["profile"].splitlines() if ln.strip()]
        text = "; ".join(i for i in items if i)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(";", 1)[0] + " …"
        return f"{head} From there you also remember: {text}"
