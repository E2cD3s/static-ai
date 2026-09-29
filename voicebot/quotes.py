"""The quote book: "Static, quote that" (or a bare "quote that") saves the last thing someone said in voice;
/quote (Discord) / !quote (Fluxer) pulls up a random one, searches, or saves a replied-to text message.
"Static, give us a quote" reads a random one out. Each platform has its own book (quotes.db_path; Fluxer's is
under fluxer.data_dir). /forget and opt-out don't touch quotes: they're things said out loud in the group,
kept on purpose - an admin can delete one with /quote delete.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,       -- who said it (0 = the bot)
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    said_at REAL NOT NULL,
    saved_by TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'voice'   -- voice | text
);
CREATE INDEX IF NOT EXISTS quotes_guild ON quotes(guild_id);
"""

# "quote that" / "quote him" / "quote what riley said" / "save that quote" / "put that in the quote book"
_SAVE = re.compile(r"\bquote\s+(?:that|this|it|him|her|them|me)\b|\bquote\s+what\s+(?P<who>\w+)\s+(?:just\s+)?said\b"
                   r"|\b(?:save|add|put)\s+(?:that|this)\s+(?:to|in|into)?\s*(?:the\s+)?quote(?:\s*book)?\b"
                   r"|\bthat'?s\s+(?:a|one)\s+for\s+the\s+quote\s*book\b", re.I)
_READ = re.compile(r"\b(?:give|tell|read)\s+(?:us|me)\s+a\s+(?:random\s+)?quote\b|\brandom\s+quote\b"
                   r"|\bquote\s+of\s+the\s+day\b|\bread\s+(?:us\s+|me\s+)?(?:something\s+)?from\s+the\s+quote\s*book\b", re.I)


def save_request(text: str) -> tuple[bool, str | None] | None:
    """(is a save request, who) - who = a name ("quote what riley said"), "me", or None (the last line)."""
    m = _SAVE.search(text)
    if not m:
        return None
    who = m.group("who") or ("me" if re.search(r"\bquote\s+me\b", text, re.I) else None)
    return True, who


def read_request(text: str) -> bool:
    return bool(_READ.search(text))


class QuoteBook:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self._lock = threading.Lock()

    def add(self, guild_id: int, user_id: int, name: str, text: str, said_at: float | None = None,
            saved_by: str = "", source: str = "voice") -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO quotes (guild_id, user_id, name, text, said_at, saved_by, source) VALUES (?,?,?,?,?,?,?)",
                (guild_id, user_id, name, text.strip(), said_at or time.time(), saved_by, source))
            self.db.commit()
            return int(cur.lastrowid)

    def get(self, quote_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute("SELECT * FROM quotes WHERE id = ?", (quote_id,)).fetchone()

    def random(self, guild_id: int, search: str = "") -> sqlite3.Row | None:
        q, args = "SELECT * FROM quotes WHERE guild_id = ?", [guild_id]
        if search:
            q += " AND (text LIKE ? OR name LIKE ?)"
            args += [f"%{search}%", f"%{search}%"]
        with self._lock:
            return self.db.execute(q + " ORDER BY RANDOM() LIMIT 1", args).fetchone()

    def search(self, guild_id: int, search: str, limit: int = 10) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute("SELECT * FROM quotes WHERE guild_id = ? AND (text LIKE ? OR name LIKE ?) "
                                   "ORDER BY id DESC LIMIT ?", (guild_id, f"%{search}%", f"%{search}%", limit)).fetchall()

    def count(self, guild_id: int | None = None) -> int:
        with self._lock:
            if guild_id is None:
                return self.db.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
            return self.db.execute("SELECT COUNT(*) FROM quotes WHERE guild_id = ?", (guild_id,)).fetchone()[0]

    def delete(self, quote_id: int) -> bool:
        with self._lock:
            cur = self.db.execute("DELETE FROM quotes WHERE id = ?", (quote_id,))
            self.db.commit()
            return cur.rowcount > 0

    def exists(self, guild_id: int, text: str) -> int | None:
        with self._lock:
            row = self.db.execute("SELECT id FROM quotes WHERE guild_id = ? AND text = ?", (guild_id, text.strip())).fetchone()
        return row["id"] if row else None


def show(row) -> str:
    """One quote as a chat line."""
    when = time.strftime("%b %-d, %Y", time.localtime(row["said_at"]))
    return f"💬 **#{row['id']}** “{row['text']}” — **{row['name']}**, {when}"
