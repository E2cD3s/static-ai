"""Server lore: moments worth remembering across calls ("jordan built the Mob Masher and it took Dorothy's head").

Profiles remember each person; this remembers the group - inside jokes, things people built, won or failed at,
running bits. Lines from voice and text chat collect per server; once there's enough and the bot has been idle
a while, a background LLM call picks out 0-3 moments (one line each) and they're stored in data/lore.db with a
small sentence embedding (all-MiniLM-L6-v2, int8 ONNX on CPU, ~17ms). Each turn, the lines just said are
embedded (names stripped - otherwise the speaker's name decides the match) and the closest memories above a
similarity bar go on that turn only, never into the system prompt. /forget and opting out delete every memory
that involves the person.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.lore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS lore (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    created REAL NOT NULL,
    text TEXT NOT NULL,
    people TEXT NOT NULL DEFAULT '[]',   -- user ids involved (JSON), for /forget
    vec BLOB NOT NULL,                   -- float32 unit vector
    recalled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS lore_guild ON lore(guild_id);
"""


class LoreStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[list[sqlite3.Row], np.ndarray]] = {}

    def add(self, guild_id: int, text: str, people: list[int], vec: np.ndarray) -> None:
        with self._lock:
            self.db.execute("INSERT INTO lore (guild_id, created, text, people, vec) VALUES (?, ?, ?, ?, ?)",
                            (guild_id, time.time(), text, json.dumps(people), vec.astype(np.float32).tobytes()))
            self.db.commit()
            self._cache.pop(guild_id, None)

    def entries(self, guild_id: int) -> tuple[list[sqlite3.Row], np.ndarray]:
        with self._lock:
            if guild_id not in self._cache:
                rows = self.db.execute("SELECT * FROM lore WHERE guild_id = ? ORDER BY id", (guild_id,)).fetchall()
                vecs = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
                self._cache[guild_id] = (rows, vecs)
            return self._cache[guild_id]

    def recalled(self, ids: list[int]) -> None:
        with self._lock:
            self.db.executemany("UPDATE lore SET recalled = recalled + 1 WHERE id = ?", [(i,) for i in ids])
            self.db.commit()

    def about(self, user_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return [r for r in self.db.execute("SELECT id, guild_id, created, text, people FROM lore ORDER BY id")
                    if user_id in json.loads(r["people"])]

    def delete(self, lore_id: int) -> None:
        with self._lock:
            self.db.execute("DELETE FROM lore WHERE id = ?", (lore_id,))
            self.db.commit()
            self._cache.clear()

    def forget(self, user_id: int) -> int:
        ids = [r["id"] for r in self.about(user_id)]
        with self._lock:
            self.db.executemany("DELETE FROM lore WHERE id = ?", [(i,) for i in ids])
            self.db.commit()
            self._cache.clear()
        return len(ids)

    def count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT COUNT(*) FROM lore").fetchone()[0]


class Embedder:
    """all-MiniLM-L6-v2, int8 ONNX (AVX2) on CPU. Loaded once; call from one thread at a time."""

    def __init__(self, cfg):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def get(f: str) -> str:  # local copy first: no network round-trips on every start
            try:
                return hf_hub_download(cfg.model_repo, f, cache_dir=cfg.cache_dir, local_files_only=True)
            except Exception:  # noqa: BLE001 - not downloaded yet
                return hf_hub_download(cfg.model_repo, f, cache_dir=cfg.cache_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(get(cfg.model_file), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(get("tokenizer.json"))
        self.tok.enable_truncation(128)

    def embed(self, text: str) -> np.ndarray:
        e = self.tok.encode(text)
        ids, mask = np.array([e.ids], dtype=np.int64), np.array([e.attention_mask], dtype=np.int64)
        out = self.sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0][0]
        v = (out * mask[0][:, None]).sum(0) / max(mask.sum(), 1)
        return (v / (np.linalg.norm(v) or 1)).astype(np.float32)


def _ago(ts: float) -> str:
    days = (time.time() - ts) / 86400
    if days < 1:
        return "earlier today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{int(days)} days ago"
    return f"{int(days // 7)} weeks ago" if days < 60 else f"{int(days // 30)} months ago"


class Lore:
    def __init__(self, bot: "VoiceBot"):
        self.bot = bot
        self.cfg = bot.cfg.lore
        self.enabled = bool(self.cfg.enabled)
        self.store = LoreStore(self.cfg.db_path)
        self.model: Embedder | None = None
        self._pending: dict[int, list[tuple[float, int, str]]] = defaultdict(list)  # guild -> (when, user id, line)
        self._recent: dict[int, dict[int, float]] = defaultdict(dict)  # guild -> lore id -> when last recalled
        self._task: asyncio.Task | None = None
        self._runner: asyncio.Task | None = None

    def load(self) -> None:
        """Blocking (downloads ~23 MB on first run) - call from a thread at startup."""
        if not self.enabled:
            return
        try:
            self.model = Embedder(self.cfg)
            self.model.embed("warm up")
            log.info("Lore memory ready (%d memories stored)", self.store.count())
        except Exception as e:  # noqa: BLE001
            log.warning("Lore memory off - the embedding model didn't load: %s", e)

    def start(self) -> None:
        if self.enabled:
            self._runner = asyncio.create_task(self._loop(), name="lore-keeper")

    # ------------------------------------------------------------------ collecting

    def observe(self, guild, user_id: int, line: str) -> None:
        """A line said in a server (user_id 0 = the bot itself). Opted-out people's lines are never kept."""
        if not self.enabled or self.model is None or guild is None:
            return
        if user_id:
            row = self.bot.profiles.store.get(user_id)
            if row and row["opted_out"]:
                return
        buf = self._pending[guild.id]
        buf.append((time.time(), user_id, " ".join(line.split())[:300]))
        del buf[: -int(self.cfg.max_pending)]

    # ------------------------------------------------------------------ recalling

    async def recall(self, guild_id: int | None, text: str, names: list[str]) -> str | None:
        """Memories related to what was just said, as a note for this turn (or None)."""
        if not self.enabled or self.model is None or not guild_id or not text.strip():
            return None
        rows, vecs = self.store.entries(guild_id)
        if not rows:
            return None
        query = re.sub(r"(?m)^\s*\[.*$", " ", text)  # [bracketed] notes aren't what they said
        if names:
            query = re.sub(r"\b(?:%s)\b:?" % "|".join(re.escape(n) for n in names if n), " ", query, flags=re.I)
        if len(re.findall(r"\w+", query)) < 3:
            return None  # "lol" / "what?" match everything a little
        vec = await asyncio.get_running_loop().run_in_executor(self.bot.mood_executor, self.model.embed, query)
        scores = vecs @ vec
        now = time.time()
        recent = self._recent[guild_id]
        cooldown = float(self.cfg.recall_cooldown_min) * 60
        # Recalled in the last minute = the same turn again (a speculative reply redone): still fine to use.
        picks = [i for i in np.argsort(-scores)[: int(self.cfg.max_recall) * 3]
                 if scores[i] >= float(self.cfg.min_similarity)
                 and not 60 < now - recent.get(rows[i]["id"], -1e12) < cooldown]
        picks = picks[: int(self.cfg.max_recall)]
        if not picks:
            return None
        for i in picks:
            recent.setdefault(rows[i]["id"], now)
            if now - recent[rows[i]["id"]] > 60:
                recent[rows[i]["id"]] = now
        self.store.recalled([rows[i]["id"] for i in picks])
        found = [f"- ({_ago(rows[i]['created'])}) {rows[i]['text']}" for i in picks]
        log.info("📜 recalled: %s", " | ".join(f"{rows[i]['text'][:60]} ({scores[i]:.2f})" for i in picks))
        return ("[Things you remember from before - bring one up only if it really fits the moment (a callback "
                "can be funny), never force it or list them:\n" + "\n".join(found) + "]")

    # ------------------------------------------------------------------ writing

    async def _loop(self) -> None:
        idle_s = float(self.cfg.idle_s)
        while True:
            await asyncio.sleep(15)
            if self.model is None or time.monotonic() - self.bot.profiles.last_activity < idle_s:
                continue
            now = time.time()
            for gid, buf in list(self._pending.items()):
                people = sum(1 for _, uid, _ in buf if uid)
                stale = buf and now - buf[-1][0] > float(self.cfg.stale_after_s)
                if people < int(self.cfg.min_lines) and not (stale and people >= int(self.cfg.min_lines_stale)):
                    if stale:
                        buf.clear()  # a few lines, long ago: not worth an LLM call
                    continue
                if time.monotonic() - self.bot.profiles.last_activity < idle_s:
                    break
                taken = len(buf)
                task = self._task = asyncio.create_task(self._extract(gid, list(buf)))
                await asyncio.wait({task})
                if task.cancelled():
                    break
                del buf[:taken]
                if task.exception():
                    log.warning("Lore extraction failed: %r", task.exception())

    def pause(self) -> None:
        """Live conversation: the LLM is needed for replies."""
        if self._task and not self._task.done():
            self._task.cancel()

    def _people(self, guild_id: int, text: str) -> list[int]:
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return []
        low = text.lower()
        out = []
        for m in guild.members:
            names = {m.name.lower(), m.display_name.lower()}
            if not m.bot and any(len(n) >= 3 and re.search(rf"\b{re.escape(n)}\b", low) for n in names):
                out.append(m.id)
        return out

    async def _extract(self, guild_id: int, lines: list[tuple[float, int, str]]) -> None:
        t0 = time.perf_counter()
        name = self.bot.cfg.bot.name
        transcript = "\n".join(line for _, _, line in lines)
        text = await self.bot.llm.complete([
            {"role": "system", "content": self.cfg.extract_prompt.replace("{name}", name)},
            {"role": "user", "content": f"Transcript:\n{transcript}\n\nMoments worth remembering (or NONE):"},
        ], endpoint=self.cfg.endpoint or None, temperature=0.2, max_tokens=200, purpose="lore")
        kept = []
        for raw in text.splitlines()[:6]:
            if len(kept) >= 3:
                break
            item = raw.strip().lstrip("-•*0123456789.) ").strip()
            if not item or item.upper().startswith("NONE") or len(item) < 12 or len(item) > 240:
                continue
            vec = await asyncio.get_running_loop().run_in_executor(self.bot.mood_executor, self.model.embed, item)
            rows, vecs = self.store.entries(guild_id)
            if len(rows) and float(np.max(vecs @ vec)) >= float(self.cfg.duplicate_similarity):
                log.info("📜 (already remembered: %s)", item)
                continue
            people = self._people(guild_id, item)
            self.store.add(guild_id, item, people, vec)
            kept.append(item)
        log.info("📜 Lore from %d lines (%.1fs): %s", len(lines), time.perf_counter() - t0,
                 " | ".join(kept) if kept else "nothing worth keeping")

    def forget(self, user_id: int) -> int:
        for buf in self._pending.values():
            buf[:] = [x for x in buf if x[1] != user_id]
        return self.store.forget(user_id)
