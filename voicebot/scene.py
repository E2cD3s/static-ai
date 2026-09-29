"""Rolling "what's going on in this call" notes, so the bot is caught up when someone calls it in.

The conversation history holds the raw lines, but it's trimmed, and ten minutes of game chatter ("the fans push
them in", "you made a blender") is hard for a 4B model to piece together on the spot. While the call is quiet
for a moment, a small background LLM call folds the new lines into 2-4 lines of notes ("jordan built a mob grinder
called the Mob Masher in Minecraft; it killed Dorothy"). The notes go on the newest turn only - never the
system prompt, which has to stay the same for Ollama's prompt cache.
"""
from __future__ import annotations

import logging
import time
from collections import deque

log = logging.getLogger("voicebot.scene")


class Scene:
    def __init__(self, cfg, llm, bot_name: str):
        self.cfg = cfg.scene
        self.llm = llm
        self.bot_name = bot_name
        self.lines: deque[str] = deque(maxlen=int(self.cfg.max_lines))
        self.summary = ""
        self.fresh = 0          # lines since the notes were last updated
        self.updated_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def add(self, line: str) -> None:
        self.lines.append(" ".join(line.split())[:300])
        self.fresh += 1

    def reset(self) -> None:
        self.lines.clear()
        self.summary, self.fresh = "", 0

    def due(self) -> bool:
        return self.enabled and self.fresh >= int(self.cfg.update_every)

    def note(self) -> str | None:
        if not self.summary:
            return None
        return f"[What's been going on in this call (your own notes - use them to follow along): {self.summary}]"

    async def update(self, context: str = "") -> None:
        """Fold the new lines into the notes. Cancel-safe: nothing changes unless it finishes.
        context: what Discord shows people doing (games, streams)."""
        new = list(self.lines)[-max(self.fresh, 1):]
        count = self.fresh
        prompt = self.cfg.prompt.replace("{name}", self.bot_name)
        t0 = time.perf_counter()
        text = await self.llm.complete([
            {"role": "system", "content": prompt},
            {"role": "user", "content": (f"{context}\n\n" if context else "") +
                                        f"Current notes:\n{self.summary or '(none yet)'}\n\n"
                                        "New lines:\n" + "\n".join(new) + "\n\nUpdated notes:"},
        ], endpoint=self.cfg.endpoint or None, temperature=0.2, max_tokens=int(self.cfg.max_tokens), purpose="scene")
        fields = [ln.strip(" -•*").replace("\\_", "_").replace("**", "") for ln in text.strip().splitlines()]
        notes = " / ".join(f for f in fields if f.lower().startswith(("doing:", "topic:", "notable:"))
                           and not f.lower().rstrip(". ").endswith(("nothing", "none")))
        if notes:
            self.summary = notes[: int(self.cfg.max_chars)]
            self.updated_at = time.monotonic()
            log.info("🧭 Scene notes (%d new lines, %.1fs): %s", count, time.perf_counter() - t0, self.summary)
        self.fresh = max(0, self.fresh - count)
