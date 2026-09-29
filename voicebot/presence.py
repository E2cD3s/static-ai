"""Rotating custom status on the bot's profile: live nerdy numbers (LLM speed, voice latency, GPU, STT/TTS
realtime factors, uptime). Bots can't show full Rich Presence (images, buttons, timers), only a status line,
so this cycles one line every `interval_s`. Same sources as /status, minus anything slow: no LLM calls, no CPU
sampling - just in-memory counters and one nvidia-smi query.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import TYPE_CHECKING

import discord

from .stats import _gpus, _pct, dur, num

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.presence")

_MAX_LEN = 128  # Discord's custom status limit


def _gpu_line(gpus: list[dict]) -> str | None:
    if not gpus:
        return None
    g = gpus[0]
    name = g.get("name", "GPU").replace("NVIDIA ", "").replace("GeForce ", "")
    try:
        used, total = float(g["memory.used"]) / 1024, float(g["memory.total"]) / 1024
        return (f"🖥 {name} · {g['utilization.gpu']}% · {used:.1f}/{total:.0f} GB VRAM · "
                f"{g['temperature.gpu']}°C · {float(g['power.draw']):.0f} W")
    except (KeyError, ValueError):
        return f"🖥 {name}"


def lines(bot: "VoiceBot", gpus: list[dict]) -> list[str]:
    """Every status line there's data for right now, in rotation order."""
    out = []
    u = bot.llm.usage
    tot = u.total()
    model = bot.llm.model_for(bot.llm.voice_endpoint)
    speeds = [s for _, _, s in u.recent]
    if speeds:
        out.append(f"🧠 {model} · {statistics.median(speeds):.0f} tok/s"
                   + (f" · TTFT {tot.ttft_s / tot.timed * 1000:.0f}ms" if tot.timed else ""))
    else:
        out.append(f"🧠 {model} · local on Ollama")

    turns = [t["total_ms"] for t in bot.voice_turns if t.get("total_ms")]
    if turns:
        out.append(f"⏱ voice p50 {_pct(turns, 50):.0f}ms · p90 {_pct(turns, 90):.0f}ms "
                   f"end-of-speech→audio ({len(turns)} turns)")

    if (g := _gpu_line(gpus)):
        out.append(g)

    stt, tts = bot.stt, bot.tts
    parts = []
    if stt is not None and stt.busy_s:
        parts.append(f"👂 whisper {stt.audio_s / stt.busy_s:.0f}× realtime")
    if tts is not None and tts.busy_s:
        parts.append(f"🗣 kokoro {tts.audio_s / tts.busy_s:.0f}× realtime")
    if parts:
        out.append(" · ".join(parts))

    if tot.requests:
        out.append(f"📊 {num(tot.completion_tokens)} tokens out · {num(tot.prompt_tokens)} in · "
                   f"{tot.requests} LLM calls")

    c = bot.counters
    out.append(f"⚡ up {dur(time.time() - bot.started_at)} · {c['utterances']} heard · "
               f"{c['voice replies'] + c['text replies']} replies · ping {bot.latency * 1000:.0f}ms")

    people = sum(len(s.humans_in_channel()) for s in bot.sessions.values() if s.vc.channel is not None)
    if bot.sessions:
        out.append(f"🔊 in voice with {people} {'person' if people == 1 else 'people'} · /status for more")
    return [line[:_MAX_LEN] for line in out]


class Presence:
    def __init__(self, bot: "VoiceBot"):
        self.bot = bot
        self.cfg = bot.cfg.presence
        self._task: asyncio.Task | None = None
        self._i = 0
        self._last = ""

    def start(self) -> None:
        if self.cfg.enabled and self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        interval = max(15.0, float(self.cfg.interval_s))  # Discord rate-limits presence updates
        while True:
            try:
                gpus, _ = await _gpus() if self.cfg.gpu else ([], "")
                options = lines(self.bot, gpus)
                text = options[self._i % len(options)]
                self._i += 1
                if text != self._last:
                    await self.bot.change_presence(status=discord.Status.online,
                                                   activity=discord.CustomActivity(name=text))
                    self._last = text
                    if self._i == 1:
                        log.info("Profile status: rotating %d stat lines every %.0fs (now: %s)", len(options), interval, text)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - never let the status line take anything down
                log.exception("Presence update failed")
            await asyncio.sleep(interval)
