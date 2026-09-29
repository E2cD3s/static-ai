"""Structured version of the /status numbers for the public web status page (the Discord /status command
renders the same sources as embeds in stats.py). Nothing here names people: only counts, and the server and
channel names the bot is in voice with."""
from __future__ import annotations

import asyncio
import platform
import statistics
import time
from typing import TYPE_CHECKING

from ..sentiment import STRATEGIES
from ..stats import _avg, _gpus, _machine, _ollama, _pct, _versions
from ..tuning import LEVEL_NAMES

if TYPE_CHECKING:
    from ..bot import VoiceBot


def _f(d: dict, key: str) -> float | None:
    try:
        return float(d.get(key, ""))
    except ValueError:
        return None


def _rtf(t) -> float | None:
    return t.audio_s / t.busy_s if t is not None and t.busy_s else None


def _reply(bot: "VoiceBot") -> dict:
    turns = [t for t in bot.voice_turns if t.get("total_ms")]
    totals = [t["total_ms"] for t in turns]

    def avg(*keys: str) -> float:
        vals = [sum(t.get(k, 0) for k in keys) for t in turns if any(k in t for k in keys)]
        return _avg(vals) or 0.0

    # Sequential parts of one reply, end of speech -> first audio. chunk1 is measured from the LLM request, so it
    # already includes time to first token.
    stages = [
        {"key": "hear", "label": "Hearing you finish", "ms": avg("turn_end_ms")},
        {"key": "look", "label": "Looking things up", "ms": avg("search_ms", "mood_wait_ms")},
        {"key": "think", "label": "Writing the first sentence", "ms": avg("chunk1_ms")},
        {"key": "speak", "label": "Turning it into speech", "ms": avg("tts_first_ms")},
    ]
    spec = [t for t in turns if t.get("speculative")]
    return {
        "count": len(turns), "p50": _pct(totals, 50), "p90": _pct(totals, 90),
        "best": min(totals) if totals else None, "worst": max(totals) if totals else None,
        "trend": [round(x) for x in totals[-40:]], "stages": stages,
        "speculative": len(spec), "stt_ms": avg("stt_ms") if turns else None,
        "llm_ttft_ms": avg("llm_ttft_ms") if turns else None,
    }


def _think(bot: "VoiceBot", ollama: dict) -> dict:
    u = bot.llm.usage
    tot = u.total()
    speeds = [s for _, _, s in u.recent]
    ctx = 0
    models = []
    for mdl in ollama.get("models", []):
        d = mdl.get("details", {})
        size, vram = mdl.get("size", 0), mdl.get("size_vram", 0)
        ctx = max(ctx, int(mdl.get("context_length") or 0))
        models.append({"name": mdl.get("name", "?"), "params": d.get("parameter_size", ""),
                       "quant": d.get("quantization_level", ""), "size_gb": size / 2**30,
                       "gpu_pct": vram / size * 100 if size else 0, "ctx": int(mdl.get("context_length") or 0)})
    return {
        "model": bot.llm.model_for(bot.llm.voice_endpoint),
        "tok_s": statistics.median(speeds) if speeds else None,
        "tok_s_trend": [round(s, 1) for s in speeds[-40:]],
        "ttft_ms": tot.ttft_s / tot.timed * 1000 if tot.timed else None,
        "requests": tot.requests, "prompt_tokens": tot.prompt_tokens, "output_tokens": tot.completion_tokens,
        "cancelled": tot.cancelled, "errors": tot.errors,
        "context": [{"purpose": p, "used": u.last_prompt[p], "max": ctx}
                    for p in ("voice", "text") if u.last_prompt.get(p)],
        "by_purpose": [{"purpose": p, "requests": t.requests,
                        "tok_s": t.gen_tokens / t.gen_s if t.gen_s else None,
                        "ttft_ms": t.ttft_s / t.timed * 1000 if t.timed else None}
                       for p, t in sorted(u.by_purpose.items(), key=lambda kv: -kv[1].requests)],
        "ollama": {"version": ollama.get("version"), "models": models} if ollama else None,
    }


def _machine_view(gpus: list[dict], cuda: str, m: dict) -> dict:
    gpu = None
    if gpus:
        g = gpus[0]
        gpu = {"name": g.get("name", "GPU").replace("NVIDIA ", "").replace("GeForce ", ""),
               "util": _f(g, "utilization.gpu"), "vram_used": (_f(g, "memory.used") or 0) / 1024,
               "vram_total": (_f(g, "memory.total") or 0) / 1024, "temp": _f(g, "temperature.gpu"),
               "power": _f(g, "power.draw"), "power_limit": _f(g, "power.limit"), "fan": _f(g, "fan.speed"),
               "clock_core": _f(g, "clocks.sm"), "clock_mem": _f(g, "clocks.mem"),
               "driver": g.get("driver_version"), "cuda": cuda}
    mem, swap, disk, p = m["mem"], m["swap"], m["disk"], m["proc"]
    return {
        "gpu": gpu,
        "cpu": {"model": m["cpu_model"], "cores": m["cores"], "threads": m["threads"], "util": m["cpu"],
                "per_core": [round(x) for x in m["per_core"]], "freq": m["freq"], "temp": m["cpu_temp"],
                "load": [round(x, 2) for x in m["load"]]},
        "ram": {"used": mem.used / 2**30, "total": mem.total / 2**30},
        "swap": {"used": swap.used / 2**30, "total": swap.total / 2**30},
        "disk": {"used": disk.used / 2**30, "total": disk.total / 2**30},
        "process": {"rss": p["rss"] / 2**30, "cpu": p["cpu"], "threads": p["threads"]},
        "host_up": time.time() - m["boot"], "os": f"{platform.system()} {platform.release()}",
    }


def _features(bot: "VoiceBot") -> dict:
    s = bot.search.stats
    backends = []
    for b in bot.search.backends:
        n = s[f"{b} ok"] + s[f"{b} empty"]
        backends.append({"name": b, "ok": s[f"{b} ok"], "empty": s[f"{b} empty"], "fail": s[f"{b} fail"],
                         "ms": s[f"{b} ms"] / n if n else None})
    p = bot.planner
    pending = p.store.pending() if p.enabled else {}
    m = bot.mood
    mood = None
    if m.enabled and m.model is not None:
        mood = {"reads": m.reads, "avg_ms": m.total_ms / m.reads if m.reads else None,
                "top": sorted(m.counts.items(), key=lambda kv: -kv[1])[:6]}
    memory = None
    try:
        known, written = bot.profiles.store.db.execute(
            "SELECT COUNT(*), SUM(profile != '') FROM users").fetchone()
        memory = {"people": known, "profiles": written or 0}
    except Exception:  # noqa: BLE001
        pass
    return {
        "search": {"enabled": bot.search.enabled, "checks": s["checks"], "searches": s["searches"],
                   "backends": backends},
        "reminders": {"enabled": p.enabled, "pending": pending.get("reminder", 0), "polls": pending.get("poll", 0),
                      "set": p.stats["reminders set"], "sent": p.stats["reminders delivered"]},
        "mood": mood, "memory": memory,
    }


async def build(bot: "VoiceBot", card: dict) -> dict:
    t0 = time.perf_counter()
    ep = bot.llm.endpoints[bot.llm.voice_endpoint]
    (gpus, cuda), ollama, machine = await asyncio.gather(_gpus(), _ollama(ep.base_url), asyncio.to_thread(_machine))
    c = bot.counters
    voice_now = []
    for s in bot.sessions.values():
        ch = s.vc.channel
        if ch is not None:
            voice_now.append({"server": ch.guild.name, "channel": ch.name, "people": len(s.humans_in_channel()),
                              "ws_ms": s.vc.latency * 1000, "platform": "discord"})
    fx = getattr(bot, "fluxer", None)
    for s in (fx.sessions.values() if fx is not None else ()):
        ch = s.vc.channel
        if ch is not None and s.vc.is_connected():
            voice_now.append({"server": ch.guild.name, "channel": ch.name, "people": len(s.humans_in_channel()),
                              "ws_ms": None, "platform": "fluxer"})
    stt, tts = bot.stt, bot.tts
    return {
        "bot": {**card, "ping_ms": bot.latency * 1000 if bot.latency == bot.latency else None},
        "reply": _reply(bot),
        "hear": {"desc": getattr(stt, "desc", None), "rtf": _rtf(stt), "calls": getattr(stt, "calls", 0),
                 "audio_s": getattr(stt, "audio_s", 0), "avg_ms": stt.busy_s / stt.calls * 1000
                 if stt is not None and stt.calls else None},
        "think": _think(bot, ollama),
        "speak": {"desc": getattr(tts, "desc", None), "rtf": _rtf(tts), "calls": getattr(tts, "calls", 0),
                  "chars": getattr(tts, "chars", 0), "audio_s": getattr(tts, "audio_s", 0),
                  "avg_ms": tts.busy_s / tts.calls * 1000 if tts is not None and tts.calls else None},
        "machine": _machine_view(gpus, cuda, machine),
        "activity": {"heard": c["utterances"], "voice_replies": c["voice replies"], "text_replies": c["text replies"],
                     "barge_ins": c["barge-ins"], "backchannels": c["backchannels ignored"], "clips": c["clips"],
                     "spec_used": c["speculative used"], "spec_started": c["speculative started"],
                     "conversations": len(bot.text_history)},
        "voice_now": voice_now,
        "features": _features(bot),
        "versions": _versions(),
        "collected_ms": (time.perf_counter() - t0) * 1000,
        "ts": time.time(),
    }


def learned(bot: "VoiceBot", platform: str = "discord") -> list[dict]:
    """What the bot has adapted per server (admin only: it's about how people there talk to it).
    `bot` is the Discord client or the Fluxer frontend (platform="fluxer") - each has its own tuning/mood."""
    out = []
    tuning, mood = bot.tuning, bot.mood
    guilds = bot.guilds if platform == "discord" else list(bot.state.guilds.values())
    for g in sorted(guilds, key=lambda g: g.name.lower()):
        t = tuning._guilds.get(g.id) if tuning.enabled else None  # never creates state for a server
        tune = None
        if t is not None:
            tune = {"level": t.level, "level_name": LEVEL_NAMES[t.level], "followup_s": t.followup_s,
                    "base_followup_s": tuning.base_followup, "streak": t.streak,
                    "log": [{"at": at, "why": why} for at, why in t.log[-3:]][::-1]}
        strategies = []
        if mood.enabled:
            with mood._lock:  # the mood thread updates it
                arms = {k: [list(a) for a in v] for k, v in mood.state.get("arms", {}).get(str(g.id), {}).items()}
            for group, pair in sorted(arms.items()):
                tries = [int(a + b - 2) for a, b in pair]
                if not sum(tries):
                    continue
                strategies.append({"mood": group, "options": [
                    {"text": STRATEGIES.get(group, ("?", "?"))[i], "wins": int(a - 1), "tries": tries[i],
                     "rate": (a / (a + b))} for i, (a, b) in enumerate(pair)]})
        icon = getattr(g, "icon", None)
        voice = bool(g.voice_client) if platform == "discord" else g.id in bot.sessions
        out.append({"id": str(g.id), "name": g.name, "icon": icon.url if icon else None, "platform": platform,
                    "voice": voice, "tuning": tune, "strategies": strategies})
    return out
