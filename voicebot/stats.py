"""/status: every nerdy number about the bot - LLM tokens and speed, context fill, Ollama, GPU, CPU/RAM,
the voice pipeline's latency breakdown, STT/TTS realtime factors, search and reminders.

Numbers come from counters kept by the modules themselves (LLMRouter.usage, TimedSTT, TTS, WebSearch.stats,
Planner.stats, bot.counters/voice_turns) plus a live sample of the machine (nvidia-smi, psutil, Ollama's
/api/ps). Collection runs concurrently and takes ~0.3s (the CPU sample). Nothing here touches the LLM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import statistics
import time
from typing import TYPE_CHECKING

import discord
import httpx
import psutil

from .tuning import LEVEL_NAMES

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.stats")

_GPU_FIELDS = ("name", "driver_version", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu",
               "power.draw", "power.limit", "clocks.sm", "clocks.mem", "fan.speed", "pstate",
               "pcie.link.gen.current", "pcie.link.width.current")
_SPARK = "▁▂▃▄▅▆▇█"
_MAX_CHARS = 5900  # Discord: 6000 characters across all embeds of one message


# ------------------------------------------------------------------ formatting


def bar(frac: float, width: int = 12) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = round(frac * width)
    return "█" * full + "░" * (width - full)


def spark(values: list[float], top: float | None = None) -> str:
    if not values:
        return ""
    top = top or max(values) or 1
    return "".join(_SPARK[min(7, int(v / top * 7.999))] for v in values)


def num(n: float) -> str:
    """1234 -> 1.2k, 3456789 -> 3.46M"""
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3g}{unit}"
    return f"{n:.0f}"


def gib(b: float) -> str:
    return f"{b / 2**30:.1f} GiB"


def dur(s: float) -> str:
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = [f"{d}d"] * bool(d) + [f"{h}h"] * bool(h) + [f"{m}m"] * bool(m)
    return " ".join(parts[:2]) or f"{s}s"


def ms(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}ms"


def block(lines: list[str]) -> str:
    text = "\n".join(lines)
    return f"```\n{text[:1010]}\n```"


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


# ------------------------------------------------------------------ collection


async def _run(*cmd: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 4)
        return out.decode(errors="replace")
    except (OSError, asyncio.TimeoutError):
        return ""


async def _gpus() -> tuple[list[dict], str]:
    csv, full = await asyncio.gather(
        _run("nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"),
        _run("nvidia-smi"))
    gpus = [dict(zip(_GPU_FIELDS, (v.strip() for v in line.split(",")))) for line in csv.splitlines() if line.strip()]
    cuda = m.group(1) if (m := re.search(r"CUDA Version:\s*([\d.]+)", full)) else "?"
    return gpus, cuda


async def _ollama(base_url: str) -> dict:
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    try:
        async with httpx.AsyncClient(timeout=1.5) as http:
            ps, ver = await asyncio.gather(http.get(f"{root}/api/ps"), http.get(f"{root}/api/version"))
        return {"models": ps.json().get("models", []), "version": ver.json().get("version", "?")}
    except Exception:  # noqa: BLE001 - not Ollama, or down
        return {}


def _machine() -> dict:
    """Blocking (~0.3s CPU sample) - run in a thread."""
    proc = psutil.Process()
    proc.cpu_percent(None)
    per_core = psutil.cpu_percent(0.3, percpu=True)
    temps = {}
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        pass
    cpu_temp = next((t.current for key in ("coretemp", "k10temp", "cpu_thermal") for t in temps.get(key, [])), None)
    model = "?"
    try:
        with open("/proc/cpuinfo") as f:
            model = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "?")
    except OSError:
        pass
    freq = psutil.cpu_freq()
    with proc.oneshot():
        p = {"rss": proc.memory_info().rss, "cpu": proc.cpu_percent(None), "threads": proc.num_threads(),
             "fds": proc.num_fds() if hasattr(proc, "num_fds") else 0}
    return {
        "cpu_model": re.sub(r"\s+", " ", model.replace("(R)", "").replace("(TM)", "").replace(" CPU", "")),
        "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
        "per_core": per_core, "cpu": sum(per_core) / len(per_core), "freq": freq.current if freq else None,
        "load": os.getloadavg(), "cpu_temp": cpu_temp,
        "mem": psutil.virtual_memory(), "swap": psutil.swap_memory(), "disk": psutil.disk_usage("/"),
        "boot": psutil.boot_time(), "proc": p,
    }


def _versions() -> dict[str, str]:
    out = {"Python": platform.python_version(), "discord.py": discord.__version__}
    for mod, label in (("faster_whisper", "faster-whisper"), ("onnxruntime", "onnxruntime"), ("openai", "openai")):
        try:
            out[label] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


# ------------------------------------------------------------------ rendering


async def build(bot: "VoiceBot", guild: discord.Guild | None) -> list[discord.Embed]:
    t0 = time.perf_counter()
    ep = bot.llm.endpoints[bot.llm.voice_endpoint]
    (gpus, cuda), ollama, machine = await asyncio.gather(_gpus(), _ollama(ep.base_url), asyncio.to_thread(_machine))
    embeds = [
        _overview(bot, guild, machine),
        _llm(bot, ollama),
        _voice(bot, guild),
        _hardware(gpus, cuda, machine),
        _features(bot, guild),
    ]
    embeds[-1].set_footer(text=f"collected in {(time.perf_counter() - t0) * 1000:.0f}ms · "
                               + " · ".join(f"{k} {v}" for k, v in _versions().items()))
    # Stay under Discord's 6000-char message limit: trim the widest fields if ever needed.
    while sum(len(e) for e in embeds) > _MAX_CHARS:
        e, i = max(((e, i) for e in embeds for i in range(len(e.fields))), key=lambda x: len(x[0].fields[x[1]].value))
        f = e.fields[i]
        e.set_field_at(i, name=f.name, value=f.value[: len(f.value) // 2].rstrip("`\n") + "\n…```", inline=f.inline)
    return embeds


def _overview(bot: "VoiceBot", guild, m: dict) -> discord.Embed:
    e = discord.Embed(title=f"📡 {bot.cfg.bot.name} · system status", color=0x5865F2,
                      description=f"up **{dur(time.time() - bot.started_at)}** · gateway **{bot.latency * 1000:.0f}ms** · "
                                  f"{len(bot.guilds)} server(s) · as of <t:{int(time.time())}:T>")
    p = m["proc"]
    e.add_field(name="🐍 Process", inline=True, value=block([
        f"RSS     {gib(p['rss'])}",
        f"CPU     {p['cpu']:.0f}%",
        f"threads {p['threads']}  fds {p['fds']}",
        f"tasks   {len(asyncio.all_tasks())}",
    ]))
    c = bot.counters
    e.add_field(name="💬 Activity", inline=True, value=block([
        f"heard       {c['utterances']}",
        f"voice reply {c['voice replies']}",
        f"text reply  {c['text replies']}",
        f"convos      {len(bot.text_history)} text",
    ]))
    lines = []
    for s in bot.sessions.values():
        ch = s.vc.channel
        if ch is None:
            continue
        people = s.humans_in_channel()
        lines.append(f"🔊 {ch.mention} · {len(people)} people · {len(s.history)} msgs in memory · "
                     f"ws {s.vc.latency * 1000:.0f}ms")
    e.add_field(name="🎧 Voice", inline=False, value="\n".join(lines) or "not in a voice channel")
    return e


def _llm(bot: "VoiceBot", ollama: dict) -> discord.Embed:
    u = bot.llm.usage
    tot = u.total()
    e = discord.Embed(title="🧠 LLM", color=0x57F287)
    text_ep, voice_ep = bot.llm.text_endpoint, bot.llm.voice_endpoint
    models = f"voice+text **{voice_ep}** `{bot.llm.model_for(voice_ep)}`" if text_ep == voice_ep else (
        f"voice **{voice_ep}** `{bot.llm.model_for(voice_ep)}`\ntext **{text_ep}** `{bot.llm.model_for(text_ep)}`")
    e.description = models

    ctx = 0
    if ollama:
        lines = []
        for mdl in ollama.get("models", []):
            d = mdl.get("details", {})
            size, vram = mdl.get("size", 0), mdl.get("size_vram", 0)
            ctx = max(ctx, int(mdl.get("context_length") or 0))
            on_gpu = vram / size * 100 if size else 0
            exp = mdl.get("expires_at", "")
            keep = "forever" if exp[:4].isdigit() and int(exp[:4]) > 2100 else exp[11:16]  # keep_alive -1
            lines += [mdl.get("name", "?"),
                      f"  {d.get('parameter_size', '?')} {d.get('quantization_level', '')} · {d.get('family', '')}",
                      f"  {gib(size)} · {on_gpu:.0f}% GPU · ctx {ctx} · keep {keep}"]
        e.add_field(name=f"🦙 Ollama {ollama.get('version', '')}", inline=False,
                    value=block(lines or ["no model loaded"]))

    speeds = [s for _, _, s in u.recent]
    e.add_field(name="Σ Totals", inline=True, value=block([
        f"requests {tot.requests}",
        f"prompt   {num(tot.prompt_tokens)} tok",
        f"output   {num(tot.completion_tokens)} tok",
        f"speed    {tot.gen_tokens / tot.gen_s:.1f} tok/s" if tot.gen_s else "speed    -",
        f"TTFT     {ms(tot.ttft_s / tot.timed * 1000 if tot.timed else None)}",
        f"cut/err  {tot.cancelled}/{tot.errors}",
    ]))
    fill = []
    for purpose in ("voice", "text"):
        n = u.last_prompt.get(purpose)
        if n:
            fill.append(f"{purpose:5} {bar(n / ctx, 10) if ctx else ''} {num(n)}" + (f"/{num(ctx)}" if ctx else ""))
    fill.append(f"tok/s {spark(speeds[-24:])}" if speeds else "tok/s (no data yet)")
    if speeds:
        fill.append(f"      min {min(speeds):.0f} · med {statistics.median(speeds):.0f} · max {max(speeds):.0f}")
    e.add_field(name="📏 Context & speed", inline=True, value=block(fill))

    rows = [f"{'purpose':13}{'reqs':>5}{'in':>7}{'out':>7}{'tok/s':>6}{'ttft':>7}"]
    for purpose, t in sorted(u.by_purpose.items(), key=lambda kv: -kv[1].requests):
        rate = f"{t.gen_tokens / t.gen_s:.0f}" if t.gen_s else "-"
        ttft = f"{t.ttft_s / t.timed * 1000:.0f}ms" if t.timed else "-"
        rows.append(f"{purpose[:13]:13}{t.requests:>5}{num(t.prompt_tokens):>7}{num(t.completion_tokens):>7}"
                    f"{rate:>6}{ttft:>7}")
    e.add_field(name="🧾 By purpose", inline=False, value=block(rows if len(rows) > 1 else ["no LLM calls yet"]))
    return e


def _voice(bot: "VoiceBot", guild) -> discord.Embed:
    e = discord.Embed(title="🎙 Voice pipeline", color=0xFEE75C)
    stt, tts = bot.stt, bot.tts
    if stt is not None:
        e.add_field(name="👂 STT", inline=True, value=f"`{stt.desc}`\n" + block([
            f"calls  {stt.calls}",
            f"audio  {dur(stt.audio_s) if stt.audio_s >= 60 else f'{stt.audio_s:.0f}s'}",
            f"avg    {ms(stt.busy_s / stt.calls * 1000 if stt.calls else None)}",
            f"speed  {stt.audio_s / stt.busy_s:.0f}× realtime" if stt.busy_s else "speed  -",
        ]))
    if tts is not None:
        e.add_field(name="🗣 TTS", inline=True, value=f"`{tts.desc}`\n" + block([
            f"chunks {tts.calls} ({num(tts.chars)} chars)",
            f"audio  {dur(tts.audio_s) if tts.audio_s >= 60 else f'{tts.audio_s:.0f}s'}",
            f"avg    {ms(tts.busy_s / tts.calls * 1000 if tts.calls else None)}",
            f"speed  {tts.audio_s / tts.busy_s:.0f}× realtime" if tts.busy_s else "speed  -",
        ]))

    turns = [t for t in bot.voice_turns if t.get("total_ms")]
    if turns:
        totals = [t["total_ms"] for t in turns]
        spec = [t for t in turns if t.get("speculative")]

        def avg(key: str) -> str:
            return ms(_avg([t[key] for t in turns if key in t]))
        e.add_field(name=f"⏱ Latency · last {len(turns)} replies (end of speech → audio)", inline=False, value=block([
            f"p50 {ms(_pct(totals, 50))} · p90 {ms(_pct(totals, 90))} · best {ms(min(totals))} · worst {ms(max(totals))}",
            f"trend {spark(totals[-30:])}",
            f"avg: turn end {avg('turn_end_ms')} · stt {avg('stt_ms')} · search {avg('search_ms')}",
            f"     mood wait {avg('mood_wait_ms')} · llm 1st token {avg('llm_ttft_ms')}",
            f"     1st chunk {avg('chunk1_ms')} · tts {avg('tts_first_ms')}",
            f"speculative {len(spec)}/{len(turns)}" + (f" · p50 {ms(_pct([t['total_ms'] for t in spec], 50))}"
                                                        if spec else ""),
        ]))
    c = bot.counters
    started, used = c["speculative started"], c["speculative used"]
    e.add_field(name="🤝 Turn-taking", inline=True, value=block([
        f"spec used   {used}/{started}" + (f" ({used / started:.0%})" if started else ""),
        f"barge-ins   {c['barge-ins']}",
        f"backchannel {c['backchannels ignored']} ignored",
        f"waited+redo {c['replies redone after waiting']}",
        f"clips       {c['clips']}",
    ]))
    if bot.tuning.enabled and guild is not None:
        t = bot.tuning.for_guild(guild.id)
        last = f"\nlast: {t.log[-1][1][:60]}" if t.log else ""
        e.add_field(name="🎛 Self-tuning", inline=False, value=block([
            f"length {LEVEL_NAMES[t.level]} ({t.level}/3) · follow-up {t.followup_s:.0f}s "
            f"(base {bot.tuning.base_followup:.0f}s) · streak {t.streak}" + last]))
    v = bot.cfg.voice
    e.add_field(name="🎚 VAD / turn config", inline=True, value=block([
        f"pause {v.silence_short_ms}/{v.silence_ms}/{v.silence_long_ms}ms",
        f"spec stt {v.speculative_stt_ms}ms · reply {'on' if v.speculative_reply else 'off'}",
        f"barge-in {v.barge_in_ms}ms · duck {v.duck_volume}",
        f"mode {v.response_mode}",
    ]))
    return e


def _hardware(gpus: list[dict], cuda: str, m: dict) -> discord.Embed:
    e = discord.Embed(title="🖥 Hardware", color=0xEB459E)
    for g in gpus:
        def f(key, default=0.0):
            try:
                return float(g.get(key, default))
            except ValueError:
                return default
        used, total = f("memory.used"), f("memory.total", 1)
        e.add_field(name=f"🎮 {g.get('name', 'GPU')}", inline=False, value=block([
            f"util  {bar(f('utilization.gpu') / 100)} {f('utilization.gpu'):.0f}%",
            f"VRAM  {bar(used / total)} {used / 1024:.1f}/{total / 1024:.1f} GiB",
            f"temp  {f('temperature.gpu'):.0f}°C · fan {g.get('fan.speed', '?')}% · {g.get('pstate', '')}",
            f"power {f('power.draw'):.0f}/{f('power.limit'):.0f} W · core {g.get('clocks.sm')} / mem {g.get('clocks.mem')} MHz",
            f"PCIe gen{g.get('pcie.link.gen.current')} x{g.get('pcie.link.width.current')} · "
            f"driver {g.get('driver_version')} · CUDA {cuda}",
        ]))
    if not gpus:
        e.add_field(name="🎮 GPU", inline=False, value="nvidia-smi not available")
    load = " ".join(f"{x:.2f}" for x in m["load"])
    e.add_field(name=f"🧮 {m['cpu_model']}", inline=False, value=block([
        f"util  {bar(m['cpu'] / 100)} {m['cpu']:.0f}%   cores {spark(m['per_core'], 100)}",
        f"{m['cores']}C/{m['threads']}T" + (f" · {m['freq']:.0f} MHz" if m["freq"] else "")
        + (f" · {m['cpu_temp']:.0f}°C" if m["cpu_temp"] else "") + f" · load {load}",
    ]))
    mem, swap, disk = m["mem"], m["swap"], m["disk"]
    e.add_field(name="💾 Memory & disk", inline=False, value=block([
        f"RAM   {bar(mem.percent / 100)} {gib(mem.used)}/{gib(mem.total)}",
        f"swap  {bar(swap.percent / 100)} {gib(swap.used)}/{gib(swap.total)}",
        f"disk  {bar(disk.percent / 100)} {gib(disk.used)}/{gib(disk.total)}",
        f"host up {dur(time.time() - m['boot'])} · {platform.system()} {platform.release()}",
    ]))
    return e


def _features(bot: "VoiceBot", guild) -> discord.Embed:
    e = discord.Embed(title="🧩 Features", color=0xED4245)
    s = bot.search.stats
    lines = [f"checks {s['checks']} · searched {s['searches']} · skipped {s['skipped (chatter)']}"]
    for b in bot.search.backends:
        n = s[f"{b} ok"] + s[f"{b} empty"]
        lines.append(f"{b:10} ok {s[f'{b} ok']} · empty {s[f'{b} empty']} · fail {s[f'{b} fail']}"
                     + (f" · {s[f'{b} ms'] / n:.0f}ms" if n else ""))
    e.add_field(name="🔎 Web search" + ("" if bot.search.enabled else " (off)"), inline=False, value=block(lines))

    p = bot.planner
    pending = p.store.pending() if p.enabled else {}
    e.add_field(name="⏰ Reminders & polls" + ("" if p.enabled else " (off)"), inline=True, value=block([
        f"pending  {pending.get('reminder', 0)} · polls {pending.get('poll', 0)}",
        f"set      {p.stats['reminders set']} · sent {p.stats['reminders delivered']}",
        f"polls    {p.stats['polls started']} · closed {p.stats['polls closed']}",
    ]))
    m = bot.mood
    if m.enabled and m.model is not None:
        dist = sorted(m.counts.items(), key=lambda kv: -kv[1])
        lines = [f"reads {m.reads} · avg {ms(m.total_ms / m.reads if m.reads else None)} · "
                 f"calibrated people {m.calibrated()}",
                 " ".join(f"{g} {n}" for g, n in dist[:6]) or "no reads yet"]
        lines += m.summary(guild.id if guild else 0)[:4]
        e.add_field(name=f"🎭 Mood · {m.model.desc}", inline=False, value=block(lines))
    try:
        db = bot.profiles.store.db
        known, written, queued = db.execute(
            "SELECT COUNT(*), SUM(profile != ''), (SELECT COUNT(*) FROM pending) FROM users").fetchone()
        sizes = sum(os.path.getsize(f) for f in (bot.cfg.profiles.db_path, bot.cfg.reminders.db_path)
                    if os.path.exists(f))
        e.add_field(name="🗂 Profiles", inline=True, value=block([
            f"people   {known} · profiles {written or 0}",
            f"queued   {queued} lines",
            f"db size  {sizes / 1024:.0f} KiB",
        ]))
    except Exception:  # noqa: BLE001
        pass
    return e


class StatsView(discord.ui.View):
    """🔄 Refresh button under the stats message."""

    def __init__(self, bot: "VoiceBot"):
        super().__init__(timeout=900)
        self.bot = bot
        self.message: discord.Message | None = None

    @discord.ui.button(label="Refresh", emoji="🔄", style=discord.ButtonStyle.secondary)
    async def refresh(self, interaction: discord.Interaction, _button: discord.ui.Button):
        await interaction.response.defer()
        await interaction.edit_original_response(embeds=await build(self.bot, interaction.guild), view=self)

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                await self.message.edit(view=None)
            except discord.HTTPException:
                pass
