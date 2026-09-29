"""Static on Fluxer: the same persona and pipeline as the Discord bot, with its own data.

Runs in the same process as the Discord client (bot.py's VoiceBot), which hosts the shared pieces: the LLM
router, Whisper, Kokoro, the mood/memory ONNX models, web search, the addressee check, and the thread pools.
Everything that is *about people* is Fluxer's own: profiles, memories (lore), mood learning, tuning,
reminders and the muted list live under fluxer.data_dir (see _fluxer_cfg).

Never touches the Discord connection. run() is supervised: gateway trouble is logged and retried, and nothing
here can end the process (main.run_multi only watches Discord and close())."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import calc, fun, members, quotes
from ..quotes import QuoteBook
from ..clips import encode_mp3
from ..config import Cfg
from ..lore import Lore
from ..profiles import ProfileManager
from ..reminders import Planner, parse_duration, parse_when, spoken_time
from ..sentiment import MoodReader
from ..text_utils import (EchoGuard, RepeatGuard, SpeakerGuard, now_note, prompt_examples, recent_replies,
                          split_message, trim_history)
from ..tuning import LEVEL_NAMES, TuningStore
from ..vision import PLACEHOLDER, Vision, image_parts
from ..voice_session import VoiceSession
from . import help as fx_help
from .objects import POLL_EMOJI, FxChannel, FxFile, FxGuild, FxHTTPError, FxMessage, FxPoll, FxState, FxUser
from .voice import FxVoiceClient

if TYPE_CHECKING:
    from ..bot import VoiceBot

log = logging.getLogger("voicebot.fluxer")

# "what do you think of my pfp" -> look at the actual avatar (same as bot.py)
_AVATAR_RE = re.compile(r"\b(?:pfps?|avatars?|profile ?(?:pics?|pictures?|photos?))\b", re.I)
_YOUR_AVATAR_RE = re.compile(r"\byour (?:pfp|avatar|profile ?(?:pic|picture|photo))\b", re.I)

GATEWAY_INTENTS = (1 << 16) - 1  # everything the Python wrapper knows about; Fluxer sends what applies


class _Overlay(Cfg):
    """A config section that reads `over` first, then the live `base` section - so dashboard edits to shared
    settings (voice timing, prompts, LLM) reach Fluxer too, while its paths/ids stay its own."""

    def __init__(self, base: dict, over: dict):
        dict.__init__(self)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_over", over)

    def __getitem__(self, key):
        over = object.__getattribute__(self, "_over")
        return over[key] if key in over else object.__getattribute__(self, "_base")[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return key in object.__getattribute__(self, "_over") or key in object.__getattribute__(self, "_base")

    def keys(self):
        return {**object.__getattribute__(self, "_base"), **object.__getattribute__(self, "_over")}.keys()

    def items(self):
        return {k: self[k] for k in self.keys()}.items()

    def __iter__(self):
        return iter(self.keys())


def _fluxer_cfg(cfg: Cfg) -> Cfg:
    fx = cfg.fluxer
    d = Path(fx.data_dir)
    sections = {
        "bot": {"creator_ids": fx.get("creator_ids") or []},
        "discord": {"allowed_user_ids": [], "text_channel_ids": fx.text_channel_ids,
                    "context_messages": fx.context_messages, "respond_in_dms": fx.respond_in_dms},
        "profiles": {"db_path": str(d / "profiles.db")},
        "reminders": {"db_path": str(d / "reminders.db")},
        "sentiment": {"path": str(d / "mood.json")},
        "lore": {"db_path": str(d / "lore.db")},
        "tuning": {"path": str(d / "tuning.json")},
        "quotes": {"db_path": str(d / "quotes.db")},
    }
    root = _Overlay(cfg, {k: _Overlay(cfg[k], v) for k, v in sections.items()})
    return root


class _Profiles(ProfileManager):
    """Fluxer's profiles. Live conversation on either platform pauses background LLM jobs on both
    (one Ollama slot): our activity pokes Discord's, and Discord's counts as ours."""

    def __init__(self, bot: "FluxerBot"):
        self._own_activity = 0.0
        super().__init__(bot)

    @property
    def _last_activity(self) -> float:  # type: ignore[override]
        return max(self._own_activity, self.bot.host.profiles.last_activity)

    @_last_activity.setter
    def _last_activity(self, value: float) -> None:
        self._own_activity = value

    def activity(self, cancel: bool = True) -> None:
        super().activity(cancel)
        self.bot.host.profiles.activity(cancel)


class _Vision(Vision):
    """Fluxer attachments/avatars download themselves (FxAttachment/FxAsset.read, Fluxer media URLs only)."""

    async def _download(self, source) -> bytes:
        return await source.read()


class _Planner(Planner):
    """Reminders work as-is (FxUser.send DMs, FxChannel.send pings). Polls are reaction polls (FxPoll): counted
    from the reactions when they close, the result posted in the chat and announced in voice like on Discord."""

    def build_poll(self, question: str, options: list[str], seconds: float, multiple: bool = False):
        seconds = min(max(float(seconds), 60.0), 32 * 86400.0)
        ends = time.time() + seconds
        return FxPoll(question[:300], [o[:80] for o in options[:10]], multiple, ends), ends

    async def _fire_poll(self, row) -> None:
        channel = await self._channel(row["channel_id"])
        if channel is None:
            return
        try:
            msg = await channel.fetch_message(row["message_id"])
        except FxHTTPError:
            return  # deleted
        options = [ln.split(" ", 1)[1] for ln in msg.content.splitlines()
                   if any(ln.startswith(e + " ") for e in POLL_EMOJI)]
        counts = [(opt, max(0, msg.reactions.get(POLL_EMOJI[i], 1) - 1)) for i, opt in enumerate(options)]
        ranked = sorted(counts, key=lambda c: -c[1])
        total = sum(n for _, n in counts)
        summary = ", ".join(f"{opt}: {n}" for opt, n in ranked)
        self.stats["polls closed"] += 1
        log.info("📊 Fluxer poll #%d closed: %s -> %s", row["id"], row["text"], summary)
        top = ranked[0][1] if ranked else 0
        winners = [opt for opt, n in ranked if n == top and n > 0]
        head = f"🏆 **{' / '.join(winners)}**" if winners else "Nobody voted."
        try:
            await channel.send(f"📊 Poll closed: **{row['text']}**\n{head}\n-# "
                               + " · ".join(f"{opt} {n}" for opt, n in ranked), reference=msg)
        except FxHTTPError as e:
            log.warning("Posting poll results failed: %s", e)
        session, here = self._voice_listeners(row["guild_id"])
        if session and here:
            result = f"Results: {summary} ({total} votes)." if total else "Nobody voted."
            session.post_event(f"The poll \"{row['text']}\" just closed. {result} Announce the result in a few words",
                               respond=True)


class FluxerBot:
    def __init__(self, host: "VoiceBot"):
        self.host = host
        self.cfg = _fluxer_cfg(host.cfg)
        self.fx = host.cfg.fluxer
        self.data_dir = Path(self.fx.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state = FxState(self.fx.api_url, self.fx.token)
        self.state.text_for_voice = self._text_channel  # voice channels have no chat: sends go here instead
        self.gateway = None
        self.user: FxUser | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self._voice_waiters: dict[int, asyncio.Future] = {}
        self.sessions: dict[int, VoiceSession] = {}
        self._session_text: dict[int, FxChannel] = {}   # guild -> text channel the voice session reports to
        self.text_history: dict[int, list[dict]] = {}
        self._last_seen: dict[int, int] = {}
        self._last_active: dict[int, float] = {}
        self.started_at = time.time()
        self.muted_path = self.data_dir / "muted.json"
        self.muted: set[int] = self._load_muted()
        prefix = re.escape(str(self.fx.prefix or "!"))
        self._cmd_re = re.compile(rf"^\s*{prefix}([a-z]+)\b\s*(.*)$", re.I | re.S)
        names = [host.cfg.bot.name, *host.cfg.voice.wake_words] if self.fx.respond_to_name else []
        self._name_re = (re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in names if n), re.I)
                         if any(names) else None)
        # Fluxer's own data (see module docstring)
        self.vision = _Vision(self)
        self.profiles = _Profiles(self)
        self.planner = _Planner(self)
        self.tuning = TuningStore(self.cfg)
        self.mood = MoodReader(self)
        self.lore = Lore(self)
        self.quotes = QuoteBook(self.cfg.quotes.db_path)  # Fluxer's own quote book
        self.links.profile_stores["fluxer"] = self.profiles.store
        self.profiles.linked = lambda uid: self.links.note("fluxer", uid)
        self._started_jobs = False

    # ------------------------------------------------------------------ shared with the Discord host
    llm = property(lambda self: self.host.llm)
    stt = property(lambda self: self.host.stt)
    tts = property(lambda self: self.host.tts)
    tts_cache = property(lambda self: self.host.tts_cache)
    stt_executor = property(lambda self: self.host.stt_executor)
    tts_executor = property(lambda self: self.host.tts_executor)
    mood_executor = property(lambda self: self.host.mood_executor)
    search = property(lambda self: self.host.search)
    weather = property(lambda self: self.host.weather)
    links = property(lambda self: self.host.links)
    addressee = property(lambda self: self.host.addressee)
    counters = property(lambda self: self.host.counters)
    voice_turns = property(lambda self: self.host.voice_turns)

    def persona(self) -> str:
        return self.host.persona()

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Connect and stay connected. Never raises (except cancellation)."""
        from fluxer.gateway import Gateway
        from fluxer.http import HTTPClient

        await self.host.models_ready.wait()
        # Share the loaded mood/memory models instead of loading a second copy (RAM is tight).
        self.mood.model = self.host.mood.model
        self.mood.enabled = self.mood.enabled and self.mood.model is not None
        self.lore.model = self.host.lore.model
        delay = 5
        while not self._closed:
            http = HTTPClient(self.fx.token, api_url=self.fx.api_url)
            self.gateway = Gateway(http_client=http, token=self.fx.token, intents=GATEWAY_INTENTS,
                                   dispatch=self._dispatch)
            try:
                log.info("Connecting to Fluxer (%s)", self.fx.api_url)
                await self.gateway.connect()  # reconnects/resumes by itself; returns when closed
                delay = 5
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Fluxer connection failed - retrying in %ds", delay)
            finally:
                try:
                    await self.gateway.close()
                    await http.close()
                except Exception:  # noqa: BLE001
                    pass
            if not self._closed:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300)

    async def close(self) -> None:
        self._closed = True
        for gid in list(self.sessions):
            await self.leave(gid)
        if self.gateway is not None:
            try:
                await self.gateway.close()
            except Exception:  # noqa: BLE001
                pass
        await self.planner.close()
        self.mood.maybe_save(force=True)
        await self.state.close()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    async def _dispatch(self, event: str, data: Any) -> None:
        try:
            await self._handle(event, data)
        except Exception:  # noqa: BLE001 - one bad event never takes the connection down
            log.exception("Fluxer event %s failed", event)

    async def _handle(self, event: str, data: Any) -> None:
        self.state.on_event(event, data)
        if event == "READY":
            self.state.on_ready(data)
            self.user = self.state.user_from(data["user"])
            first = not self._ready.is_set()
            self._ready.set()
            log.info("Logged in to Fluxer as %s (%s), %d server(s)", self.user.name, self.user.id,
                     len(self.state.guilds))
            if first:
                self._start_jobs()
        elif event == "GUILD_CREATE":
            guild = self.state.guilds.get(int(data["id"]))
            if guild is not None:
                await self._load_members(guild)
                log.info("Fluxer server: %s (%d members)", guild.name, len(guild.members))
                if self.fx.guild_ids and guild.id not in self.fx.guild_ids:
                    log.warning("Fluxer server %s isn't in fluxer.guild_ids - ignoring it", guild.name)
                await self._auto_join(guild)
        elif event == "MESSAGE_CREATE":
            await self.on_message(self.state.message_from(data))
        elif event == "VOICE_STATE_UPDATE":
            await self.on_voice_state(data)
        elif event == "VOICE_SERVER_UPDATE":
            fut = self._voice_waiters.pop(int(data.get("guild_id") or 0), None)
            if fut is not None and not fut.done():
                fut.set_result(data)

    async def _load_members(self, guild: FxGuild) -> None:
        """GUILD_CREATE only carries some members (those in voice, and us), so fetch the whole list - member
        lookups, roasts, the roster and the server owner's name need it. Kept current by member events after."""
        after = 0
        try:
            while True:
                params = {"limit": "1000", **({"after": str(after)} if after else {})}  # after=0 is refused
                batch = await self.state.request("GET", f"/guilds/{guild.id}/members", params=params)
                for m in batch or []:
                    guild.set_member(m)
                if not batch or len(batch) < 1000:
                    break
                after = max(int(m["user"]["id"]) for m in batch)
        except FxHTTPError as e:
            log.warning("Couldn't load the member list of %s: %s", guild.name, e)

    def _start_jobs(self) -> None:
        if self._started_jobs:
            return
        self._started_jobs = True
        self.profiles.start()
        self.planner.start()
        self.lore.start()

    def _allowed_guild(self, guild: FxGuild | None) -> bool:
        return guild is None or not self.fx.guild_ids or guild.id in self.fx.guild_ids

    # ------------------------------------------------------------------ discord.Client-ish lookups (shared code)

    def get_guild(self, gid: int) -> FxGuild | None:
        return self.state.guilds.get(gid)

    def get_channel(self, cid: int) -> FxChannel | None:
        return self.state.channels.get(cid)

    async def fetch_channel(self, cid: int) -> FxChannel:
        return self.state.channel_from(await self.state.request("GET", f"/channels/{cid}"))

    def get_user(self, uid: int) -> FxUser | None:
        return self.state.users.get(uid)

    async def fetch_user(self, uid: int) -> FxUser:
        return self.state.user_from(await self.state.request("GET", f"/users/{uid}"))

    # ------------------------------------------------------------------ live view / mute

    def feed(self, guild, kind: str, who: str, text: str, where: str = "") -> None:
        if guild is not None:  # into the dashboard's Live view, tagged as Fluxer
            self.host.feed(guild, kind, who, text, where, platform="fluxer")

    def _load_muted(self) -> set[int]:
        try:
            return {int(g) for g in json.loads(self.muted_path.read_text())}
        except (FileNotFoundError, ValueError, TypeError):
            return set()

    def is_muted(self, guild_id: int | None) -> bool:
        return guild_id in self.muted

    def set_muted(self, guild_id: int, muted: bool) -> None:
        (self.muted.add if muted else self.muted.discard)(guild_id)
        self.muted_path.write_text(json.dumps(sorted(self.muted)))
        if muted and (session := self.sessions.get(guild_id)):
            session.interrupt()

    # ------------------------------------------------------------------ voice

    def _text_channel(self, guild: FxGuild) -> FxChannel | None:
        """Where a voice session posts clips and reminders: the channel !join came from, else #general, else
        the first text channel."""
        if (ch := self._session_text.get(guild.id)) is not None:
            return ch
        texts = sorted((c for c in guild.channels.values() if c.type == FxChannel.TEXT), key=lambda c: c.id)
        return next((c for c in texts if c.name == "general"), texts[0] if texts else None)

    async def join_channel(self, channel: FxChannel, greet: bool = True, text: FxChannel | None = None) -> VoiceSession:
        guild = channel.guild
        if text is not None:
            self._session_text[guild.id] = text
        session = self.sessions.get(guild.id)
        if session and session.vc.is_connected():
            if session.vc.channel is channel:
                return session
            await self.leave(guild.id)
        fut = self._voice_waiters[guild.id] = asyncio.get_running_loop().create_future()
        await self.gateway.update_voice_state(guild_id=str(guild.id), channel_id=str(channel.id),
                                              self_mute=False, self_deaf=False)
        try:
            server = await asyncio.wait_for(fut, 15)
        except asyncio.TimeoutError:
            self._voice_waiters.pop(guild.id, None)
            raise RuntimeError("Fluxer didn't answer the voice join (does the bot have Connect in that channel?)")
        vc = FxVoiceClient(guild, channel, self._voice_lost)
        await vc.connect(server["endpoint"], server["token"])
        session = VoiceSession(self, vc)
        session._fx_seen = {m.id for m in channel.members}  # for join/leave announcements (on_voice_state)
        session.start()
        self.sessions[guild.id] = session
        people = session.humans_in_channel()
        if greet and self.cfg.voice.greet_on_join and people:
            session.post_event(f"You ({self.cfg.bot.name}) just joined the voice channel. Already here: "
                               f"{', '.join(people)}. Say a quick hi to them in your own words - you're the one "
                               "arriving, so don't greet yourself.", respond=True)
        return session

    async def leave(self, guild_id: int) -> bool:
        session = self.sessions.pop(guild_id, None)
        if session is None:
            return False
        await session.close()
        if self.gateway is not None:
            try:
                await self.gateway.update_voice_state(guild_id=str(guild_id), channel_id=None)
            except Exception:  # noqa: BLE001
                pass
        return True

    def _voice_lost(self, vc: FxVoiceClient) -> None:
        """LiveKit dropped us: rejoin if people are still there."""
        async def rejoin():
            session = self.sessions.get(vc.guild.id)
            if session is None or session.vc is not vc:
                return
            await self.leave(vc.guild.id)
            await asyncio.sleep(3)
            if any(not m.bot for m in vc.channel.members) and not self._closed:
                log.info("Rejoining #%s after losing the voice connection", vc.channel)
                try:
                    await self.join_channel(vc.channel, greet=False)
                except Exception:  # noqa: BLE001
                    log.exception("Rejoining Fluxer voice failed")
        asyncio.create_task(rejoin())

    async def _auto_join(self, guild: FxGuild) -> None:
        cid = self.fx.auto_join_channel_id
        ch = guild.channels.get(cid) if cid else None
        if ch is not None and ch.is_voice and guild.id not in self.sessions:
            try:
                await self.join_channel(ch)
            except Exception:  # noqa: BLE001
                log.exception("Fluxer auto-join failed")

    async def on_voice_state(self, data: dict) -> None:
        gid, uid = int(data.get("guild_id") or 0), int(data.get("user_id") or 0)
        session = self.sessions.get(gid)
        if session is None:
            return
        if self.user is not None and uid == self.user.id:
            if not data.get("channel_id"):  # someone disconnected us (our own leave() drops the session first)
                log.info("Disconnected from Fluxer voice in %s", session.vc.guild)
                kicked_from = session.vc.channel
                said_bye = time.monotonic() - session.bye_at < 120  # it was on its way out anyway
                await self.leave(gid)
                if said_bye:
                    log.info("Disconnected right after saying bye / being asked to leave - staying gone")
                elif self.cfg.voice.get("boomerang"):
                    asyncio.create_task(self._boomerang(kicked_from))
            return
        guild = session.vc.guild
        member = guild.get_member(uid)
        if member is None or member.bot:
            return
        cid = int(data.get("channel_id") or 0)
        here = session.vc.channel.id
        seen = session._fx_seen  # who was here before this update (the cache already has the new state)
        if cid == here and uid not in seen:
            seen.add(uid)
            session.post_event(f"{member.display_name} joined the voice channel", respond=self.cfg.voice.greet_on_join)
        elif cid != here and uid in seen:
            seen.discard(uid)
            session.post_event(f"{member.display_name} left the voice channel", respond=False)
        if self.cfg.voice.auto_leave_when_empty and not any(not m.bot for m in session.vc.channel.members):
            log.info("Fluxer voice channel empty, leaving")
            await self.leave(gid)

    async def _boomerang(self, channel: FxChannel) -> None:
        """Like the Discord bot: force-disconnected -> come straight back, call it out, leave on its own terms.
        Once per 10 minutes per server. Fluxer has no audit log for bots, so it doesn't know who did it."""
        guild = channel.guild
        now = time.monotonic()
        last = getattr(self, "_last_boomerang", {})
        self._last_boomerang = last
        if now - last.get(guild.id, -1e9) < 600:
            log.info("🪃 kicked again from Fluxer #%s - staying gone", channel.name)
            return
        last[guild.id] = now
        await asyncio.sleep(float(self.cfg.voice.get("boomerang_delay_s") or 4))
        if not any(not m.bot for m in channel.members) or self._closed:
            return
        try:
            session = await self.join_channel(channel, greet=False)
        except Exception:  # noqa: BLE001
            log.exception("Boomerang rejoin failed")
            return
        session.leave_after_reply = True
        log.info("🪃 back in Fluxer #%s to talk shit to whoever kicked me", channel.name)
        session.post_event(
            "Someone just force-disconnected you - like a coward - and you came straight back to call it out "
            "before leaving on your own terms. You don't know who did it: roast whoever it was, hard - two short, "
            "savage, funny lines (the power trip, the cowardly click). No hedging, no being a good sport about it. "
            "Then one cocky line on your way out.", respond=True)

    async def post_clip(self, channel, audio, who: set[int], requester: str) -> str:
        seconds = audio.size / 48000
        guild = getattr(channel, "guild", None)
        names = [self.cfg.bot.name if uid == getattr(self.user, "id", 0)
                 else (m.display_name if guild and (m := guild.get_member(uid)) else str(uid)) for uid in sorted(who)]
        target = self._text_channel(guild) if guild is not None else channel
        try:
            mp3 = await asyncio.to_thread(encode_mp3, audio, int(self.cfg.clips.bitrate))
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            await target.send(f"🎬 **Clip** · {seconds:.0f}s · by {requester}"
                              + (f" · feat. {', '.join(names)}" if names else ""),
                              file=FxFile(io.BytesIO(mp3), f"clip_{stamp}.mp3"))
        except (FxHTTPError, AttributeError) as e:
            log.warning("Fluxer clip upload failed: %s", e)
            return f"[You tried to post a clip but Fluxer refused ({e}). Tell {requester}.]"
        self.counters["clips"] += 1
        log.info("🎬 clip %.0fs by %s on Fluxer", seconds, requester)
        who_txt = f" ({', '.join(names)} talking)" if names else ""
        return f"[You just clipped the last {seconds:.0f} seconds{who_txt} and posted it in the chat. React in a few words.]"

    def post_transcript(self, text: str) -> None:
        cid = self.fx.transcript_channel_id
        channel = self.get_channel(cid) if cid else None
        if channel:
            for chunk in split_message(text):
                asyncio.create_task(channel.send(chunk))

    def roast_note(self, requester, target, lines: list[str], mode: str) -> str:
        row = self.profiles.store.get(target.id)
        usable = row is not None and not row["opted_out"]
        log.info("%s %s asked for a %s of %s (Fluxer)", "🔥" if mode == "roast" else "💘", requester.display_name,
                 mode, target.display_name)
        return members.roast_note(requester.display_name, target, row["profile"] if usable else "", lines, "", mode)

    # ------------------------------------------------------------------ text chat

    def _trigger(self, message: FxMessage) -> str | None:
        fx = self.fx
        if message.guild is None:
            return "dm" if fx.respond_in_dms else None
        if message.channel.id in fx.text_channel_ids:
            return "channel"
        if fx.respond_to_mentions and any(u.id == self.user.id for u in message.mentions):
            return "mention"
        ref = message.referenced
        if fx.respond_to_replies and ref is not None and ref.author.id == self.user.id:
            return "reply"
        if self._name_re and self._name_re.search(message.content):
            return "name"
        return None

    def _clean(self, message: FxMessage) -> str:
        """<@id> -> @name (and our own mention removed), like discord.py's clean_content."""
        text = message.content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        guild = message.guild

        def user(m: re.Match) -> str:
            uid = int(m.group(1))
            u = (guild.get_member(uid) if guild else None) or self.get_user(uid)
            return f"@{u.display_name}" if u else "@someone"

        def channel(m: re.Match) -> str:
            ch = self.get_channel(int(m.group(1)))
            return f"#{ch.name}" if ch else "#channel"

        def role(m: re.Match) -> str:
            r = guild.roles.get(int(m.group(1))) if guild else None
            return f"@{r.name}" if r else "@role"

        text = re.sub(r"<@!?(\d+)>", user, text)
        text = re.sub(r"<#(\d+)>", channel, text)
        return re.sub(r"<@&(\d+)>", role, text).strip()

    def _system_prompt(self, author, channel: FxChannel) -> str:
        where = "in a DM" if channel.guild is None else f"in #{channel.name}"
        mine = self.profiles.self_note(channel.guild)
        return (f"{self.persona()}\n\n{self.cfg.bot.time_prompt.strip()}\n\nYou're chatting {where}."
                + (f" {mine}" if mine else "") + f" The person talking to you now:\n{self.profiles.describe(author)}")

    def _maybe_fresh(self, channel: FxChannel) -> None:
        now, fresh = time.time(), float(self.cfg.bot.get("fresh_after_min") or 0) * 60
        quiet = now - self._last_active.get(channel.id, now)
        if fresh > 0 and quiet > fresh and self.text_history.get(channel.id):
            log.info("🆕 New conversation in Fluxer #%s (quiet for %.0f min)", channel.name, quiet / 60)
            self.text_history.pop(channel.id, None)
            self._last_seen.pop(channel.id, None)
        self._last_active[channel.id] = now

    async def _backfill(self, message: FxMessage, history: list[dict]) -> None:
        """What people said since we last looked at this channel, so a name-drop comes with its context."""
        n = int(self.fx.context_messages)
        if n <= 0 or message.guild is None or message.channel.id in self.fx.text_channel_ids:
            return
        try:
            older = await message.channel.history(limit=n, before=message.id,
                                                  after=self._last_seen.get(message.channel.id, 0))
        except FxHTTPError as e:
            log.debug("No Fluxer channel history for context: %s", e)
            return
        fresh = float(self.cfg.bot.get("fresh_after_min") or 0) * 60
        now = datetime.now().astimezone()
        for m in sorted(older, key=lambda m: m.id):
            if m.author.bot or m.id == message.id or not m.content.strip():
                continue
            if fresh > 0 and m.created_at and (now - m.created_at).total_seconds() > fresh:
                continue
            history.append({"role": "user", "content": f"{m.author.display_name}: {self._clean(m)}"})

    async def _command_or_search(self, content: str, message: FxMessage, history: list[dict]) -> str | None:
        if self.planner.wants(content):
            voice = getattr(message.author, "voice", None)
            note = await self.planner.handle(content, message.author, message.channel,
                                             [m["content"] for m in history[-4:-1]],
                                             voice.channel.members if voice and voice.channel else [])
            if note:
                return note
        if found := calc.note(content):
            log.info("🧮 %s", found)
            return found
        voice = getattr(message.author, "voice", None)
        people = [m.display_name for m in voice.channel.members if not m.bot] if voice and voice.channel else []
        if self.cfg.fun.enabled and (found := fun.note(content, people, message.author.display_name)):
            log.info("🎲 %s", found)
            return found
        if found := await self.weather.note(content):
            return found
        if mode := members.roast_request(content):
            lines = [(ln.split(": ", 1)[0], ln.split(": ", 1)[1]) for m in history[-12:] if m["role"] == "user"
                     and isinstance(m["content"], str) for ln in m["content"].splitlines()
                     if ": " in ln and not ln.lstrip().startswith("[")]
            pool = [*message.mentions, *(message.guild.members if message.guild else [message.author])]
            if (target := members.roast_target(pool, content, message.author, [n for n, _ in lines])) is not None:
                return self.roast_note(message.author, target, [t for n, t in lines if n == target.display_name], mode)
        if message.guild is not None and (note := await members.lookup(
                message.guild, message.author, content, skip_names=[*self.cfg.voice.wake_words, self.cfg.bot.name])):
            return note
        return await self.search.lookup(history, context=self.profiles.presence_note([message.author]))

    async def on_message(self, message: FxMessage) -> None:
        if self.user is None or message.author.bot or message.author.id == self.user.id:
            return
        if message.type not in (0, 19) or not self._allowed_guild(message.guild):
            return  # 19 = reply; skip joins, pins and other system messages
        if (m := self._cmd_re.match(message.content)) and await self._command(message, m.group(1).lower(),
                                                                                m.group(2).strip()):
            return
        trigger = self._trigger(message)
        if trigger is None:
            return
        content = self._clean(message)
        if not content and not self.vision.images_in(message):
            return
        author, channel, guild = message.author, message.channel, message.guild
        where = f"#{channel.name}"
        if guild and self.is_muted(guild.id):
            log.info("(muted) 💬 Fluxer [%s] %s: %s", trigger, author.display_name, content)
            self.feed(guild, "text_in", author.display_name, content, where)
            return
        self.profiles.activity()
        typing = asyncio.create_task(self._typing(channel))
        self._maybe_fresh(channel)
        history = self.text_history.setdefault(channel.id, [])
        try:
            # Images: this message's, the one it replies to (unless that's ours).
            posted = await self.vision.fetch(message)
            ref = message.referenced
            ref_posted = await self.vision.fetch(ref) if ref is not None and ref.author.id != self.user.id else []
            parts = [content, *[PLACEHOLDER] * len(posted)]
            if ref_posted:
                parts.append(f"(replying to {ref.author.display_name}'s " + " ".join([PLACEHOLDER] * len(ref_posted)) + ")")
            said = " ".join(p for p in parts if p)
            line = f"{author.display_name}: {said}"
            log.info("💬 Fluxer [%s] %s", trigger, line)
            self.feed(guild, "text_in", author.display_name, said, where)
            await self._backfill(message, history)
            self._last_seen[channel.id] = message.id
            note = self.profiles.presence_note([author])
            turn = f"{note}\n{line}" if note and not any(note in m["content"] for m in history) else line
            entry = {"role": "user", "content": turn}
            history.append(entry)
            trim_history(history, int(self.cfg.bot.max_history_messages))
            to_caption = [(entry, img) for img in posted + ref_posted]
            messages = [{"role": "system", "content": self._system_prompt(author, channel)}] + history
            lookup = asyncio.create_task(self._command_or_search(turn, message, history))
            names = [author.display_name, self.cfg.bot.name, *self.cfg.voice.wake_words]
            memories = asyncio.create_task(self.lore.recall(guild.id if guild else None, turn, names))
            mood = (asyncio.get_running_loop().run_in_executor(
                self.mood_executor, self.mood.read, author.id, author.display_name, content)
                if self.mood.model is not None else None)
            try:
                avatars = await self._avatars_asked_about(message, message.content)
                seen = await self.vision.prepare((avatars + posted + ref_posted)[: int(self.cfg.vision.max_images)])
                found = await lookup
                try:
                    lore_note = await memories
                except Exception as e:  # noqa: BLE001
                    log.warning("Lore recall failed: %s", e)
                    lore_note = None
            finally:
                lookup.cancel()
                memories.cancel()
            reads = [r for r in [await mood] if r] if mood else []
            mood_note = self.mood.choose(guild.id if guild else 0, reads)
            prompt = ((f"{found}\n\n" if found else "") + (f"{lore_note}\n" if lore_note else "")
                      + (f"{mood_note}\n" if mood_note else "") + f"{now_note()}\n{turn}")
            messages[-1] = {"role": "user", "content": prompt}
            if seen:
                messages[-1] = {"role": "user", "content": image_parts(prompt, seen)}
                try:
                    reply = await self.llm.complete(messages)
                except Exception as e:  # noqa: BLE001
                    log.warning("LLM call with %d image(s) failed (%s) - retrying without them", len(seen), e)
                    messages[-1] = {"role": "user", "content": prompt}
                    reply = await self.llm.complete(messages)
            else:
                reply = await self.llm.complete(messages)
        except Exception as e:  # noqa: BLE001
            log.exception("LLM error (Fluxer)")
            if history and history[-1]["role"] == "user":
                history.pop()
            await self._send(message.reply, f"⚠️ LLM error: `{e}`")
            return
        finally:
            typing.cancel()
            self.profiles.activity(cancel=False)

        self.profiles.observe(author.id, author.display_name, line, author)
        self.lore.observe(guild, author.id, line)
        speakers = {author.display_name, *self.host._creator_names}
        for h in history:
            c = h["content"] if isinstance(h["content"], str) else ""
            if h["role"] == "user" and ":" in c[:40]:
                speakers.add(c.split(":", 1)[0].strip())
        guard = SpeakerGuard(self.cfg.bot.name, speakers)
        repeats = RepeatGuard(recent_replies(history) + prompt_examples(messages[0]["content"]))

        def clean(text: str) -> tuple[str, list[str]]:
            stripped = EchoGuard([said]).strip(text, guard.strip_label)
            return repeats.filter(guard.clean(stripped))

        raw = reply
        reply, dropped = clean(reply)
        if not reply and raw.strip():
            try:
                raw = await self.llm.complete(messages)
                reply, dropped = clean(raw)
            except Exception as e:  # noqa: BLE001
                log.warning("Retry after an empty reply failed: %s", e)
        for s in dropped:
            log.info("(repeat skipped: %s)", s)
        if not reply:
            log.warning("No Fluxer text reply to %s: the filters left nothing of %.300r", author.display_name, raw)
            return
        log.info("🤖 Fluxer [text] %s", reply)
        self.feed(guild, "text_out", self.cfg.bot.name, reply, where)
        self._last_active[channel.id] = time.time()
        self.counters["text replies"] += 1
        history.append({"role": "assistant", "content": reply})
        self.mood.replied(guild.id if guild else 0, reads)
        self.profiles.observe_reply([author.id], f"{self.cfg.bot.name}: {reply}")
        self.lore.observe(guild, 0, f"{self.cfg.bot.name}: {reply}")
        self.vision.caption_later(to_caption)  # so later turns still know what the images were
        for i, chunk in enumerate(split_message(reply)):
            await self._send(message.reply if i == 0 else channel.send, chunk)

    async def _avatars_asked_about(self, message: FxMessage, content: str) -> list:
        """Profile pictures to look at when the message talks about a pfp: ours, the people mentioned, or theirs."""
        if not (self.vision.enabled and _AVATAR_RE.search(content)):
            return []
        if _YOUR_AVATAR_RE.search(content):
            people = [self.user]
        else:
            people = [m for m in message.mentions if m.id != self.user.id] or [message.author]
        people = [p for p in people[:3] if p.display_avatar is not None]
        found = [await self.vision.fetch_avatar(p) for p in people]
        for img, p in zip(found, people):
            if img and p.id == self.user.id:
                img.label = "your own profile picture"
        return [img for img in found if img]

    @staticmethod
    async def _send(fn, text: str) -> None:
        try:
            await fn(text)
        except FxHTTPError as e:
            log.warning("Fluxer send failed: %s", e)

    @staticmethod
    async def _typing(channel: FxChannel) -> None:
        async with channel.typing():
            await asyncio.Event().wait()

    # ------------------------------------------------------------------ ! commands
    # Discord's slash commands as text commands (Fluxer has no bot slash commands yet). Discord answers some of
    # them privately ("only you can see this"); Fluxer can't, so those answers go by DM.

    def _is_owner(self, user) -> bool:
        """Bot-wide settings (the model): creators + fluxer.admin_ids, like Discord's creator_ids + dashboard admins."""
        return user.id in (self.fx.get("creator_ids") or []) or user.id in self.fx.admin_ids

    def _is_manager(self, message: FxMessage) -> bool:
        perms = getattr(message.author, "guild_permissions", None)
        return self._is_owner(message.author) or bool(perms and (perms.manage_guild or perms.administrator))

    def _is_admin(self, message: FxMessage) -> bool:
        return self._is_manager(message)

    async def _dm(self, message: FxMessage, text: str) -> None:
        """A private answer: DM, with a short note in the channel. Falls back to the channel if DMs fail."""
        chunks = split_message(text)
        try:
            for chunk in chunks:
                await message.author.send(chunk)
            if message.guild is not None:
                await self._send(message.reply, "📬 Sent you a DM.")
        except FxHTTPError:
            for i, chunk in enumerate(chunks):
                await self._send(message.reply if i == 0 else message.channel.send, chunk)

    def _target(self, message: FxMessage):
        """Whose profile a command is about: someone @mentioned (managers only) or the author. None = not allowed."""
        others = [u for u in message.mentions if u.id != self.user.id and u.id != message.author.id]
        if not others:
            return message.author
        return others[0] if self._is_manager(message) else None

    async def _command(self, message: FxMessage, cmd: str, arg: str) -> bool:
        """True if it was one of ours (handled), False to treat it as chat."""
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return False
        try:
            await handler(message, arg)
        except Exception as e:  # noqa: BLE001
            log.exception("Fluxer command %s failed", cmd)
            await self._send(message.reply, f"⚠️ `{cmd}` failed: {e}")
        return True

    async def _cmd_help(self, message: FxMessage, arg: str) -> None:
        await self._send(message.reply, self._help_text())

    async def _cmd_join(self, message: FxMessage, arg: str) -> None:
        voice = getattr(message.author, "voice", None)
        if message.guild is None or voice is None or voice.channel is None:
            await self._send(message.reply, f"Join a voice channel first, then `{self.fx.prefix}join`.")
            return
        try:
            await self.join_channel(voice.channel, text=message.channel)
        except Exception as e:  # noqa: BLE001
            log.exception("Fluxer join failed")
            await self._send(message.reply, f"Couldn't join: {e}")

    async def _cmd_leave(self, message: FxMessage, arg: str) -> None:
        ok = message.guild is not None and await self.leave(message.guild.id)
        await self._send(message.reply, "👋" if ok else "I'm not in a voice channel here.")

    async def _cmd_stop(self, message: FxMessage, arg: str) -> None:
        if message.guild is not None and (s := self.sessions.get(message.guild.id)):
            s.interrupt()
            await self._send(message.reply, "🤐")
        else:
            await self._send(message.reply, "I'm not in a voice channel here.")

    async def _cmd_reset(self, message: FxMessage, arg: str) -> None:
        if not self._is_admin(message):
            await self._send(message.reply, "Admins only.")
            return
        self.text_history.pop(message.channel.id, None)
        self._last_seen.pop(message.channel.id, None)
        if message.guild is not None and (s := self.sessions.get(message.guild.id)):
            s.reset()
        await self._send(message.reply, "🧹 Conversation memory cleared here.")

    async def _cmd_say(self, message: FxMessage, arg: str) -> None:
        if not self._is_admin(message):
            await self._send(message.reply, "Admins only.")
        elif message.guild is None or not (s := self.sessions.get(message.guild.id)):
            await self._send(message.reply, "I'm not in a voice channel here.")
        elif not arg:
            await self._send(message.reply, f"Usage: `{self.fx.prefix}say <text>`")
        else:
            await s.say(arg)

    async def _cmd_llm(self, message: FxMessage, arg: str) -> None:
        """!llm <endpoint> [model] [text|voice|both]"""
        if not self._is_owner(message.author):
            await self._send(message.reply, "Only the bot's owner can do that.")
            return
        parts = arg.split()
        if not parts or parts[0] not in self.llm.endpoints:
            await self._send(message.reply, f"Usage: `{self.fx.prefix}llm <endpoint> [model] [text|voice|both]` - "
                                            f"endpoints: {', '.join(self.llm.endpoints)}\n"
                                            f"Now: text `{self.llm.describe(False)}` · voice `{self.llm.describe(True)}`")
            return
        endpoint, rest = parts[0], parts[1:]
        scope = rest.pop() if rest and rest[-1] in ("text", "voice", "both") else "both"
        if scope in ("both", "text"):
            self.llm.text_endpoint = endpoint
        if scope in ("both", "voice"):
            self.llm.voice_endpoint = endpoint
        if rest:
            self.llm.model_overrides[endpoint] = rest[0]
        await self.llm.warmup(attempts=1)
        await self._send(message.reply, f"Text: `{self.llm.describe(False)}`\nVoice: `{self.llm.describe(True)}`\n"
                                        "-# Shared with the Discord bot (same models).")

    async def _cmd_models(self, message: FxMessage, arg: str) -> None:
        if not self._is_owner(message.author):
            await self._send(message.reply, "Only the bot's owner can do that.")
            return
        name = arg.strip() or self.llm.text_endpoint
        if name not in self.llm.endpoints:
            await self._send(message.reply, f"Unknown endpoint. Options: {', '.join(self.llm.endpoints)}")
            return
        try:
            ids = await self.llm.list_models(name)
            await self._send(message.reply, f"**{name}**\n```\n{(chr(10).join(ids) or '(none)')[:1800]}\n```")
        except Exception as e:  # noqa: BLE001
            await self._send(message.reply, f"Failed: `{e}`")

    async def _cmd_status(self, message: FxMessage, arg: str) -> None:
        await self._send(message.reply, self._status_text(message.guild))

    async def _cmd_clip(self, message: FxMessage, arg: str) -> None:
        guild = message.guild
        s = self.sessions.get(guild.id) if guild else None
        if s is None or s.clips is None:
            await self._send(message.reply, "I'm not in a voice channel here (or clips are off).")
            return
        try:
            seconds = min(max(float(arg or self.cfg.clips.default_s), 3.0), float(self.cfg.clips.buffer_s))
        except ValueError:
            seconds = float(self.cfg.clips.default_s)
        audio, who = s.clips.clip(time.monotonic(), seconds)
        if audio.size < 48000:
            await self._send(message.reply, "I haven't heard anything to clip yet.")
            return
        self._session_text[guild.id] = message.channel
        note = await self.post_clip(message.channel, audio, who, message.author.display_name)
        if "refused" in note:
            await self._send(message.reply, "Couldn't upload the clip.")

    async def _cmd_tuning(self, message: FxMessage, arg: str) -> None:
        if message.guild is None or not self._is_admin(message):
            await self._send(message.reply, "Admins only, in a server.")
            return
        t = self.tuning.for_guild(message.guild.id)
        if arg.strip().lower() == "reset":
            t.reset()
        lines = [f"**Reply length:** {LEVEL_NAMES[t.level]} (level {t.level}/3)"
                 + (" · next reply may run longer" if t.more_next else ""),
                 f"**Follow-up window:** {t.followup_s:.0f}s (configured {self.tuning.base_followup:.0f}s)",
                 f"**Uninterrupted streak:** {t.streak}"]
        if not self.tuning.enabled:
            lines.insert(0, "_Self-tuning is off in config._")
        if t.log:
            lines.append("**Recent adjustments:**")
            lines += [f"<t:{int(ts)}:R> {why}" for ts, why in reversed(t.log)]
        if self.mood.enabled:
            lines.append("**Mood strategies** (★ = working best, wins/tries):")
            lines += [f"`{line}`" for line in self.mood.summary(message.guild.id)] or ["_nothing learned yet_"]
        await self._dm(message, "\n".join(lines))

    async def _cmd_remind(self, message: FxMessage, arg: str) -> None:
        """!remind <when> | <what>   or   !remind in 20 minutes to check the oven   (add "everyone" for the voice call)"""
        if not self.planner.enabled:
            await self._send(message.reply, "Reminders are turned off in config.")
            return
        p = self.fx.prefix
        everyone = bool(re.search(r"\beveryone\b", arg, re.I))
        text = re.sub(r"\beveryone\b", "", arg, flags=re.I).strip()
        if "|" in text:
            when, what = (x.strip() for x in text.split("|", 1))
        else:
            m = re.match(r"(.+?)\s+(?:to|that|about)\s+(.+)$", text, re.I)
            when, what = (m.group(1), m.group(2)) if m else (text, "")
        due = parse_when(when) if when else None
        if due is None or not what:
            await self._send(message.reply, f"Try `{p}remind in 20 minutes to check the oven` or "
                                            f"`{p}remind friday 6pm | game night` (add `everyone` for the whole call).")
            return
        user = message.author
        if err := self.planner.check_quota(user.id):
            await self._send(message.reply, f"Can't: {err}.")
            return
        if everyone:
            voice = getattr(user, "voice", None)
            if not voice or not voice.channel:
                await self._send(message.reply, "You're not in a voice channel.")
                return
            targets = [m.id for m in voice.channel.members if not m.bot]
        else:
            targets = [user.id]
        deliver = "dm" if targets == [user.id] else "channel"
        job_id = self.planner.add_reminder(user, targets, what, due.timestamp(), deliver, message.channel)
        whom = "you" if deliver == "dm" else f"{len(targets)} people"
        await self._send(message.reply, f"⏰ Reminder `#{job_id}` for {whom} {spoken_time(due.timestamp())}, "
                                        f"{'by DM' if deliver == 'dm' else 'here'}: {what}")

    async def _cmd_reminders(self, message: FxMessage, arg: str) -> None:
        """!reminders  |  !reminders cancel <id>"""
        uid = message.author.id
        m = re.match(r"(?:cancel|delete|remove)\s+#?(\d+)", arg.strip(), re.I)
        if m:
            row = self.planner.store.get(int(m.group(1)))
            if row is None or row["kind"] != "reminder" or (row["creator_id"] != uid and not self._is_manager(message)):
                await self._send(message.reply, "That's not one of your reminders.")
                return
            self.planner.cancel(row["id"])
            await self._send(message.reply, f"🗑️ Cancelled: {row['text']}")
            return
        rows = self.planner.store.reminders_for(uid)
        lines = [f"`#{r['id']}` {spoken_time(r['due'])} - {r['text']}"
                 + ("" if r["creator_id"] == uid else f" _(from {r['creator_name']})_") for r in rows[:20]]
        await self._dm(message, "\n".join(lines) + f"\n-# Cancel one with `{self.fx.prefix}reminders cancel <id>`"
                       if lines else "No reminders pending.")

    async def _cmd_poll(self, message: FxMessage, arg: str) -> None:
        """!poll question | choice, choice, choice [| 10 minutes] [multi]"""
        if not self.planner.enabled:
            await self._send(message.reply, "Polls are turned off in config.")
            return
        parts = [x.strip() for x in arg.split("|")]
        if len(parts) < 2:
            await self._send(message.reply, f"Usage: `{self.fx.prefix}poll Pizza tonight? | yes, no, maybe | 10 minutes`")
            return
        question = parts[0]
        choices = list(dict.fromkeys(o.strip() for o in re.split(r"[,\n]", parts[1]) if o.strip()))
        multiple = any(x.lower() in ("multi", "multiple") for x in parts[2:])
        duration = next((x for x in parts[2:] if x.lower() not in ("multi", "multiple")), "")
        if not 2 <= len(choices) <= 10:
            await self._send(message.reply, "A poll needs 2-10 choices, separated by commas.")
            return
        seconds = parse_duration(duration, float(self.cfg.reminders.poll_minutes) * 60)
        if seconds is None:
            await self._send(message.reply, f"Couldn't read the duration \"{duration}\". Try `5 minutes`.")
            return
        if err := self.planner.check_quota(message.author.id):
            await self._send(message.reply, f"Can't: {err}.")
            return
        poll, ends = self.planner.build_poll(question, choices, seconds, multiple)
        msg = await message.channel.send(f"Poll by {message.author.display_name}", poll=poll)
        self.planner.track_poll(msg, message.author, question, ends)

    async def _cmd_profile(self, message: FxMessage, arg: str) -> None:
        target = self._target(message)
        if target is None:
            await self._send(message.reply, "You can only do that for yourself.")
            return
        text = self.profiles.summary(target.id)
        if not self.profiles.enabled:
            text = "_Profiles are disabled in config._\n" + text
        await self._dm(message, text[:1990])

    async def _cmd_forget(self, message: FxMessage, arg: str) -> None:
        target = self._target(message)
        if target is None:
            await self._send(message.reply, "You can only do that for yourself.")
            return
        self.profiles.store.forget(target.id)
        self.mood.forget(target.id)
        self.lore.forget(target.id)
        self.links.unlink("fluxer", target.id)
        await self._send(message.reply, f"🗑️ Forgot everything about {target.display_name}.")

    async def _cmd_profiling(self, message: FxMessage, arg: str) -> None:
        a = arg.strip().lower()
        if a not in ("on", "off"):
            await self._send(message.reply, f"`{self.fx.prefix}profiling off` stops me building a profile of you "
                                            f"(and deletes it); `{self.fx.prefix}profiling on` turns it back on.")
            return
        enabled = a == "on"
        self.profiles.store.set_opted_out(message.author.id, not enabled)
        if not enabled:
            self.lore.forget(message.author.id)
        await self._send(message.reply, "✅ I'll remember things about you again." if enabled
                         else "🙈 Got it - I won't build a profile of you, and I deleted what I had.")

    async def _cmd_quote(self, message: FxMessage, arg: str) -> None:
        """!quote (random) · !quote <words/name> (search) · reply to a message with !quote (save it) ·
        !quote delete <id> (admins / whoever saved it)"""
        if not self.cfg.quotes.enabled:
            await self._send(message.reply, "The quote book is turned off in config.")
            return
        if message.guild is None:
            await self._send(message.reply, "Quotes live in a server - try it there.")
            return
        gid, book = message.guild.id, self.quotes
        ref = message.referenced
        if ref is not None and not arg:
            text = self._clean(ref)
            if not text:
                await self._send(message.reply, "Nothing to quote in that message.")
            elif (qid := book.exists(gid, text)) is not None:
                await self._send(message.reply, f"Already in the book as #{qid}.")
            else:
                when = ref.created_at.timestamp() if ref.created_at else None
                qid = book.add(gid, ref.author.id, ref.author.display_name, text, when, message.author.display_name, "text")
                log.info("💬 Fluxer quote #%d saved by %s: %s", qid, message.author.display_name, text[:80])
                await self._send(message.reply, quotes.show(book.get(qid)) + " · saved")
            return
        m = re.match(r"(?:delete|remove)\s+#?(\d+)$", arg.strip(), re.I)
        if m:
            row = book.get(int(m.group(1)))
            if row is None or row["guild_id"] != gid:
                await self._send(message.reply, "No quote with that number here.")
            elif not (self._is_manager(message) or row["saved_by"] == message.author.display_name):
                await self._send(message.reply, "Only admins or whoever saved it can delete it.")
            else:
                book.delete(row["id"])
                await self._send(message.reply, f"🗑️ Deleted quote #{row['id']}.")
            return
        if arg.strip():
            rows = book.search(gid, arg.strip())
            await self._send(message.reply, "\n".join(quotes.show(r) for r in rows)[:1990]
                             or f"No quotes matching \"{arg.strip()}\".")
            return
        row = book.random(gid)
        await self._send(message.reply, quotes.show(row) if row else
                         "The quote book is empty. Say \"Static, quote that\" in voice, or reply to a message with "
                         f"`{self.fx.prefix}quote`.")

    async def _cmd_link(self, message: FxMessage, arg: str) -> None:
        """!link (get a code for Discord) · !link <code> (enter a code from Discord's /link)"""
        if not self.links.enabled:
            await self._send(message.reply, "Account linking is turned off.")
            return
        u = message.author
        if arg.strip():
            ok, msg = self.links.complete("fluxer", u.id, u.name, arg)
            if ok:
                log.info("🔗 %s linked their Discord and Fluxer accounts", u.name)
            await self._send(message.reply, msg.replace("with unlink", f"with `{self.fx.prefix}unlink`"))
            return
        code = self.links.start("fluxer", u.id, u.name)
        await self._dm(message, f"🔗 Your code: **{code}** - on Discord, use `/link code:{code}` within 10 minutes.\n"
                                "Once linked, Static knows both accounts are you and shares what it remembers about you "
                                f"between them. Nothing else is shared; `{self.fx.prefix}unlink` undoes it.")

    async def _cmd_unlink(self, message: FxMessage, arg: str) -> None:
        done = self.links.unlink("fluxer", message.author.id)
        await self._send(message.reply, "🔗 Unlinked." if done else "You're not linked.")

    def _help_text(self) -> str:
        """!help, from the same command list as the web help page's Fluxer tab (fluxer/help.py)."""
        p, name = self.fx.prefix, self.cfg.bot.name
        lines = [f"**{name}** - talk to me by saying my name, @mentioning me or replying to me. In voice, say \"{name}\" "
                 "and talk. Out loud or in chat you can also ask me to clip or quote that, set reminders, start polls, "
                 "flip a coin, roll dice, pick someone, split teams, or check the weather."]
        groups: dict[str, list[str]] = {}
        for cmd, args, _desc, _who, group in fx_help.COMMANDS:
            groups.setdefault(group, []).append(f"`{p}{cmd}{(' ' + args) if args and args.startswith('[') else ''}`")
        lines += [f"**{g}:** " + " · ".join(cmds) for g, cmds in groups.items()]
        if self.host.cfg.dashboard.public_url:
            lines.append(f"-# Full guide: {self.host.cfg.dashboard.public_url.rstrip('/')}/help?p=fluxer")
        return "\n".join(lines)

    def _status_text(self, guild: FxGuild | None) -> str:
        up = time.time() - self.started_at
        turns = sorted(t["total_ms"] for t in list(self.voice_turns) if t.get("total_ms"))
        lat = (f"voice replies {turns[len(turns) // 2]:.0f} ms median, {turns[int(len(turns) * 0.9)]:.0f} ms p90 "
               f"({len(turns)} recent)" if turns else "no voice replies yet")
        voice = self.sessions.get(guild.id) if guild else None
        usage = getattr(self.llm, "usage", None)
        lines = [f"📡 **{self.cfg.bot.name}** on Fluxer · up {up / 3600:.1f} h · platform mode "
                 f"`{self.host.cfg.platform.mode}`",
                 f"🧠 text `{self.llm.describe(False)}` · voice `{self.llm.describe(True)}`",
                 f"🎙 {'in ' + voice.vc.channel.name if voice else 'not in voice here'} · {lat}",
                 f"🗂 {len(self.state.guilds)} Fluxer server(s) · {len(self.sessions)} voice call(s) · "
                 f"search {'on' if self.search.enabled else 'off'} · mood {'on' if self.mood.model else 'off'} · "
                 f"memories {'on' if self.lore.model else 'off'}"]
        if isinstance(usage, dict) and usage:
            lines.append("-# LLM usage is shared with Discord - see the dashboard for tokens/speed")
        return "\n".join(lines)
