"""Discord client: slash commands, text chat, voice session management."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from datetime import datetime, timedelta
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import voice_recv

from . import calc, fun, helpinfo, members, quotes
from .addressee import AddresseeCheck
from .clips import encode_mp3
from .llm import LLMRouter
from .lore import Lore
from .stt import create_stt
from .presence import Presence
from .web.server import INVITE_PERMS, Dashboard
from .profiles import ProfileManager
from .reminders import Planner, parse_duration, parse_when, spoken_time
from .search import WebSearch
from .sentiment import MoodReader
from .stats import StatsView
from .stats import build as build_stats
from .text_utils import (EchoGuard, RepeatGuard, SpeakerGuard, clean_for_speech, now_note, prompt_examples, recent_replies,
                         split_message, trim_history)
from .links import Links
from .quotes import QuoteBook
from .tts import TTS
from .weather import Weather
from .tuning import LEVEL_NAMES, TuningStore
from .vision import PLACEHOLDER, PostedImage, Vision, image_parts
from .voice_session import VoiceSession

# Server nicknames and display names differ per server and are often fancy-font/emoji, which confuses the
# model and TTS. Every prompt, voice label, log line and name match uses display_name, so it's the account
# username (unique, plain a-z0-9_.) everywhere. People are tracked by user id; names are only labels.
discord.Member.display_name = property(lambda self: self.name)
discord.user.BaseUser.display_name = property(lambda self: self.name)

log = logging.getLogger("voicebot")
MUTED_PATH = Path("data/muted.json")
CREATORS_PATH = Path("data/creators.json")

# "what do you think of my pfp" -> look at the actual avatar, not just the remembered description
_AVATAR_RE = re.compile(r"\b(?:pfps?|avatars?|profile ?(?:pics?|pictures?|photos?))\b", re.I)
_YOUR_AVATAR_RE = re.compile(r"\byour (?:pfp|avatar|profile ?(?:pic|picture|photo))\b", re.I)


class VoiceBot(discord.Client):
    def __init__(self, cfg, config_path="config.yaml"):
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: enable in the Developer Portal
        intents.voice_states = True
        intents.presences = bool(cfg.profiles.include_activities)  # privileged: enable in the Developer Portal
        # privileged: the full member list, so "what roles does Casey have?" finds people who aren't in voice
        # (fancy-font names included). Enable "Server Members Intent" in the Developer Portal first, or login fails.
        intents.members = bool(cfg.discord.get("members_intent"))
        # Replies are LLM text that people can steer ("say @everyone"): never let them ping anyone. Sends that
        # are meant to ping (reminders) pass their own allowed_mentions.
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self._creator_note = ""  # filled in setup_hook once we can look the creators up
        self._creator_names: list[str] = []
        self.cfg = cfg
        self.tree = app_commands.CommandTree(self)
        self.llm = LLMRouter(cfg)
        self.stt = None
        self.tts = None
        self.tts_cache: dict[str, bytes] = {}  # fixed lines (search fillers) pre-synthesized: they play instantly
        # One thread each so STT of the next utterance never waits behind TTS of the current reply.
        self.stt_executor = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self.tts_executor = ThreadPoolExecutor(1, thread_name_prefix="tts")
        self.mood_executor = ThreadPoolExecutor(1, thread_name_prefix="mood")
        self.sessions: dict[int, VoiceSession] = {}
        self._last_boomerang: dict[int, float] = {}  # guild id -> when it last came back after being kicked
        self.started_at = time.time()
        self.counters: Counter[str] = Counter()          # events for /stats (barge-ins, replies, ...)
        self.voice_turns: deque[dict] = deque(maxlen=200)  # latency breakdown of recent voice replies
        self.text_history: dict[int, list[dict]] = {}
        self._last_seen: dict[int, int] = {}  # channel id -> newest message id already in text_history
        self._last_active: dict[int, float] = {}  # channel id -> when we last talked there (fresh_after_min)
        self._threads: set[int] = set()       # threads we've talked in (Thread.me isn't always filled in)
        self.live: deque[dict] = deque(maxlen=400)  # dashboard Live view: heard/said lines, RAM only
        self._live_seq = 0
        self.muted: set[int] = self._load_muted()  # servers an admin muted from the dashboard
        names = [cfg.bot.name, *cfg.voice.wake_words] if cfg.discord.respond_to_name else []
        self._name_re = (re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in names if n), re.I)
                         if any(names) else None)
        self.vision = Vision(self)
        self.search = WebSearch(cfg, self.llm)
        self.addressee = AddresseeCheck(cfg, self.llm)
        self.profiles = ProfileManager(self)
        self.planner = Planner(self)
        self.presence = Presence(self)
        self.dashboard = Dashboard(self, config_path)
        self.restart_requested = False  # set by the dashboard; main.py exits non-zero so systemd restarts us
        self.tuning = TuningStore(cfg)
        self.mood = MoodReader(self)
        self.lore = Lore(self)
        self.weather = Weather(cfg)    # shared with the Fluxer bot
        self.quotes = QuoteBook(cfg.quotes.db_path)
        self.links = Links(cfg)        # shared with the Fluxer bot (the one shared file: who's who)
        self.links.profile_stores["discord"] = self.profiles.store
        self.profiles.linked = lambda uid: self.links.note("discord", uid)
        self._services_started = False
        self.discord_online = False       # logged in to Discord (false in platform.mode: fluxer)
        self.models_ready = asyncio.Event()  # STT/TTS/LLM warm: the Fluxer frontend waits on this
        self.closed = asyncio.Event()        # close() ran (the dashboard's restart, or shutdown)
        self.fluxer = None                   # the Fluxer frontend when platform.mode includes it
        _register_commands(self)

    # ------------------------------------------------------------------ startup

    async def setup_hook(self) -> None:
        await self.start_services()
        await self._sync_commands()

    async def start_services(self, discord_online: bool = True) -> None:
        """Dashboard, models and background jobs. Runs without a Discord login too (platform.mode: fluxer), where
        this object only hosts the shared models and the dashboard: Discord's own background jobs stay off."""
        if self._services_started:
            return
        self._services_started = True
        self.discord_online = discord_online
        self._remember_application_id()
        loop = asyncio.get_running_loop()
        try:
            await self.dashboard.start()  # first, so /status answers while the models load
        except OSError as e:
            log.error("Dashboard didn't start (port %s): %s", self.cfg.dashboard.port, e)
        log.info("Loading STT (%s) and TTS (%s)...", self.cfg.stt.backend, self.cfg.tts.backend)
        self.stt, self.tts = await asyncio.gather(
            loop.run_in_executor(self.stt_executor, create_stt, self.cfg),
            loop.run_in_executor(self.tts_executor, TTS, self.cfg),
        )
        await asyncio.gather(
            loop.run_in_executor(self.stt_executor, self.stt.warmup),
            loop.run_in_executor(self.tts_executor, self.tts.warmup),
            loop.run_in_executor(self.mood_executor, self.mood.load),
            loop.run_in_executor(self.mood_executor, self.lore.load),
            self.llm.warmup(),
        )
        for line in self.cfg.search.voice_fillers if self.search.enabled and self.cfg.search.voice else []:
            speech = clean_for_speech(line)
            self.tts_cache[speech] = await loop.run_in_executor(self.tts_executor, self.tts.synthesize_discord, speech)
        await self._load_creators()
        log.info("Models warm. Text LLM: %s | Voice LLM: %s", self.llm.describe(False), self.llm.describe(True))
        log.info("Web search: %s", " -> ".join(self.search.backends) if self.search.enabled else "off")
        self.models_ready.set()
        if discord_online:
            self.profiles.start()
            self.planner.start()
            self.lore.start()

    async def _sync_commands(self) -> None:
        if self.cfg.discord.guild_ids:
            for gid in self.cfg.discord.guild_ids:
                guild = discord.Object(id=gid)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            log.info("Slash commands synced to %d guild(s)", len(self.cfg.discord.guild_ids))
            # Commands live per server here, so any global ones are leftovers (e.g. from an older bot on this
            # application) that Discord would show next to ours as duplicates: clear them.
            stale = await self.tree.fetch_commands()
            if stale:
                self.tree.clear_commands(guild=None)
                await self.tree.sync()
                log.info("Removed %d leftover global command(s): %s", len(stale), ", ".join(c.name for c in stale))
        else:
            await self.tree.sync()
            log.info("Slash commands synced globally (can take a while to appear)")

    async def on_ready(self) -> None:
        log.info("Logged in as %s (%s)", self.user, self.user.id)
        self.presence.start()  # once; discord.py re-sends the last status itself after reconnects
        cid = self.cfg.voice.auto_join_channel_id
        if cid:
            channel = self.get_channel(cid)
            if isinstance(channel, discord.VoiceChannel | discord.StageChannel):
                try:
                    await self.join_channel(channel)
                except Exception:  # noqa: BLE001
                    log.exception("Auto-join failed")
            else:
                log.warning("voice.auto_join_channel_id %s is not a voice channel I can see", cid)

    # ------------------------------------------------------------------ live view / mute (dashboard)

    def feed(self, guild: discord.Guild | None, kind: str, who: str, text: str, where: str = "",
             platform: str = "discord") -> None:
        """One line for the dashboard's Live view. kind: heard, reply, cut, event, text_in, text_out.
        DMs stay out of it. The Fluxer frontend feeds its lines in here too (platform="fluxer")."""
        if guild is None:
            return
        self._live_seq += 1
        self.live.append({"seq": self._live_seq, "ts": time.time(), "guild": str(guild.id), "server": guild.name,
                          "where": where, "kind": kind, "who": who, "text": text, "platform": platform})

    @staticmethod
    def _load_muted() -> set[int]:
        try:
            return {int(g) for g in json.loads(MUTED_PATH.read_text())}
        except FileNotFoundError:
            return set()
        except (ValueError, TypeError) as e:
            log.warning("Ignoring unreadable %s: %s", MUTED_PATH, e)
            return set()

    def is_muted(self, guild_id: int | None) -> bool:
        return guild_id in self.muted

    def set_muted(self, guild_id: int, muted: bool) -> None:
        """Muted: keeps listening (and following the conversation) in that server but never replies, in voice
        or text. Kept across restarts."""
        (self.muted.add if muted else self.muted.discard)(guild_id)
        MUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
        MUTED_PATH.write_text(json.dumps(sorted(self.muted)))
        if muted and (session := self.sessions.get(guild_id)):
            session.interrupt()

    # ------------------------------------------------------------------ voice

    async def _who_disconnected(self, guild) -> str | None:
        """Who just used Discord's "Disconnect" on the bot, from the audit log (needs View Audit Log)."""
        await asyncio.sleep(1.5)  # the audit log entry lands a moment after the voice state change
        try:
            async for e in guild.audit_logs(limit=5, action=discord.AuditLogAction.member_disconnect):
                if (discord.utils.utcnow() - e.created_at).total_seconds() < 120 and e.user is not None:
                    return e.user.display_name
        except (discord.Forbidden, discord.HTTPException) as e:
            log.info("Couldn't read the audit log to see who kicked me (%s) - give me View Audit Log", e)
        return None

    async def boomerang(self, channel) -> None:
        """Someone force-disconnected the bot (Discord's Disconnect): come straight back, talk shit to whoever did
        it, then leave on its own terms (VoiceSession.leave_after_reply). Once per 10 minutes per server - a second
        kick means they really want it gone."""
        guild = channel.guild
        now = time.monotonic()
        if now - self._last_boomerang.get(guild.id, -1e9) < 600:
            log.info("🪃 kicked again from #%s - staying gone", channel.name)
            return
        self._last_boomerang[guild.id] = now
        kicker = await self._who_disconnected(guild)
        await asyncio.sleep(float(self.cfg.voice.get("boomerang_delay_s") or 4))
        if not any(not m.bot for m in channel.members):
            return  # everyone's gone: nobody to come back to
        session = await self.join_channel(channel, greet=False)
        session.leave_after_reply = True
        who = kicker or "someone"
        log.info("🪃 back in #%s to talk shit to %s for kicking me", channel.name, who)
        target = who if kicker else "whoever did it"
        session.post_event(
            f"{who} just force-disconnected you - right-click, Disconnect, like a coward - and you came straight back "
            f"to call them out before leaving on your own terms. Roast {target} for it, hard: two short, savage, funny "
            f"lines aimed right at them (the power trip, the cowardly right-click, anything you know about them or "
            "what they're doing right now). No hedging, no 'nah', no 'I guess', no being a good sport about it. Then "
            "one cocky line on your way out.", respond=True)

    async def join_channel(self, channel: discord.VoiceChannel, greet: bool = True) -> VoiceSession:
        guild = channel.guild
        session = self.sessions.get(guild.id)
        if session and session.vc.is_connected():
            if session.vc.channel.id != channel.id:
                await session.vc.move_to(channel)
            return session
        if guild.voice_client:
            await guild.voice_client.disconnect(force=True)
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        session = VoiceSession(self, vc)
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
        if session:
            await session.close()
            return True
        return False

    async def on_voice_state_update(self, member: discord.Member, before, after) -> None:
        session = self.sessions.get(member.guild.id)
        if not session:
            return
        if member.id == self.user.id:
            if after.channel is None:  # our own leave() drops the session first, so this is someone else's doing
                kicked_from = before.channel
                said_bye = time.monotonic() - session.bye_at < 120  # it was on its way out anyway
                await self.leave(member.guild.id)
                if said_bye:
                    log.info("Disconnected right after saying bye / being asked to leave - staying gone")
                elif kicked_from is not None and self.cfg.voice.get("boomerang"):
                    asyncio.create_task(self.boomerang(kicked_from))
            return
        channel = session.vc.channel
        if channel is None or member.bot:
            return
        was_here = before.channel is not None and before.channel.id == channel.id
        is_here = after.channel is not None and after.channel.id == channel.id
        if is_here and not was_here:
            session.post_event(f"{member.display_name} joined the voice channel", respond=self.cfg.voice.greet_on_join)
        elif was_here and not is_here:
            session.post_event(f"{member.display_name} left the voice channel", respond=False)
        if self.cfg.voice.auto_leave_when_empty:
            channel = session.vc.channel
            if channel and not any(not m.bot for m in channel.members):
                log.info("Voice channel empty, leaving")
                await self.leave(member.guild.id)

    async def post_clip(self, channel, audio, who: set[int], requester: str) -> str:
        """Encode and post a voice clip. Returns the note the bot reacts to it with."""
        seconds = audio.size / 48000
        guild = getattr(channel, "guild", None)
        names = []
        for uid in sorted(who):
            if uid == self.user.id:
                names.append(self.cfg.bot.name)
            elif member := (guild.get_member(uid) if guild else None) or self.get_user(uid):
                names.append(member.display_name)
        try:
            mp3 = await asyncio.to_thread(encode_mp3, audio, int(self.cfg.clips.bitrate))
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            await channel.send(
                f"🎬 **Clip** · {seconds:.0f}s · by {requester}" + (f" · feat. {', '.join(names)}" if names else ""),
                file=discord.File(io.BytesIO(mp3), filename=f"clip_{stamp}.mp3"),
                allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException as e:
            log.warning("Clip upload failed: %s", e)
            return f"[You tried to post a clip but Discord refused ({e.text or e}). Tell {requester}.]"
        self.counters["clips"] += 1
        log.info("🎬 clip %.0fs by %s (%s), %d KiB", seconds, requester, ", ".join(names), len(mp3) // 1024)
        who_txt = f" ({', '.join(names)} talking)" if names else ""
        return f"[You just clipped the last {seconds:.0f} seconds{who_txt} and posted it in the chat. React in a few words.]"

    def post_transcript(self, text: str) -> None:
        cid = self.cfg.voice.transcript_channel_id
        channel = self.get_channel(cid) if cid else None
        if channel:
            for chunk in split_message(text):
                asyncio.create_task(channel.send(chunk, allowed_mentions=discord.AllowedMentions.none()))

    def _remember_application_id(self) -> None:
        """The dashboard's "Login with Discord" needs the application id, which Discord only tells us at login:
        remember it, so the dashboard login keeps working when Discord is off (platform.mode: fluxer)."""
        path = Path("data/discord_app.json")
        try:
            if self.application_id:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"application_id": self.application_id}))
            elif path.exists():
                self._connection.application_id = int(json.loads(path.read_text())["application_id"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("Couldn't remember the Discord application id: %s", e)

    async def _load_creators(self) -> None:
        """Creator usernames for the persona. Remembered in data/creators.json, so the persona is the same when
        Discord is offline (platform.mode: fluxer) and the lookup can't run."""
        try:
            known = {int(k): v for k, v in json.loads(CREATORS_PATH.read_text()).items()}
        except (FileNotFoundError, ValueError, TypeError, AttributeError):
            known = {}
        names = []
        for uid in self.cfg.bot.creator_ids:
            try:
                if self.http.token is None:  # not logged in: use what we remembered
                    raise LookupError("Discord is offline")
                known[uid] = (self.get_user(uid) or await self.fetch_user(uid)).name
            except (discord.HTTPException, LookupError) as e:
                if uid not in known:
                    log.warning("bot.creator_ids: couldn't look up user %s: %s", uid, e)
            if uid in known:
                names.append(known[uid])
        try:
            CREATORS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CREATORS_PATH.write_text(json.dumps({str(k): v for k, v in known.items()}))
        except OSError as e:
            log.warning("Couldn't save %s: %s", CREATORS_PATH, e)
        self._creator_note = (self.cfg.bot.creator_prompt.replace("{creators}", " and ".join(names)).strip()
                              if names else "")
        self._creator_names = names
        if names:
            log.info("Creator: %s", ", ".join(names))

    def persona(self) -> str:
        """The system prompt's fixed head: persona + who made the bot. Identical every turn (prompt cache)."""
        persona = self.cfg.bot.system_prompt.replace("{name}", self.cfg.bot.name).strip()
        return f"{persona}\n{self._creator_note}" if self._creator_note else persona

    # ------------------------------------------------------------------ text chat

    def _system_prompt(self, author: discord.abc.User, channel) -> str:
        persona = self.persona()
        where = ("in a DM" if isinstance(author, discord.User)
                 else f"in the thread \"{channel.name}\"" if isinstance(channel, discord.Thread)
                 else f"in #{channel.name}")
        time_prompt = self.cfg.bot.time_prompt.strip()
        mine = self.profiles.self_note(getattr(channel, "guild", None))
        return (f"{persona}\n\n{time_prompt}\n\nYou're chatting {where}." + (f" {mine}" if mine else "")
                + f" The person talking to you now:\n{self.profiles.describe(author)}")

    def _trigger(self, message: discord.Message) -> str | None:
        """Why we should answer this message, or None to stay out of it."""
        d = self.cfg.discord
        channel = message.channel
        if message.guild is None:
            return "dm" if d.respond_in_dms else None
        if channel.id in d.text_channel_ids:
            return "channel"
        if d.respond_to_mentions and self.user in message.mentions:
            return "mention"
        ref = message.reference.resolved if message.reference else None
        if d.respond_to_replies and isinstance(ref, discord.Message) and ref.author.id == self.user.id:
            return "reply"
        if d.respond_in_threads and isinstance(channel, discord.Thread) and (
                channel.id in self._threads or channel.owner_id == self.user.id or channel.me is not None):
            return "thread"
        if self._name_re and self._name_re.search(message.content):
            return "name"
        return None

    async def close(self) -> None:
        if self.fluxer is not None:
            await self.fluxer.close()
        await self.dashboard.stop()
        await self.vision.close()
        await self.search.close()
        await self.weather.close()
        await self.planner.close()
        self.mood.maybe_save(force=True)
        await super().close()
        self.closed.set()

    def _fresh_s(self) -> float:
        return float(self.cfg.bot.get("fresh_after_min") or 0) * 60

    def _maybe_fresh(self, channel) -> None:
        """Someone's talking here after a long quiet: drop the old conversation so replies don't carry on
        from it (what the bot knows about people lives in profiles, not here)."""
        now, fresh = time.time(), self._fresh_s()
        quiet = now - self._last_active.get(channel.id, now)
        if fresh > 0 and quiet > fresh and self.text_history.get(channel.id):
            log.info("🆕 New conversation in #%s (quiet for %.0f min)", getattr(channel, "name", "DM"), quiet / 60)
            self.text_history.pop(channel.id, None)
            self._last_seen.pop(channel.id, None)
        self._last_active[channel.id] = now

    async def _backfill(self, message: discord.Message, history: list[dict]) -> list[tuple[dict, PostedImage]]:
        """Add what people said since we last looked at this channel, so a name-drop or reply comes with
        the conversation that led up to it. Our own messages are already in history.
        Returns the images posted in there, with the history entry holding each one's placeholder."""
        n = int(self.cfg.discord.context_messages)
        if n <= 0 or message.guild is None or message.channel.id in self.cfg.discord.text_channel_ids:
            return []
        last = self._last_seen.get(message.channel.id) or 0
        if (fresh := self._fresh_s()) > 0:  # an old conversation isn't context for a new one
            last = max(last, discord.utils.time_snowflake(discord.utils.utcnow() - timedelta(seconds=fresh)))
        try:
            older = [m async for m in message.channel.history(
                limit=n, before=message, after=discord.Object(id=last) if last else None, oldest_first=False)]
        except discord.HTTPException as e:
            log.debug("No channel history for context: %s", e)
            return []
        older = sorted((m for m in older if not m.author.bot), key=lambda m: m.id)
        fetched = await asyncio.gather(*(self.vision.fetch(m) for m in older))
        images = []
        for m, posted in zip(older, fetched):
            text = " ".join([m.clean_content.strip(), *[PLACEHOLDER] * len(posted)]).strip()
            if not text:
                continue
            entry = {"role": "user", "content": f"{m.author.display_name}: {text}"}
            history.append(entry)
            images += [(entry, img) for img in posted]
        return images

    async def _avatars_asked_about(self, message: discord.Message, content: str) -> list[PostedImage]:
        """Avatars to look at when the message talks about a pfp: ours, the people mentioned, or the author's."""
        if not (self.vision.enabled and _AVATAR_RE.search(content)):
            return []
        if _YOUR_AVATAR_RE.search(content):
            people = [self.user]
        else:
            people = [m for m in message.mentions if m.id != self.user.id] or [message.author]
        found = [await self.vision.fetch_avatar(p) for p in people[:3]]
        for img, p in zip(found, people):
            if img and p.id == self.user.id:
                img.label = "your own profile picture"
        return [img for img in found if img]

    def _as_member(self, user: discord.abc.User) -> discord.abc.User:
        """DMs give a plain User with no activities; look them up in a shared server instead."""
        if isinstance(user, discord.Member):
            return user
        for guild in user.mutual_guilds:
            member = guild.get_member(user.id)
            if member:
                return member
        return user

    @staticmethod
    async def _typing(channel) -> None:
        """'Static is typing...' until cancelled. A task, so the typing request never delays the reply."""
        try:
            async with channel.typing():
                await asyncio.Event().wait()
        except discord.HTTPException:
            pass

    def roast_note(self, requester, target, lines: list[str], mode: str) -> str:
        """Roast / rizz note for one person, with everything the bot knows about them as material."""
        row = self.profiles.store.get(target.id)
        usable = row is not None and not row["opted_out"]
        log.info("%s %s asked for a %s of %s", "🔥" if mode == "roast" else "💘", requester.display_name, mode,
                 target.display_name)
        return members.roast_note(requester.display_name, target, row["profile"] if usable else "", lines,
                                  self.vision.avatar_note(target, row) if usable else "", mode)

    async def _command_or_search(self, content: str, message: discord.Message, history: list[dict]) -> str | None:
        """A reminder/poll command's result note, else web search results (or None): extra text for this turn."""
        if self.planner.wants(content):
            voice = getattr(message.author, "voice", None)
            note = await self.planner.handle(content, message.author, message.channel,
                                             [m["content"] for m in history[-4:-1]],
                                             voice.channel.members if voice and voice.channel else [])
            if note:
                return note
        if found := calc.note(content):  # "what's 17 times 23" / "5 miles in km" / "time in Tokyo": exact
            log.info("🧮 %s", found)
            return found
        voice = getattr(message.author, "voice", None)
        people = [m.display_name for m in voice.channel.members if not m.bot] if voice and voice.channel else []
        if self.cfg.fun.enabled and (found := fun.note(content, people, message.author.display_name)):
            log.info("🎲 %s", found)  # coin / dice / pick someone / teams: really random
            return found
        if found := await self.weather.note(content):
            return found
        if mode := members.roast_request(content):  # "roast @jordan" / "talk shit about him" / "rizz him up"
            lines = [(ln.split(": ", 1)[0], ln.split(": ", 1)[1]) for m in history[-12:] if m["role"] == "user"
                     and isinstance(m["content"], str) for ln in m["content"].splitlines()
                     if ": " in ln and not ln.lstrip().startswith("[")]
            pool = [*message.mentions, *(message.guild.members if message.guild else [message.author])]
            if (target := members.roast_target(pool, content, message.author, [n for n, _ in lines])) is not None:
                return self.roast_note(message.author, target, [t for n, t in lines if n == target.display_name], mode)
        if (note := await members.lookup(message.guild, message.author, content,
                                         skip_names=[*self.cfg.voice.wake_words, self.cfg.bot.name])):
            return note  # about this server's people: never a web search
        # What they're playing, so "what's the best weapon?" searches that game.
        return await self.search.lookup(history, context=self.profiles.presence_note([self._as_member(message.author)]))

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        allowed = self.cfg.discord.allowed_user_ids
        if allowed and message.author.id not in allowed:
            return
        trigger = self._trigger(message)
        if trigger is None:
            return

        # clean_content turns <@id>, <#id>, <@&id> into @name / #channel / @role so the model knows who's meant.
        content = message.content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        content = discord.Message.clean_content.function(_with_content(message, content)).strip()
        if not content and not self.vision.images_in(message):
            return
        author = message.author
        if message.guild and self.is_muted(message.guild.id):
            log.info("(muted) 💬 [%s] %s: %s", trigger, author.display_name, content)
            self.feed(message.guild, "text_in", author.display_name, content, f"#{message.channel.name}")
            return
        self.profiles.activity()
        typing = asyncio.create_task(self._typing(message.channel))  # runs alongside the history fetch + LLM
        self._maybe_fresh(message.channel)
        history = self.text_history.setdefault(message.channel.id, [])
        try:
            # Images: this message's, the one it replies to (unless that's ours), then earlier ones.
            posted = await self.vision.fetch(message)
            ref = message.reference.resolved if message.reference else None
            ref_posted = (await self.vision.fetch(ref) if isinstance(ref, discord.Message)
                          and ref.author.id != self.user.id else [])
            parts = [content, *[PLACEHOLDER] * len(posted)]
            if ref_posted:
                parts.append(f"(replying to {ref.author.display_name}'s " + " ".join([PLACEHOLDER] * len(ref_posted)) + ")")
            said = " ".join(p for p in parts if p)
            line = f"{author.display_name}: {said}"
            log.info("💬 [%s] %s", trigger, line)
            self.feed(message.guild, "text_in", author.display_name, " ".join(p for p in parts if p),
                      f"#{message.channel.name}")

            earlier = await self._backfill(message, history)
            self._last_seen[message.channel.id] = message.id
            # What they're doing goes in only when it's not already in the conversation (see VoiceSession).
            note = self.profiles.presence_note([self._as_member(author)])
            content = f"{note}\n{line}" if note and not any(note in m["content"] for m in history) else line
            entry = {"role": "user", "content": content}
            history.append(entry)
            trim_history(history, int(self.cfg.bot.max_history_messages))
            to_caption = earlier + [(entry, img) for img in posted + ref_posted]

            messages = [{"role": "system", "content": self._system_prompt(author, message.channel)}] + history
            # A reminder/poll command, or else the search decision (GPU) - either runs while avatars download
            # and images get resized (network/CPU).
            lookup = asyncio.create_task(self._command_or_search(content, message, history))
            names = [author.display_name, self.cfg.bot.name, *self.cfg.voice.wake_words]
            memories = asyncio.create_task(self.lore.recall(message.guild.id if message.guild else None, content, names))
            mood = (asyncio.get_running_loop().run_in_executor(
                self.mood_executor, self.mood.read, author.id, author.display_name, content)
                if self.mood.model is not None and content else None)
            try:
                avatars = await self._avatars_asked_about(message, message.content)
                seen = await self.vision.prepare(
                    (avatars + posted + ref_posted + [img for _, img in reversed(earlier)])[: int(self.cfg.vision.max_images)])
                found = await lookup
                try:
                    lore_note = await memories
                except Exception as e:  # noqa: BLE001 - a memory is a nice-to-have
                    log.warning("Lore recall failed: %s", e)
                    lore_note = None
            finally:
                lookup.cancel()  # no-op when done; stops it if the image work raised
                memories.cancel()
            reads = [r for r in [await mood] if r] if mood else []
            mood_note = self.mood.choose(message.guild.id if message.guild else 0, reads)
            # This turn only, not kept in history. Before their line: the model answers what comes last.
            prompt = ((f"{found}\n\n" if found else "") + (f"{lore_note}\n" if lore_note else "")
                      + (f"{mood_note}\n" if mood_note else "") + f"{now_note()}\n{content}")
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
            log.exception("LLM error")
            if history and history[-1]["role"] == "user":
                history.pop()
            await message.reply(f"⚠️ LLM error: `{e}`", mention_author=False)
            return
        finally:
            typing.cancel()
            self.profiles.activity(cancel=False)

        self.profiles.observe(author.id, author.display_name, line, author)
        self.lore.observe(message.guild, author.id, line)
        speakers = {author.display_name, *self._creator_names}
        for m in history:  # "name: text" user turns (text or multimodal)
            c = m["content"] if isinstance(m["content"], str) else next(
                (p.get("text", "") for p in m["content"] if p.get("type") == "text"), "")
            if m["role"] == "user" and ":" in c[:40]:
                speakers.add(c.split(":", 1)[0].strip())
        guard = SpeakerGuard(self.cfg.bot.name, speakers)
        repeats = RepeatGuard(recent_replies(history) + prompt_examples(messages[0]["content"]))

        def clean(text: str) -> tuple[str, list[str]]:
            stripped = EchoGuard([content_said]).strip(text, guard.strip_label)
            if stripped != text.strip():
                log.info("(dropped the model's copy of %s's message)", author.display_name)
            return repeats.filter(guard.clean(stripped))

        content_said = said
        raw_reply = reply
        reply, dropped = clean(reply)
        if not reply and raw_reply.strip():  # it only echoed them (or repeated itself): one more try
            try:
                raw_reply = await self.llm.complete(messages)
                reply, dropped = clean(raw_reply)
            except Exception as e:  # noqa: BLE001
                log.warning("Retry after an empty reply failed: %s", e)
        for s in dropped:
            log.info("(repeat skipped: %s)", s)
        if not reply:
            log.warning("No text reply to %s: the filters left nothing of %.300r", author.display_name, raw_reply)
            return
        log.info("🤖 [text] %s", reply)
        self.feed(message.guild, "text_out", self.cfg.bot.name, reply, f"#{message.channel.name}")
        self._last_active[message.channel.id] = time.time()
        self.counters["text replies"] += 1
        if isinstance(message.channel, discord.Thread):
            self._threads.add(message.channel.id)
        history.append({"role": "assistant", "content": reply})
        self.mood.replied(message.guild.id if message.guild else 0, reads)
        self.profiles.observe_reply([author.id], f"{self.cfg.bot.name}: {reply}")
        self.lore.observe(message.guild, 0, f"{self.cfg.bot.name}: {reply}")
        self.vision.caption_later(to_caption)  # so later turns still know what the images were
        for i, chunk in enumerate(split_message(reply)):
            if i == 0:
                await message.reply(chunk, mention_author=False)
            else:
                await message.channel.send(chunk)


def _register_commands(bot: VoiceBot) -> None:
    tree = bot.tree

    async def _owner_only(interaction: discord.Interaction) -> bool:
        """Settings that apply to every server (the model) are for the bot's owners: creators + dashboard admins."""
        owners = {int(i) for i in [*(bot.cfg.bot.get("creator_ids") or []), *(bot.cfg.dashboard.admin_ids or [])]}
        if interaction.user.id in owners:
            return True
        await interaction.response.send_message("Only the bot's owner can do that.", ephemeral=True)
        return False

    @tree.command(name="join", description="Join your voice channel and start listening")
    async def join(interaction: discord.Interaction):
        member = interaction.user
        if not isinstance(member, discord.Member) or not member.voice or not member.voice.channel:
            await interaction.response.send_message("Join a voice channel first.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            await bot.join_channel(member.voice.channel)
        except Exception as e:  # noqa: BLE001
            log.exception("Join failed")
            await interaction.followup.send(f"Couldn't join: `{e}`", ephemeral=True)
            return
        mode = bot.cfg.voice.response_mode
        text = f"🎧 Listening in {member.voice.channel.mention} (mode: {mode})"
        if missing := _missing_perms(interaction.guild, member.voice.channel):
            invite = discord.utils.oauth_url(bot.application_id, permissions=INVITE_PERMS,
                                             scopes=("bot", "applications.commands"))
            text += (f"\n⚠️ I'm missing some permissions here: {', '.join(missing)}. "
                     f"A server admin can fix that by re-adding me with this link (it keeps everything else): "
                     f"<{invite}>")
        await interaction.followup.send(text, ephemeral=True)

    @tree.command(name="leave", description="Leave the voice channel")
    async def leave(interaction: discord.Interaction):
        left = await bot.leave(interaction.guild_id)
        await interaction.response.send_message("👋 Left." if left else "I'm not in voice.", ephemeral=True)

    async def _admins_only(interaction: discord.Interaction) -> bool:
        """Server managers (Manage Server / Administrator) and the bot's owners. default_permissions below hides
        these commands from everyone else, but servers can override that, so it's checked again here."""
        owners = {int(i) for i in [*(bot.cfg.bot.get("creator_ids") or []), *(bot.cfg.dashboard.admin_ids or [])]}
        if _is_manager(interaction) or interaction.user.id in owners:
            return True
        await interaction.response.send_message("Only server admins can use that.", ephemeral=True)
        return False

    @tree.command(name="reset", description="Clear the bot's conversation memory here (admins)")
    @app_commands.default_permissions(manage_guild=True)
    async def reset(interaction: discord.Interaction):
        if interaction.guild is not None and not await _admins_only(interaction):
            return  # in a DM it's your own conversation: always allowed
        bot.text_history.pop(interaction.channel_id, None)
        session = bot.sessions.get(interaction.guild_id)
        if session:
            session.reset()
        await interaction.response.send_message("🧹 Memory cleared.", ephemeral=True)

    @tree.command(name="stop", description="Stop talking")
    async def stop(interaction: discord.Interaction):
        session = bot.sessions.get(interaction.guild_id)
        if session:
            session.interrupt()
        await interaction.response.send_message("🤫", ephemeral=True)

    @tree.command(name="say", description="Make the bot say something in the voice channel (admins)")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.guild_only()
    async def say(interaction: discord.Interaction, text: str):
        if not await _admins_only(interaction):
            return
        session = bot.sessions.get(interaction.guild_id)
        if not session:
            await interaction.response.send_message("I'm not in voice. Use /join.", ephemeral=True)
            return
        voice = getattr(interaction.user, "voice", None)
        if not voice or voice.channel != session.vc.channel:
            await interaction.response.send_message("Join my voice channel to use this.", ephemeral=True)
            return
        await session.say(text)
        await interaction.response.send_message("🔊", ephemeral=True)

    async def endpoint_autocomplete(interaction: discord.Interaction, current: str):
        return [app_commands.Choice(name=n, value=n) for n in bot.llm.endpoints if current.lower() in n.lower()][:25]

    @tree.command(name="llm", description="Switch the LLM endpoint (and optionally model)")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(endpoint="Endpoint name from config.yaml", model="Override model name",
                           scope="Apply to text, voice, or both")
    @app_commands.choices(scope=[app_commands.Choice(name=s, value=s) for s in ("both", "voice", "text")])
    @app_commands.autocomplete(endpoint=endpoint_autocomplete)
    async def llm(interaction: discord.Interaction, endpoint: str, model: str | None = None,
                  scope: str = "both"):
        if not await _owner_only(interaction):
            return
        if endpoint not in bot.llm.endpoints:
            await interaction.response.send_message(f"Unknown endpoint. Options: {', '.join(bot.llm.endpoints)}",
                                                    ephemeral=True)
            return
        if scope in ("both", "text"):
            bot.llm.text_endpoint = endpoint
        if scope in ("both", "voice"):
            bot.llm.voice_endpoint = endpoint
        if model:
            bot.llm.model_overrides[endpoint] = model
        await interaction.response.defer(ephemeral=True, thinking=True)
        await bot.llm.warmup(attempts=1)
        await interaction.followup.send(
            f"Text: `{bot.llm.describe(False)}`\nVoice: `{bot.llm.describe(True)}`", ephemeral=True)

    @tree.command(name="models", description="List models available on an endpoint")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.autocomplete(endpoint=endpoint_autocomplete)
    async def models(interaction: discord.Interaction, endpoint: str | None = None):
        if not await _owner_only(interaction):
            return
        name = endpoint or bot.llm.text_endpoint
        if name not in bot.llm.endpoints:
            await interaction.response.send_message("Unknown endpoint.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            ids = await bot.llm.list_models(name)
            body = "\n".join(ids) or "(none)"
            await interaction.followup.send(f"**{name}**\n```\n{body[:1800]}\n```", ephemeral=True)
        except Exception as e:  # noqa: BLE001
            await interaction.followup.send(f"Failed: `{e}`", ephemeral=True)

    @tree.command(name="status", description="All the nerdy numbers: LLM tokens & speed, GPU, latency, pipeline")
    @app_commands.describe(private="Only you can see it")
    async def status(interaction: discord.Interaction, private: bool = False):
        await interaction.response.defer(ephemeral=private, thinking=True)
        view = StatsView(bot)
        view.message = await interaction.followup.send(embeds=await build_stats(bot, interaction.guild), view=view,
                                                       ephemeral=private, wait=True)

    @tree.command(name="help", description="How to talk to the bot, what it can do, and its commands")
    async def help_cmd(interaction: discord.Interaction):
        await interaction.response.send_message(embed=_help_embed(bot), ephemeral=True)

    @tree.command(name="clip", description="Post the last few seconds of the voice channel as an MP3")
    @app_commands.describe(seconds="How far back (default from config, max = the rolling buffer)")
    async def clip(interaction: discord.Interaction, seconds: app_commands.Range[int, 3, 600] | None = None):
        session = bot.sessions.get(interaction.guild_id)
        if not session or session.clips is None:
            await interaction.response.send_message("I'm not recording a voice channel here (or clips are off).",
                                                    ephemeral=True)
            return
        c = bot.cfg.clips
        seconds = min(float(seconds or c.default_s), float(c.buffer_s))
        await interaction.response.defer(ephemeral=True, thinking=True)
        audio, who = session.clips.clip(time.monotonic(), seconds)
        if audio.size < 48000:
            await interaction.followup.send("Nothing to clip - it's been quiet.", ephemeral=True)
            return
        note = await bot.post_clip(interaction.channel, audio, who, interaction.user.display_name)
        await interaction.followup.send("🎬 Clipped." if note.startswith("[You just") else "Couldn't post it.",
                                        ephemeral=True)

    @tree.command(name="tuning", description="How the bot has tuned its voice replies from reactions (admins)")
    @app_commands.describe(reset="Forget what it learned")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.guild_only()
    async def tuning(interaction: discord.Interaction, reset: bool = False):
        if not await _admins_only(interaction):
            return
        t = bot.tuning.for_guild(interaction.guild_id or 0)
        if reset:
            if not _is_manager(interaction):
                await interaction.response.send_message("Only server managers can reset it.", ephemeral=True)
                return
            t.reset()
        lines = [f"**Reply length:** {LEVEL_NAMES[t.level]} (level {t.level}/3)"
                 + (" · next reply may run longer" if t.more_next else ""),
                 f"**Follow-up window:** {t.followup_s:.0f}s (configured {bot.tuning.base_followup:.0f}s)",
                 f"**Uninterrupted streak:** {t.streak}"]
        if not bot.tuning.enabled:
            lines.insert(0, "_Self-tuning is off in config._")
        if t.log:
            lines.append("**Recent adjustments:**")
            lines += [f"<t:{int(ts)}:R> {why}" for ts, why in reversed(t.log)]
        if bot.mood.enabled:
            learned = bot.mood.summary(interaction.guild_id or 0)
            lines.append("**Mood strategies** (★ = working best, wins/tries):")
            lines += [f"`{line}`" for line in learned] or ["_nothing learned yet_"]
            th = bot.mood.state["users"].get(str(interaction.user.id), {}).get("th", {})
            if any(v != 0.5 for v in th.values()):
                lines.append("**Your calibration** (higher = needs more to read you that way): "
                             + ", ".join(f"{g} {v:.2f}" for g, v in th.items() if v != 0.5))
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    # ------------------------------------------------------------------ reminders / polls

    async def _planner_off(interaction: discord.Interaction) -> bool:
        if not bot.planner.enabled:
            await interaction.response.send_message("Reminders and polls are turned off in config.", ephemeral=True)
        return not bot.planner.enabled

    @tree.command(name="remind", description="Set a reminder: a DM for you, or a ping for a group")
    @app_commands.describe(when="e.g. in 20 minutes, at 5pm, tomorrow at 9am, friday 6pm", what="What to remind about",
                           who="Just you, or everyone in your voice channel right now",
                           person="Remind someone else instead", deliver="Default: DM for just you, ping here for others")
    @app_commands.choices(
        who=[app_commands.Choice(name="me", value="me"),
             app_commands.Choice(name="everyone in my voice channel", value="voice")],
        deliver=[app_commands.Choice(name="DM", value="dm"),
                 app_commands.Choice(name="ping in this channel", value="channel")])
    async def remind(interaction: discord.Interaction, when: str, what: str, who: str = "me",
                     person: discord.Member | None = None, deliver: str | None = None):
        if await _planner_off(interaction):
            return
        user = interaction.user
        due = parse_when(when)
        if due is None:
            await interaction.response.send_message(
                f"Couldn't work out when \"{when}\" is. Try `in 20 minutes`, `at 5pm` or `tomorrow at 9am`.",
                ephemeral=True)
            return
        if err := bot.planner.check_quota(user.id):
            await interaction.response.send_message(f"Can't: {err}.", ephemeral=True)
            return
        if person is not None:
            targets = [person.id]
        elif who == "voice":
            voice = getattr(user, "voice", None)
            if not voice or not voice.channel:
                await interaction.response.send_message("You're not in a voice channel.", ephemeral=True)
                return
            targets = [m.id for m in voice.channel.members if not m.bot]
        else:
            targets = [user.id]
        just_me = targets == [user.id]
        deliver = deliver or ("dm" if just_me else "channel")
        job_id = bot.planner.add_reminder(user, targets, what, due.timestamp(), deliver, interaction.channel)
        ts = int(due.timestamp())
        whom = "you" if just_me else " ".join(f"<@{t}>" for t in targets)
        await interaction.response.send_message(
            f"⏰ Reminder `#{job_id}` for {whom} <t:{ts}:R> (<t:{ts}:f>), {'by DM' if deliver == 'dm' else 'here'}: {what}",
            ephemeral=just_me, allowed_mentions=discord.AllowedMentions.none())

    async def reminder_autocomplete(interaction: discord.Interaction, current: str):
        uid = interaction.user.id
        rows = [r for r in bot.planner.store.reminders_for(uid) if r["creator_id"] == uid]
        return [app_commands.Choice(name=f"#{r['id']} {spoken_time(r['due'])}: {r['text']}"[:100], value=r["id"])
                for r in rows if current.lower() in f"{r['id']} {r['text'].lower()}"][:25]

    @tree.command(name="reminders", description="List your pending reminders, or cancel one")
    @app_commands.describe(cancel="A reminder to cancel")
    @app_commands.autocomplete(cancel=reminder_autocomplete)
    async def reminders(interaction: discord.Interaction, cancel: int | None = None):
        if await _planner_off(interaction):
            return
        uid = interaction.user.id
        if cancel is not None:
            row = bot.planner.store.get(cancel)
            if row is None or row["kind"] != "reminder" or (row["creator_id"] != uid and not _is_manager(interaction)):
                await interaction.response.send_message("That's not one of your reminders.", ephemeral=True)
                return
            bot.planner.cancel(cancel)
            await interaction.response.send_message(f"🗑️ Cancelled: {row['text']}", ephemeral=True)
            return
        rows = bot.planner.store.reminders_for(uid)
        lines = [f"`#{r['id']}` <t:{int(r['due'])}:R> - {r['text']}"
                 + ("" if r["creator_id"] == uid else f" _(from {r['creator_name']})_") for r in rows[:20]]
        await interaction.response.send_message("\n".join(lines) or "No reminders pending.", ephemeral=True)

    @tree.command(name="poll", description="Start a poll - the bot announces the result in voice when it closes")
    @app_commands.describe(question="What to ask", options="2-10 choices, separated by commas",
                           duration="How long it runs, e.g. 5 minutes or 2 hours", multiple="Allow picking more than one")
    async def poll(interaction: discord.Interaction, question: str, options: str, duration: str | None = None,
                   multiple: bool = False):
        if await _planner_off(interaction):
            return
        choices = list(dict.fromkeys(o.strip() for o in re.split(r"[|,\n]", options) if o.strip()))
        if not 2 <= len(choices) <= 10:
            await interaction.response.send_message("A poll needs 2-10 choices, separated by commas.", ephemeral=True)
            return
        seconds = parse_duration(duration or "", float(bot.cfg.reminders.poll_minutes) * 60)
        if seconds is None:
            await interaction.response.send_message(f"Couldn't read the duration \"{duration}\". Try `5 minutes`.",
                                                    ephemeral=True)
            return
        if err := bot.planner.check_quota(interaction.user.id):
            await interaction.response.send_message(f"Can't: {err}.", ephemeral=True)
            return
        p, ends = bot.planner.build_poll(question, choices, seconds, multiple)
        await interaction.response.send_message(f"📊 Poll by {interaction.user.mention} · closes <t:{int(ends)}:R>",
                                                poll=p, allowed_mentions=discord.AllowedMentions.none())
        bot.planner.track_poll(await interaction.original_response(), interaction.user, question, ends)

    # ------------------------------------------------------------------ profiles / privacy

    def _is_manager(interaction: discord.Interaction) -> bool:
        perms = getattr(interaction.user, "guild_permissions", None)
        return bool(perms and (perms.manage_guild or perms.administrator))

    async def _target(interaction: discord.Interaction, user: discord.Member | None):
        """Resolve whose profile a command is about; only server managers may act on other people."""
        if user is None or user.id == interaction.user.id:
            return interaction.user
        if not _is_manager(interaction):
            await interaction.response.send_message("You can only do that for yourself.", ephemeral=True)
            return None
        return user

    @tree.command(name="profile", description="See what the bot remembers about you (managers: about anyone)")
    async def profile(interaction: discord.Interaction, user: discord.Member | None = None):
        target = await _target(interaction, user)
        if target is None:
            return
        text = bot.profiles.summary(target.id)
        if not bot.profiles.enabled:
            text = "_Profiles are disabled in config._\n" + text
        await interaction.response.send_message(text[:1990], ephemeral=True)

    @tree.command(name="forget", description="Delete everything the bot remembers about you (managers: about anyone)")
    async def forget(interaction: discord.Interaction, user: discord.Member | None = None):
        target = await _target(interaction, user)
        if target is None:
            return
        bot.profiles.store.forget(target.id)
        bot.mood.forget(target.id)
        bot.lore.forget(target.id)
        bot.links.unlink("discord", target.id)
        await interaction.response.send_message(f"🗑️ Forgot everything about {target.display_name}.", ephemeral=True)

    @tree.command(name="profiling", description="Turn off (or back on) the bot building a profile of you")
    async def profiling(interaction: discord.Interaction, enabled: bool):
        bot.profiles.store.set_opted_out(interaction.user.id, not enabled)
        if not enabled:
            bot.lore.forget(interaction.user.id)
        msg = ("✅ I'll remember things about you again." if enabled
               else "🙈 Got it - I won't build a profile of you, and I deleted what I had.")
        await interaction.response.send_message(msg, ephemeral=True)

    # ------------------------------------------------------------------ quote book / account links

    @tree.command(name="quote", description="A random quote from the quote book, or search it (\"Static, quote that\" in voice saves one)")
    @app_commands.describe(search="Words or a name to look for", delete="Delete a quote by its number (admins, or whoever saved it)")
    @app_commands.guild_only()
    async def quote(interaction: discord.Interaction, search: str | None = None, delete: int | None = None):
        if not bot.cfg.quotes.enabled:
            await interaction.response.send_message("The quote book is turned off in config.", ephemeral=True)
            return
        gid = interaction.guild_id or 0
        if delete is not None:
            row = bot.quotes.get(delete)
            if row is None or row["guild_id"] != gid:
                await interaction.response.send_message("No quote with that number here.", ephemeral=True)
            elif not (_is_manager(interaction) or row["saved_by"] == interaction.user.display_name):
                await interaction.response.send_message("Only admins or whoever saved it can delete it.", ephemeral=True)
            else:
                bot.quotes.delete(delete)
                await interaction.response.send_message(f"🗑️ Deleted quote #{delete}.", ephemeral=True)
            return
        if search:
            rows = bot.quotes.search(gid, search)
            text = "\n".join(quotes.show(r) for r in rows) or f"No quotes matching \"{search}\"."
        else:
            row = bot.quotes.random(gid)
            text = quotes.show(row) if row else "The quote book is empty. Say \"Static, quote that\" in voice to start it."
        await interaction.response.send_message(text[:1990], allowed_mentions=discord.AllowedMentions.none())

    @tree.context_menu(name="Save as quote")
    @app_commands.guild_only()
    async def save_quote(interaction: discord.Interaction, message: discord.Message):
        if not bot.cfg.quotes.enabled or not message.content.strip():
            await interaction.response.send_message("Nothing to quote there (or the quote book is off).", ephemeral=True)
            return
        gid = interaction.guild_id or 0
        if (qid := bot.quotes.exists(gid, message.clean_content)) is not None:
            await interaction.response.send_message(f"Already in the book as #{qid}.", ephemeral=True)
            return
        qid = bot.quotes.add(gid, message.author.id, message.author.display_name, message.clean_content,
                             message.created_at.timestamp(), interaction.user.display_name, "text")
        log.info("💬 quote #%d saved by %s: %s", qid, interaction.user.display_name, message.clean_content[:80])
        await interaction.response.send_message(quotes.show(bot.quotes.get(qid)) + " · saved",
                                                allowed_mentions=discord.AllowedMentions.none())

    @tree.command(name="link", description="Link your Discord account to your Fluxer account (so Static knows it's you on both)")
    @app_commands.describe(code="The code you got from !link on Fluxer (leave empty to get a code for Fluxer instead)")
    async def link(interaction: discord.Interaction, code: str | None = None):
        if not bot.links.enabled:
            await interaction.response.send_message("Account linking is turned off.", ephemeral=True)
            return
        u = interaction.user
        if code:
            ok, msg = bot.links.complete("discord", u.id, u.name, code)
            if ok:
                log.info("🔗 %s linked their Discord and Fluxer accounts", u.name)
            await interaction.response.send_message(msg.replace("with unlink", "with /unlink"), ephemeral=True)
            return
        code = bot.links.start("discord", u.id, u.name)
        await interaction.response.send_message(
            f"🔗 Your code: **{code}** - on Fluxer, type `{bot.cfg.fluxer.prefix}link {code}` within 10 minutes.\n"
            "Once linked, Static knows both accounts are you and shares what it remembers about you between them. "
            "Nothing else is shared; `/unlink` undoes it.", ephemeral=True)

    @tree.command(name="unlink", description="Unlink your Discord and Fluxer accounts")
    async def unlink(interaction: discord.Interaction):
        done = bot.links.unlink("discord", interaction.user.id)
        await interaction.response.send_message("🔗 Unlinked." if done else "You're not linked.", ephemeral=True)


def _missing_perms(guild, channel) -> list[str]:
    """Permissions from the invite link the bot doesn't have (server-wide ones like View Audit Log, plus this
    voice channel's), as Discord names them."""
    have_guild = guild.me.guild_permissions
    have_here = channel.permissions_for(guild.me)
    server_wide = {"view_audit_log"}
    return [name.replace("_", " ").title().replace("Vc", "VC") for name, wanted in INVITE_PERMS
            if wanted and not getattr(have_guild if name in server_wide else have_here, name)]


class _with_content:
    """A message stand-in with different text, for discord.Message.clean_content (which reads .content,
    .mentions, .role_mentions, .channel_mentions and .guild)."""

    def __init__(self, message: discord.Message, content: str):
        self._m, self.content = message, content

    def __getattr__(self, name):
        return getattr(self._m, name)


def _help_embed(bot: VoiceBot) -> discord.Embed:
    """/help: the short version of the dashboard's /help page (built from the same live details)."""
    d = helpinfo.info(bot)
    name, v, t, f = d["name"], d["voice"], d["text"], d["features"]
    wake = (v["wake_words"][0] if v["wake_words"] else name).capitalize()
    e = discord.Embed(title=f"📡 How to talk to {name}", url=d["url"], color=0xA77CF5,
                      description=f"{name} hangs out in voice and chat. Talk to it like anyone else in the call. "
                                  "It all runs on its owner's PC: no cloud AI.")
    if d["avatar"]:
        e.set_thumbnail(url=d["avatar"])
    if v["mode"] == "wake_word":
        follow = (f"After it answers, keep talking for ~{v['followup_s']:.0f}s without the name"
                  + (" (just you: others say its name to join in)." if v["followup_scope"] == "speaker" else "."))
        voice = f"`/join` from your voice channel, then say **\"{wake}, …\"**. {follow}"
    else:
        voice = "`/join` from your voice channel and just talk: it answers everything."
    voice += "\nTalk over it for a second to cut it off (a laugh won't), or `/stop`."
    if v["resume_s"]:
        voice += " Cut it off by accident? Say \"go on\"."
    if v["leave_on_request"]:
        voice += f" \"{wake}, you can leave now\" or `/leave` sends it away."
    e.add_field(name="🎙 Voice", value=voice, inline=False)
    ways = [w for w, on in (("@mention it", t["mentions"]), ("reply to it", t["replies"]),
                            (f"say \"{wake}\"", t["name"]), ("DM it", t["dms"])) if on]
    text = ", ".join(ways[:-1]) + f" or {ways[-1]}." if len(ways) > 1 else (ways[0] + "." if ways else "")
    if f["vision"]:
        text += " It can see images you post."
    e.add_field(name="💬 Text", value=text[:1024] or "-", inline=False)
    extras = []
    if f["search"]:
        extras.append("**Look it up:** ask about anything current and it searches the web first.")
    extras.append(f"**Server info:** \"{wake}, what roles does <someone> have?\" or \"list my roles\" - even for people not in the call.")
    if f["reminders"]:
        extras.append(f"**Reminders & polls:** \"{wake}, remind me in 20 minutes to…\", \"make a poll: pizza or tacos\".")
    if f["clips"]:
        extras.append(f"**Clips:** \"clip that!\" posts the last {f['clip_default_s']:.0f}s of the call as an MP3.")
    if f["quotes"]:
        extras.append(f"**Quote book:** \"{wake}, quote that\" saves the last line; `/quote` pulls one up "
                      "(or right-click a message > Apps > Save as quote).")
    if f["fun"]:
        extras.append("**Game night:** \"flip a coin\", \"roll 2d6\", \"pick someone\", \"split us into two teams\".")
    if f["weather"]:
        extras.append(f"**Weather:** \"{wake}, what's the weather in Denver tomorrow?\"")
    if extras:
        e.add_field(name="✨ Ask it to", value="\n".join(extras)[:1024], inline=False)
    cmds = [c for c in d["commands"] if c["who"] == "everyone"]
    listing = " ".join(f"`/{c['name']}`" for c in cmds)
    e.add_field(name="⌨️ Commands", value=listing[:1024] or "-", inline=False)
    if f["profiles"]:
        e.add_field(name="🔒 Your data", value="It keeps short notes on people from conversations. `/profile` shows yours, "
                    "`/forget` deletes it, `/profiling` turns it off. Calls aren't recorded."
                    + (" Also on our Fluxer server? `/link` connects the two accounts (optional)." if f["links"] else ""),
                    inline=False)
    if d["url"]:
        e.set_footer(text=f"Full guide: {d['url']}")
    return e
