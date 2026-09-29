"""One live voice conversation in one guild: per-user VAD -> STT -> streaming LLM -> streaming TTS.

Threading model:
  * VoiceSink.write() runs on voice-recv's decoder thread, once per 20ms packet per speaker. It only
    does cheap work (downsample + VAD) and hands finished utterances to the event loop.
  * A watchdog task on the event loop ends utterances when a speaker goes quiet (Discord stops
    sending packets when someone stops talking, so we can't rely on silence frames arriving).
  * A worker task processes utterances in order. Replies run as their own task so barge-in can
    cancel them. The LLM producer and TTS consumer run concurrently, so sentence 1 is playing while
    sentence 2 is being synthesized and later sentences are still being generated.

Turn-taking, to feel like a real conversation:
  * Speculative STT during a pause; the partial transcript decides whether the pause is the end of
    the turn (end sooner) or a mid-sentence breath (wait longer).
  * When someone starts talking over the bot it ducks its volume immediately; if they keep going
    it stops (barge-in), if it was just "yeah"/"mhm" it carries on and the backchannel is ignored.
  * Join/leave events go into the conversation so the bot can greet people.
  * Speculative reply: once the speculative transcript is in and only that one person is talking,
    the whole reply pipeline (search check, LLM, first TTS chunk) starts during the pause, but its
    audio is held until the turn is confirmed over - exactly when it would have played anyway, so it
    never cuts anyone off. If they keep talking, or anyone else starts, it's thrown away.
  * Never talking over people: before the first sentence plays, the bot waits until nobody is
    mid-utterance; if someone said something new meanwhile, it answers everyone together instead.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import aclosing
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import discord
from discord.ext import voice_recv

from .audio import DISCORD_SR, FRAME_MS, STT_FRAME_BYTES, STT_SR, StreamingPCMSource, discord_to_16k_mono, make_vad
from . import calc, fun, members, quotes
from .profiles import member_presence
from .scene import Scene
from .clips import ClipRecorder, clip_request
from .text_utils import (ActionFilter, EchoGuard, RepeatGuard, continue_request, SentenceChunker, SpeakerGuard, ThinkFilter, clean_for_speech,
                         is_backchannel, leave_request, looks_finished, name_hints, now_note, prompt_examples, quiet_request,
                         says_goodbye,
                         recent_replies, split_sentences, strip_stage,
                         trim_history)

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.voice")


@dataclass
class Utterance:
    user_id: int
    name: str
    pcm: bytes                        # 16kHz mono s16le
    ended_at: float                   # time.monotonic() of last voiced frame
    stt_future: Future | None = None  # speculative transcript already in progress/done
    overlapped: bool = False          # started while the bot was talking/thinking
    text: str | None = None           # events: pre-made text, no STT
    event: bool = False
    respond: bool = True              # events: should the bot react to it?
    started_at: float = 0.0           # time.monotonic() when they started talking


class _Superseded(Exception):
    """Someone said something new while the reply waited for them to finish: answer it all together."""


@dataclass
class _Speculation:
    """A reply started during someone's pause, held back until their turn is confirmed over."""
    user_id: int
    future: Future          # the speculative transcript it was built from
    content: str            # the user turn it assumed (must match the real one to be used)
    gate: asyncio.Event     # set = the turn is over, play it
    task: asyncio.Task
    metrics: dict


class _SpeakerState:
    def __init__(self, name: str, preroll_frames: int):
        self.name = name
        self.lock = threading.Lock()
        self.preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self.frames: list[bytes] = []
        self.speaking = False
        self.consecutive_voiced = 0
        self.voiced_frames = 0
        self.last_voiced_idx = 0
        self.last_voice = 0.0
        self.started = 0.0
        self.barged = False
        self.overlapped = False
        self.spec_future: Future | None = None
        self.spec_idx = -1
        self.spec_replied: Future | None = None  # the transcript a speculative reply was tried for

    def reset(self) -> None:
        self.frames = []
        self.speaking = False
        self.consecutive_voiced = 0
        self.voiced_frames = 0
        self.last_voiced_idx = 0
        self.barged = False
        self.overlapped = False
        self.spec_future = None
        self.spec_idx = -1
        self.spec_replied = None


class VoiceSink(voice_recv.AudioSink):
    def __init__(self, session: "VoiceSession"):
        super().__init__()
        self.session = session

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData) -> None:
        if user is not None and data.pcm:
            self.session.on_audio(user, data.pcm)

    def cleanup(self) -> None:
        pass


class VoiceSession:
    def __init__(self, bot: "VoiceBot", vc: voice_recv.VoiceRecvClient):
        self.bot = bot
        self.vc = vc
        self.cfg = bot.cfg
        self.v = bot.cfg.voice
        self.loop = asyncio.get_running_loop()
        self.vad = make_vad(self.v)

        self.history: list[dict] = []
        self._last_turn = time.monotonic()  # for bot.fresh_after_min
        self._states: dict[int, _SpeakerState] = {}
        self._queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._reply_task: asyncio.Task | None = None
        self._spec: _Speculation | None = None
        self._processing = False  # the worker is handling a batch
        self._owe_reply = False   # a reply was dropped to wait for others: answer on the next batch
        self._barge_pending = False  # the current reply is being cancelled because someone talked over it
        self.clips = ClipRecorder(float(self.cfg.clips.buffer_s)) if self.cfg.clips.enabled else None
        self.tuner = bot.tuning.for_guild(vc.guild.id)
        self._source: StreamingPCMSource | None = None
        self._busy = False  # read from the sink thread; plain bool assignment is atomic
        self._last_reply_at = 0.0
        self._last_presence = ""
        self._stt_hint = ""  # names in the conversation, so Whisper spells them right
        self.leave_after_reply = False    # came back to talk shit after being kicked (bot.boomerang): leave after
        self._cut: tuple[float, str, str] | None = None
        self._reply_to: frozenset[int] = frozenset()
        self._upset: dict[int, deque] = {}  # user id -> recent strong upset readings (when, score, louder)
        self._last_check_in = 0.0
        self._checked_on: dict[int, float] = {}  # who the current reply is for: only they can stop it (read in the sink thread)  # last reply cut off by barge-in: (when, said, unsaid)
        self._last_clip_at = 0.0  # when a clip was last posted (a second "clip that" right after is the same moment)
        self._partners: dict[int, float] = {}  # user id -> when we last answered them (per-person follow-ups)
        self._search_memory: deque[tuple[float, str]] = deque(maxlen=2)  # recent search results, for follow-ups
        self._said: deque[tuple[int, str, str, float]] = deque(maxlen=40)  # (user id, name, text, when) for "quote that"
        self._barge = (0.0, 0)  # (when, user id) of the last barge-in, set in the sink thread (see _barge_fizzled)
        self._owed_to: list[int] = []
        self.bye_at = 0.0  # when it last said goodbye / was asked to leave: a kick right after isn't worth a comeback  # who a dropped/cut reply was for: its redo keeps them (barge_in_scope)
        self._echo_seen: dict[int, deque[bool]] = {}  # user id -> were their last lines copies of someone else's?
        self._echoers: frozenset[int] = frozenset()  # mics picking up the call (read in the sink thread)
        self._addr_checks: dict[tuple[int, str], asyncio.Task] = {}  # "is this follow-up for me?" (see addressee.py)
        self.scene = Scene(self.cfg, bot.llm, self.cfg.bot.name)  # rolling catch-up notes (see scene.py)
        self._scene_task: asyncio.Task | None = None
        self._last_heard = 0.0
        self.last_metrics: dict[str, float] = {}

        self._duck_lock = threading.Lock()
        self._duckers: set[int] = set()
        self._gain = 1.0

        self._start_frames = max(1, self.v.start_ms // FRAME_MS)
        self._barge_frames = max(1, self.v.barge_in_ms // FRAME_MS)
        self._min_frames = max(1, self.v.min_speech_ms // FRAME_MS)
        self._preroll_frames = max(1, self.v.preroll_ms // FRAME_MS)
        self._allowed = set(self.cfg.discord.allowed_user_ids)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.vc.listen(VoiceSink(self), after=self._receive_ended)
        self._tasks = [
            asyncio.create_task(self._watchdog(), name="vad-watchdog"),
            asyncio.create_task(self._worker(), name="voice-worker"),
            asyncio.create_task(self._scene_keeper(), name="scene-notes"),
        ]
        log.info("Listening in #%s (%s)", self.vc.channel, self.vc.guild)

    async def close(self) -> None:
        self.interrupt()
        for t in self._tasks:
            t.cancel()
        try:
            if self.vc.is_listening():
                self.vc.stop_listening()
        except Exception:  # noqa: BLE001
            pass
        if self.vc.is_connected():
            await self.vc.disconnect(force=True)

    def reset(self) -> None:
        self.history.clear()
        self._last_presence = ""  # re-announce what people are doing to the fresh conversation
        self._search_memory.clear()
        self.scene.reset()

    async def _scene_keeper(self) -> None:
        """Refreshes the catch-up notes in a pause, when enough new lines came in. Never while a reply is on
        its way, and it's cancelled the moment one starts (_pause_background)."""
        while True:
            await asyncio.sleep(1)
            if (not self.scene.due() or self._busy or self._processing or self._spec is not None
                    or self.bot.is_muted(self.vc.guild.id)):
                continue
            if (time.monotonic() - max(self._last_heard, self._last_reply_at) < float(self.cfg.scene.quiet_s)
                    or any(st.speaking for st in list(self._states.values()))):
                continue
            task = self._scene_task = asyncio.create_task(self.scene.update(self._presence()))
            await asyncio.wait({task})
            if not task.cancelled() and task.exception():
                log.warning("Scene notes update failed: %r", task.exception())
                await asyncio.sleep(30)

    def _pause_background(self) -> None:
        """A reply is coming: free the LLM (scene notes, profiles, lore)."""
        self.bot.profiles.activity()
        if self._scene_task and not self._scene_task.done():
            self._scene_task.cancel()

    def post_event(self, text: str, respond: bool) -> None:
        """Something happened in the channel (join/leave). Goes into the conversation as [text]."""
        self._queue.put_nowait(Utterance(0, "", b"", time.monotonic(), text=text, event=True, respond=respond))

    def humans_in_channel(self) -> list[str]:
        channel = self.vc.channel
        return sorted(m.display_name for m in channel.members if not m.bot) if channel else []

    # ------------------------------------------------------------------ audio in (sink thread)

    def on_audio(self, user: discord.abc.User, pcm48: bytes) -> None:
        if self.clips is not None:  # clips hear the whole channel, including people/bots we don't answer
            self.clips.add(user.id, pcm48, time.monotonic())
        if (self.v.ignore_bots and user.bot) or (self._allowed and user.id not in self._allowed):
            return
        pcm16 = discord_to_16k_mono(pcm48)
        voiced = len(pcm16) == STT_FRAME_BYTES and self.vad.is_speech(pcm16, STT_SR)
        now = time.monotonic()

        st = self._states.get(user.id)
        if st is None:
            st = self._states[user.id] = _SpeakerState(getattr(user, "display_name", user.name),
                                                       self._preroll_frames)
        utt = None
        with st.lock:
            if not st.speaking:
                st.preroll.append(pcm16)
                st.consecutive_voiced = st.consecutive_voiced + 1 if voiced else 0
                if st.consecutive_voiced >= self._start_frames:
                    st.speaking = True
                    st.frames = list(st.preroll)
                    st.preroll.clear()
                    st.voiced_frames = st.consecutive_voiced
                    st.last_voiced_idx = len(st.frames)
                    st.started = st.last_voice = now
                    st.overlapped = self._busy
                    spec = self._spec
                    if spec is not None and spec.user_id != user.id:
                        self.loop.call_soon_threadsafe(self._cancel_spec, f"{st.name} started talking")
                    if self._busy and self._source is not None and user.id not in self._echoers:
                        self._set_duck(user.id, True)  # "I hear you" - lower our voice right away
            else:
                st.frames.append(pcm16)
                if voiced:
                    st.voiced_frames += 1
                    st.last_voiced_idx = len(st.frames)
                    st.last_voice = now
                    if st.spec_future is not None:  # they kept talking: speculative transcript is stale
                        st.spec_future.cancel()
                        st.spec_future = None
                        spec = self._spec
                        if spec is not None and spec.user_id == user.id:
                            self.loop.call_soon_threadsafe(self._cancel_spec, f"{st.name} kept talking")
                if (self._busy and self._source is not None and self.v.barge_in and not st.barged
                        and user.id not in self._echoers
                        and st.voiced_frames >= self._barge_frames
                        and (self.v.barge_in_scope == "anyone" or not self._reply_to or user.id in self._reply_to)):
                    st.barged = True
                    self._barge_pending = True
                    self.bot.counters["barge-ins"] += 1
                    log.info("Barge-in by %s", st.name)
                    self._barge = (now, user.id)
                    self.loop.call_soon_threadsafe(self.interrupt)
                if now - st.started >= self.v.max_utterance_s:
                    utt = self._finalize_locked(user.id, st)
        if utt:
            self.loop.call_soon_threadsafe(self._queue.put_nowait, utt)

    def _finalize_locked(self, user_id: int, st: _SpeakerState) -> Utterance | None:
        utt = None
        if st.voiced_frames >= self._min_frames:
            # drop trailing silence (keep 100ms) so STT has less to chew on
            pcm = b"".join(st.frames[: st.last_voiced_idx + 5])
            spec = st.spec_future if st.spec_idx == st.last_voiced_idx else None
            utt = Utterance(user_id, st.name, pcm, st.last_voice, spec, overlapped=st.overlapped,
                            started_at=st.started)
        elif st.spec_future is not None:
            st.spec_future.cancel()
        self._set_duck(user_id, False)
        st.reset()
        return utt

    def _set_duck(self, user_id: int, on: bool) -> None:
        with self._duck_lock:
            if on:
                self._duckers.add(user_id)
            else:
                self._duckers.discard(user_id)
            self._gain = float(self.v.duck_volume) if self._duckers else 1.0
        source = self._source
        if source is not None:
            source.gain = self._gain

    def _receive_ended(self, error: Exception | None) -> None:
        """voice_recv's reader thread stopped (runs on that thread). The watchdog restarts it; this says why."""
        if error is not None:
            log.warning("Voice receive ended with an error in #%s: %r", self.vc.channel, error,
                        exc_info=(type(error), error, error.__traceback__))

    def _check_listening(self) -> None:
        """voice_recv stops listening for good when its receive thread hits an error (e.g. a bad packet), which
        leaves the bot connected but deaf. Start listening again."""
        if self.vc.is_connected() and not self.vc.is_listening():
            log.warning("Voice receive stopped in #%s - listening again", self.vc.channel)
            try:
                self.vc.listen(VoiceSink(self), after=self._receive_ended)
            except Exception:  # noqa: BLE001 - try again on the next check
                log.exception("Couldn't restart voice receive")

    async def _watchdog(self) -> None:
        silence = self.v.silence_ms / 1000
        short = self.v.silence_short_ms / 1000
        long = self.v.silence_long_ms / 1000
        spec_after = self.v.speculative_stt_ms / 1000 if self.v.speculative_stt_ms else 0
        next_ear_check = 0.0
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            if now >= next_ear_check:
                next_ear_check = now + 5
                self._check_listening()
            for uid, st in list(self._states.items()):
                utt = None
                with st.lock:
                    if not st.speaking:
                        continue
                    quiet = now - st.last_voice
                    limit = silence
                    f = st.spec_future
                    if f is not None and f.done() and not f.cancelled() and f.exception() is None:
                        # Adaptive end-of-turn: "...and then I" -> keep listening; "What do you think?" -> go.
                        verdict = looks_finished(f.result())
                        if verdict is True:
                            limit = short
                        elif verdict is False:
                            limit = long
                    if quiet >= limit:
                        utt = self._finalize_locked(uid, st)
                    elif (f is not None and f.done() and not f.cancelled() and f.exception() is None
                          and st.spec_replied is not f and self._can_speculate(uid)):
                        st.spec_replied = f
                        self._start_spec(uid, st.name, f, st.last_voice, b"".join(st.frames[: st.spec_idx + 5]))
                    elif (spec_after and quiet >= spec_after and f is None
                          and st.voiced_frames >= self._min_frames):
                        # Speculative STT: transcribe during the pause so the text is (nearly) ready
                        # when the turn ends, and so we can judge whether the turn has ended.
                        st.spec_idx = st.last_voiced_idx
                        pcm = b"".join(st.frames[: st.last_voiced_idx + 5])
                        st.spec_future = self.bot.stt_executor.submit(self.bot.stt.transcribe, pcm, self._stt_hint)
                if utt:
                    self._queue.put_nowait(utt)

    # ------------------------------------------------------------------ processing (event loop)

    async def _worker(self) -> None:
        while True:
            batch = [await self._queue.get()]
            while not self._queue.empty():  # merge everything that piled up while we were busy
                batch.append(self._queue.get_nowait())
            self._processing = True
            try:
                await self._process(batch)
            except asyncio.CancelledError:
                if self._reply_task:
                    self._reply_task.cancel()
                raise
            except Exception:  # noqa: BLE001
                log.exception("Voice turn failed")
            finally:
                self._processing = False

    async def _process(self, batch: list[Utterance]) -> None:
        self._pause_background()  # pause background LLM work: the LLM is needed live
        spec, self._spec = self._spec, None
        owed, self._owe_reply = self._owe_reply, False
        try:
            await self._handle_batch(batch, spec, owed)
        finally:
            if spec is not None and not spec.gate.is_set():
                spec.task.cancel()

    async def _handle_batch(self, batch: list[Utterance], spec: _Speculation | None, owed: bool) -> None:
        t0 = time.perf_counter()
        results: list[tuple[Utterance, str]] = []
        for u in batch:
            if u.event:
                results.append((u, u.text))
                continue
            if u.stt_future is not None:
                text = await asyncio.wrap_future(u.stt_future)
            else:
                text = await self.loop.run_in_executor(self.bot.stt_executor, self.bot.stt.transcribe, u.pcm,
                                                       self._stt_hint)
            if not text:
                continue
            if u.overlapped and is_backchannel(text):
                self.bot.counters["backchannels ignored"] += 1
                log.info("(backchannel from %s ignored: %s)", u.name, text)
                continue
            results.append((u, text))
        results = self._drop_echoes(results)
        stt_ms = (time.perf_counter() - t0) * 1000
        self.bot.counters["utterances"] += sum(1 for u, _ in results if not u.event)
        if not results:
            if resume := self._barge_fizzled([]):  # the interruption had no words: finish what it was saying
                self.history.append({"role": "user", "content": resume})
                await self._run_reply(asyncio.create_task(self._reply(time.monotonic(), stt_ms, self._owed_to)))
                return
            if owed:  # only backchannels came in while we waited: still answer the earlier turn
                await self._run_reply(asyncio.create_task(self._reply(time.monotonic(), stt_ms, self._owed_to)))
            return

        speech = [(u, t) for u, t in results if not u.event]
        if self._barge[1] and any(u.user_id == self._barge[1] for u, _ in speech):
            self._barge = (0.0, 0)  # whoever cut it off said real words: nothing to resume (see _barge_fizzled)
        where = f"🔊 {self.vc.channel.name}" if self.vc.channel else ""
        for u, text in results:
            if u.event:
                log.info("• %s", text)
                self.bot.feed(self.vc.guild, "event", "", text, where)
            else:
                log.info("🎙 %s: %s", u.name, text)
                self.bot.feed(self.vc.guild, "heard", u.name, text, where)
        if speech:
            self.bot.post_transcript("\n".join(f"🎙 **{u.name}:** {t}" for u, t in speech))
        for u, t in speech:
            self.bot.profiles.observe(u.user_id, u.name, f"{u.name}: {t}", self.vc.guild.get_member(u.user_id))
            self.bot.lore.observe(self.vc.guild, u.user_id, f"{u.name}: {t}")
            self._said.append((u.user_id, u.name, t, u.started_at or u.ended_at))
        for u, t in results:
            self.scene.add(f"[{t}]" if u.event else f"{u.name}: {t}")
        self._last_heard = time.monotonic()

        fresh = float(self.cfg.bot.get("fresh_after_min") or 0) * 60
        if fresh > 0 and self.history and time.monotonic() - self._last_turn > fresh:
            log.info("🆕 New voice conversation (quiet for %.0f min)", (time.monotonic() - self._last_turn) / 60)
            self.reset()
        self._last_turn = time.monotonic()
        note = self._presence()
        if note != self._last_presence:
            self._last_presence = note
            log.info("👀 %s", note or "(no activities)")
        self._update_stt_hint()
        content = self._turn_content("\n".join(f"[{t}]" if u.event else f"{u.name}: {t}" for u, t in results), note)
        self.history.append({"role": "user", "content": content})
        self._trim_history()
        if self.bot.is_muted(self.vc.guild.id):
            return  # muted from the dashboard: keep following the conversation, never answer

        talking_to = [(u, t) for u, t in speech if self._is_addressed(t, u.user_id)]
        if (hush := next(((u, t) for u, t in speech if quiet_request(t) and (self._named(t) or (u, t) in talking_to)),
                         None)) is not None:
            # "Shut up" / "nobody's talking to you": stop joining in without the name, and don't answer that line.
            log.info("🤐 %s told me to be quiet - follow-ups need my name again", hush[0].name)
            self._partners.clear()
            self._last_reply_at = 0.0
            self._owe_reply = False
            talking_to = [x for x in talking_to if x is not hush and self._named(x[1]) and not quiet_request(x[1])]
        talking_to = await self._really_for_me(talking_to)
        resume_note = None
        if (self._cut and self.v.resume_after_cut_s and time.monotonic() - self._cut[0] < float(self.v.resume_after_cut_s)
                and (asked := [(u, t) for u, t in speech if continue_request(t)])):
            _, said, unsaid = self._cut
            self._cut = None
            log.info("↩ %s asked me to carry on", asked[0][0].name)
            resume_note = (f"[{asked[0][0].name} wants you to carry on: you were cut off mid-reply. "
                           + (f"You'd said: \"{said} —\". " if said else "You hadn't said anything yet. ")
                           + (f"You were about to say: \"{unsaid}\". " if unsaid else "")
                           + "Finish that thought now, without repeating what you already said and without "
                             "commenting on being interrupted.]")
            talking_to += [x for x in asked if x not in talking_to]  # a "go on" needs no wake word
        addressed = bool(talking_to)
        named = any(self._named(t) for _, t in speech)
        if addressed:
            for u, t in talking_to:
                since = u.started_at - self._last_reply_at if self._last_reply_at and u.started_at else None
                self.tuner.heard(t, self._named(t) and self.v.response_mode == "wake_word", since)
        # Read everyone's mood in the background (keeps voice baselines + learning current even when we don't
        # reply); the reply picks the results up after its search check, so it usually costs nothing.
        moods = [f for u, t in speech if (f := self._read_mood(u.user_id, u.name, t, u.pcm, addressed))]
        clip_note = await self._clip_requests(speech, addressed)
        quote_note = self._quote_requests(speech, addressed)
        names = list(self.v.wake_words) or [self.cfg.bot.name]
        leaving = (next((u for u, t in speech if leave_request(t, names)), None)
                   if addressed and self.v.leave_on_request else None)
        leaver = leaving.name if leaving else None
        if leaver:
            self.bye_at = time.monotonic()
        leave_note = (f"[{leaver} asked you to leave the call. You're leaving right after this line: say a quick bye "
                      "in a few words, don't argue or ask to stay.]") if leaver else None
        wants_reply = (owed or any(u.event and u.respond for u, _ in results) or addressed or clip_note is not None
                       or quote_note is not None)
        member_note = None
        if talking_to:
            said = " ".join(t for _, t in talking_to)
            member_note = calc.note(said)  # "what's 17 times 23" / "5 miles in km" / "time in Tokyo": exact
            if member_note is None and self.cfg.fun.enabled:  # coin / dice / pick someone / teams: really random
                if member_note := fun.note(said, self.humans_in_channel(), talking_to[0][0].name):
                    log.info("🎲 %s", member_note)
            if member_note is None:  # "what's the weather in Denver tomorrow"
                member_note = await self.bot.weather.note(said)
            if member_note:
                log.info("🧮 %s", member_note)
            requester = self.vc.guild.get_member(talking_to[0][0].user_id)
            # "roast jordan" / "rizz him up"
            if member_note is None and (mode := members.roast_request(said)) and requester is not None:
                member_note = self._roast_note(said, requester, mode)
            if member_note is None:  # "what roles does Casey have?" / "check my roles": the server, not the web
                member_note = await members.lookup(self.vc.guild, requester, said,
                                                   skip_names=[*self.v.wake_words, self.cfg.bot.name])
        focus_note = self._focus_note(speech, talking_to)
        extra = "\n".join(n for n in (focus_note, clip_note, quote_note, leave_note, resume_note, member_note,
                                      await self._run_commands(speech) if wants_reply and speech else None)
                          if n) or None  # clip/command results the reply should confirm
        # Answered only because it's inside the follow-up window (wake-word mode, nobody said its name).
        followup = self.v.response_mode == "wake_word" and addressed and not named
        if (spec is not None and extra is None and wants_reply and not spec.task.done() and len(results) == 1
                and results[0][0].stt_future is spec.future and content == spec.content):
            # The reply started during their pause is for exactly this turn: let it play now.
            spec.metrics["turn_end_ms"] = (time.monotonic() - results[0][0].ended_at) * 1000
            self.bot.counters["speculative used"] += 1
            self._busy = True
            spec.gate.set()
            await self._run_reply(spec.task)
            self._note_partners(talking_to)
            return
        if spec is not None:
            log.debug("(speculative reply didn't match the turn - dropped)")
        if not wants_reply and (resume := self._barge_fizzled(speech)):
            self.history[-1]["content"] += "\n" + resume  # this turn's lines were nothing to answer: carry on
            await self._run_reply(asyncio.create_task(self._reply(time.monotonic(), stt_ms, self._owed_to)))
            return
        if not wants_reply:
            if moods and (note := await self._check_in_note(moods)):
                await self._run_reply(asyncio.create_task(self._reply(
                    max(u.ended_at for u, _ in results), stt_ms, [int(note[0])], note=note[1])))
            return
        ended_at = max(u.ended_at for u, _ in results)
        speaker_ids = [u.user_id for u, _ in speech]
        await self._run_reply(asyncio.create_task(self._reply(ended_at, stt_ms, speaker_ids, note=extra,
                                                              followup=followup, moods=moods,
                                                              protected=leaver is not None)))
        self._note_partners(talking_to)
        if leaver:
            await asyncio.sleep(0.5)  # let the last of the goodbye play out
            log.info("👋 %s asked me to leave - leaving #%s", leaver, self.vc.channel)
            asyncio.create_task(self.bot.leave(self.vc.guild.id))  # not awaited: it closes this worker
        elif self.leave_after_reply:
            self.leave_after_reply = False
            log.info("👋 said my piece - leaving #%s on my own terms", self.vc.channel)
            await asyncio.sleep(0.5)
            asyncio.create_task(self.bot.leave(self.vc.guild.id))

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    @classmethod
    def _copies(cls, a: str, b: str) -> tuple[bool, bool]:
        """(near-identical, one mostly inside the other) - "Hey man, what's her problem?" inside
        "No, I was bored. Hey man, what's her problem?" counts as a copy."""
        na, nb = cls._norm(a), cls._norm(b)
        if min(len(na), len(nb)) < 8:
            return False, False
        sm = SequenceMatcher(None, na, nb)
        inside = sum(bl.size for bl in sm.get_matching_blocks()) / min(len(na), len(nb))
        return sm.ratio() > 0.8, inside > 0.8

    def _drop_echoes(self, results: list[tuple[Utterance, str]]) -> list[tuple[Utterance, str]]:
        """One person's voice picked up by someone else's mic (speakers instead of headphones, same room) arrives
        a second time from the other person - Static heard things twice, and the copy cut it off (barge-in) and made
        it wait for the floor. A line that's a near-copy of someone else's line from the last few seconds is
        dropped. Someone whose lines keep being copies is an echo source (_echoers): they're still heard, but they
        can't duck/interrupt Static or hold up its replies, and their mixed-in copies are dropped too."""
        keep: list[tuple[Utterance, str]] = []
        now = time.monotonic()
        recent = [(uid, name, text, at) for uid, name, text, at in self._said if uid and now - at < 8]
        for u, t in sorted(results, key=lambda r: r[0].started_at or r[0].ended_at):
            if u.event:
                keep.append((u, t))
                continue
            start = u.started_at or u.ended_at
            others = [(k[0].name, k[1], k[0].started_at or k[0].ended_at) for k in keep if k[0].user_id != u.user_id]
            others += [(name, text, at) for uid, name, text, at in recent if uid != u.user_id]
            same = inside = None
            for name, text, at in others:
                if abs(at - start) < 3:
                    twin, contained = self._copies(text, t)
                    if twin or contained:
                        same, inside = (name if twin else same), (name if contained else inside)
            seen = self._echo_seen.setdefault(u.user_id, deque(maxlen=8))
            seen.append(bool(same or inside))
            self._update_echoers(u)
            if same or (inside and u.user_id in self._echoers):
                log.info("(dropped %s's copy of %s's line - their mic picked it up: %s)", u.name, same or inside, t)
                continue
            keep.append((u, t))
        return keep

    def _update_echoers(self, u: Utterance) -> None:
        seen = self._echo_seen[u.user_id]
        echoing = sum(seen) >= 3
        if echoing and u.user_id not in self._echoers:
            self._echoers = self._echoers | {u.user_id}
            log.info("🔁 %s's mic is picking up the call (speakers?) - they can't cut Static off or hold up its "
                     "replies until it stops", u.name)
        elif not echoing and u.user_id in self._echoers and len(seen) == seen.maxlen:
            self._echoers = self._echoers - {u.user_id}
            log.info("🔁 %s's mic sounds clean again", u.name)

    @staticmethod
    def _focus_note(speech: list[tuple[Utterance, str]], talking_to: list[tuple[Utterance, str]]) -> str | None:
        """Several people spoke in one turn and only some of it was to the bot: say who to answer. The 4B model
        otherwise answers whichever line is last or loudest ("Hey Static, how come you never talk to me?" got an
        answer to someone else's aside)."""
        speakers = {u.user_id for u, _ in speech}
        if len(speakers) < 2 or not talking_to or len(talking_to) == len(speech):
            return None
        names = list(dict.fromkeys(u.name for u, _ in talking_to))
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return (f"[{who} {'is' if len(names) == 1 else 'are'} talking to you; the other lines were said to each other. "
                f"Answer {who}.]")

    def _barge_fizzled(self, speech: list[tuple[Utterance, str]]) -> str | None:
        """Someone talked over a reply and it stopped (barge-in), but whoever did it said nothing usable: game or
        stream audio, a cough, a "yeah". The reply was left hanging (a question went unanswered and it looked
        broken), so finish it. None while they're still talking, once they said real words, or if it's old."""
        at, uid = self._barge
        if not at or not self._cut or time.monotonic() - at > 12:
            return None
        st = self._states.get(uid)
        if (st is not None and st.speaking) or any(u.user_id == uid for u, _ in speech):
            return None
        self._barge = (0.0, 0)
        _, said, unsaid = self._cut
        self._cut = None
        log.info("↩ %s's interruption had no words in it - finishing the reply", st.name if st else uid)
        if not said:  # cut off before it said anything: just answer
            return ("[You were about to answer but got cut off by background noise - nobody said anything to you. "
                    "Answer the last thing said to you now, without commenting on being interrupted.]")
        return ("[You got cut off mid-reply, but it was only background noise - nobody said anything to you. "
                f"You'd said: \"{said} —\". " + (f"You were about to say: \"{unsaid}\". " if unsaid else "")
                + "Finish that thought now, without repeating what you already said and without commenting on "
                  "being interrupted.]")

    def _recent_lines(self, n: int = 12) -> list[tuple[str, str]]:
        """(name, text) of the last things people said, oldest first."""
        out = []
        for m in self.history[-n:]:
            if m["role"] == "user":
                out += [tuple(ln.split(": ", 1)) for ln in m["content"].splitlines()
                        if ": " in ln and not ln.lstrip().startswith("[")]
        return out

    def _roast_note(self, said: str, requester, mode: str) -> str | None:
        channel = self.vc.channel
        pool = [*(channel.members if channel else []), *self.vc.guild.members]
        recent = self._recent_lines()
        target = members.roast_target(pool, said, requester, [name for name, _ in recent])
        if target is None:
            return None
        return self.bot.roast_note(requester, target, [t for name, t in recent if name == target.display_name], mode)

    async def _check_in_note(self, moods: list[asyncio.Future]) -> tuple[int, str] | None:
        """Someone not talking to the bot sounds upset, and it's not a one-off: (their id, a note asking the bot to
        check on them). Conservative on purpose - this crowd trash-talks, and a bot that "checks in" every time
        someone swears gets old fast."""
        c = self.cfg.sentiment.get("check_in") or {}
        if not c.get("enabled") or self._busy:
            return None
        reads = [r for r in await asyncio.gather(*moods, return_exceptions=True)
                 if r is not None and not isinstance(r, BaseException)]
        now = time.monotonic()
        for r in reads:
            if r.group not in c.get("groups", ()) or r.score < float(c.get("min_score", 0.75)) or r.misread:
                continue
            louder = any("louder" in t for t in r.tone)
            q = self._upset.setdefault(r.user_id, deque(maxlen=8))
            q.append((now, r.score, louder))
            while q and now - q[0][0] > float(c.get("window_s", 120)):
                q.popleft()
            strong = r.score >= float(c.get("strong_score", 0.9)) and louder
            if not (len(q) >= int(c.get("needed", 2)) or strong):
                continue
            if (now - self._last_check_in < float(c.get("cooldown_min", 10)) * 60
                    or now - self._checked_on.get(r.user_id, -1e9) < float(c.get("per_person_min", 30)) * 60
                    or now - self._last_reply_at < 20):
                log.info("💢 %s sounds %s (%.2f) - not checking in (cooldown)", r.name, r.group, r.score)
                continue
            self._last_check_in = self._checked_on[r.user_id] = now
            q.clear()
            log.info("💢 %s sounds %s (%.2f%s) - checking in", r.name, r.group, r.score, ", louder" if louder else "")
            how = {"angry": "pretty heated", "sad": "down", "anxious": "stressed"}.get(r.group, "upset")
            member = self.vc.guild.get_member(r.user_id)
            doing = "; ".join(member_presence(member)) if member is not None else ""
            said = [ln.split(": ", 1)[1] for m in self.history[-8:] if m["role"] == "user"
                    for ln in m["content"].splitlines() if ln.startswith(f"{r.name}: ")][-3:]
            context = (f" What they're doing right now: {doing}." if doing else "") + (
                " Their last lines: " + " / ".join(f'"{x}"' for x in said) + "." if said else "")
            return r.user_id, (
                f"[Nobody's talking to you, but {r.name} sounds {how} right now.{context} Chime in once, on your own "
                "and briefly, to check on them. If what they're doing or saying explains it (losing a match, a bad "
                "pull, a game or song they're stuck on), work that in - that's what makes it land. Either sincere "
                "('you good?') or a light joke to take the edge off (like asking if they need an adult). Match the "
                "room - don't lecture, don't make it a big deal, and don't mention reading anyone's mood.]")
        return None

    def _read_mood(self, uid: int, name: str, text: str, pcm: bytes | None, to_bot: bool,
                   observe: bool = True) -> asyncio.Future | None:
        if self.bot.mood.model is None or not text:
            return None
        return self.loop.run_in_executor(self.bot.mood_executor, self.bot.mood.read, uid, name, text, pcm, to_bot,
                                         observe)

    async def _clip_requests(self, speech: list[tuple[Utterance, str]], addressed: bool) -> str | None:
        """'Static, clip that' / 'clip the last 10 seconds'. A bare 'clip that' works without the wake word."""
        if self.clips is None:
            return None
        c = self.cfg.clips
        for u, text in speech:
            seconds = clip_request(text, float(c.buffer_s), float(c.default_s))
            if seconds is None or not (addressed or len(text.split()) <= 6):
                continue
            if time.monotonic() - self._last_clip_at < float(self.cfg.clips.get("dedupe_s", 10)):
                log.info("(clip request from %s skipped: one was just posted)", u.name)
                return f"[{u.name} asked for a clip too, but you posted one of this exact moment seconds ago. Tell them it's already up.]"
            # End just before they started saying it, so the clip is the moment, not the request.
            end = (u.started_at or u.ended_at) - self.v.preroll_ms / 1000 - 0.1
            audio, who = self.clips.clip(end, seconds)
            if audio.size < DISCORD_SR:
                return f"[{u.name} asked you to clip that, but you haven't heard anything in the last {seconds:.0f}s. Tell them.]"
            note = await self.bot.post_clip(self.vc.channel, audio, who, u.name)
            if "refused" not in note:
                self._last_clip_at = time.monotonic()
            return note
        return None

    def _quote_requests(self, speech: list[tuple[Utterance, str]], addressed: bool) -> str | None:
        """'Static, quote that' saves the last line someone else said (or 'quote what riley said' / 'quote me');
        'give us a random quote' reads one out. A bare 'quote that' works without the wake word, like clips."""
        if not self.cfg.quotes.enabled:
            return None
        book = self.bot.quotes
        gid = self.vc.guild.id
        for u, text in speech:
            req = quotes.save_request(text)
            if req is not None and (addressed or len(text.split()) <= 6):
                who = req[1]
                window = float(self.cfg.quotes.get("window_s", 90))
                start = u.started_at or u.ended_at
                earlier = [x for x in self._said if x[3] < start - 0.05 and start - x[3] < window]
                if who == "me":
                    earlier = [x for x in earlier if x[0] == u.user_id]
                elif who:
                    earlier = [x for x in earlier if x[1].lower().startswith(who.lower())]
                else:
                    earlier = [x for x in earlier if x[0] != u.user_id] or earlier
                if not earlier:
                    return f"[{u.name} asked you to quote that, but you didn't catch a line to save. Tell them.]"
                uid, name, said, _ = earlier[-1]
                if (qid := book.exists(gid, said)) is not None:
                    return f"[{u.name} asked you to quote that, but it's already quote #{qid}. Say so.]"
                qid = book.add(gid, uid, name, said, time.time(), u.name, "voice")
                log.info("💬 quote #%d (%s) saved by %s: %s", qid, name, u.name, said)
                self._post(quotes.show(book.get(qid)) + f" · saved by {u.name}")
                return f"[You just saved {name}'s line \"{said}\" to the quote book as #{qid}. React in a few words.]"
            if addressed and quotes.read_request(text):
                row = book.random(gid)
                if row is None:
                    return "[They want a quote from the quote book, but it's empty. Tell them to say \"quote that\" after something good.]"
                return (f"[Read out quote #{row['id']} from the quote book: {row['name']} once said \"{row['text']}\". "
                        "Say who said it and the quote itself, word for word, then one short line of your own.]")
        return None

    def _post(self, text: str) -> None:
        """A line in the voice channel's chat (Fluxer: the session's text channel). Never raises."""
        channel = self.vc.channel

        async def send():
            try:
                await channel.send(text)
            except Exception as e:  # noqa: BLE001 - posting is a courtesy
                log.info("Couldn't post in #%s: %s", channel, e)
        if channel is not None:
            asyncio.create_task(send())

    async def _run_commands(self, speech: list[tuple[Utterance, str]]) -> str | None:
        """Reminder/poll requests in this turn ("remind us in 20 minutes to..."). Returns the note the reply
        confirms them with. Only ever runs for a confirmed turn - never speculatively - so nothing happens twice."""
        planner = self.bot.planner
        if not any(planner.wants(t) for _, t in speech):
            return None
        channel = self.vc.channel
        members = [m for m in channel.members if not m.bot] if channel else []
        context = [m["content"] for m in self.history[-4:-1]]
        notes = []
        for u, text in speech:
            author = self.vc.guild.get_member(u.user_id) if planner.wants(text) else None
            if author is not None and (note := await planner.handle(text, author, channel, context, members)):
                notes.append(note)
        return "\n".join(notes) or None

    async def _run_reply(self, task: asyncio.Task) -> None:
        self._reply_task = task
        await asyncio.wait({task})  # doesn't raise if the reply is cancelled by barge-in
        if not task.cancelled() and task.exception():
            log.error("Reply failed: %r", task.exception())

    def _presence(self) -> str:
        channel = self.vc.channel
        return self.bot.profiles.presence_note([m for m in channel.members if not m.bot] if channel else [])

    def _recent_searches(self, found: str | None) -> list[str]:
        """Earlier search results still worth showing (see search.remember_s), relabelled as background.
        Remembers `found` for the next turns if it has results."""
        now = time.monotonic()
        keep = float(self.cfg.search.remember_s or 0)
        earlier = []
        for at, text in self._search_memory:
            if now - at < keep and text != found:
                head, _, body = text.partition("\n")
                q = head.split('"')[1] if head.count('"') >= 2 else "that"
                earlier.append(f'[Earlier web search for "{q}" - kept in case they ask a follow-up about '
                               f"it; ignore it if they've moved on. Same rules: only facts stated here.]\n{body}")
        if found and found.startswith("[Web search") and keep > 0:
            self._search_memory.append((now, found))
        return earlier

    def _update_stt_hint(self) -> None:
        """Names Whisper should expect: people in the call, the games/apps they're in, and names the bot just
        said ("Arknights: Endfield", not "Arknight's infield"; "Purrchena", not "Pergina")."""
        if not self.v.stt_name_hints:
            return
        channel = self.vc.channel
        names: list[str] = []
        for m in (channel.members if channel else []):
            if m.bot:
                continue
            names.append(m.display_name)
            for a in getattr(m, "activities", None) or ():
                if not isinstance(a, (discord.CustomActivity, discord.Spotify)) and getattr(a, "name", None):
                    names.append(a.name)
        for msg in [m for m in self.history if m["role"] == "assistant"][-2:]:
            names += name_hints(msg["content"])
        seen, out = set(), []
        for n in reversed(names):  # newest first, so the cap drops the oldest
            if n.lower() not in seen and len(n) < 40:
                seen.add(n.lower())
                out.append(n)
        self._stt_hint = ", ".join(reversed(out[:15])) + "." if out else ""

    def _turn_content(self, lines: str, note: str) -> str:
        """What goes into history for a turn. The presence note is included only when it changed (or got
        trimmed out of history): it becomes part of the conversation, so the model knows without being
        nudged to mention it every turn, and the cached prompt prefix stays reusable."""
        if note and not any(note in m["content"] for m in self.history):
            return f"{note}\n{lines}"
        return lines

    # ------------------------------------------------------------------ speculative replies

    def _can_speculate(self, uid: int) -> bool:
        """Only when the bot is idle and this is the only person talking."""
        if not self.v.speculative_reply or self._spec is not None or self._busy or self._processing:
            return False
        if self.bot.is_muted(self.vc.guild.id):
            return False
        if (self._reply_task and not self._reply_task.done()) or not self._queue.empty():
            return False
        return not any(st.speaking for other, st in list(self._states.items())
                       if other != uid and other not in self._echoers)

    def _start_spec(self, uid: int, name: str, future: Future, ended_at: float, pcm: bytes) -> None:
        text = future.result()
        if (not text or not self._is_addressed(text, uid) or self.bot.planner.wants(text)
                or (self.clips is not None and clip_request(text, 60, 30) is not None)
                or quotes.save_request(text) is not None):
            return  # commands (reminders, polls, clips) have side effects: they wait for the confirmed turn
        self._pause_background()  # the LLM is about to be needed live
        content = self._turn_content(f"{name}: {text}", self._presence())
        gate, metrics = asyncio.Event(), {}
        followup = self._is_followup(text)
        mood = self._read_mood(uid, name, text, pcm, True, observe=False)  # the confirmed turn observes/learns
        # A follow-up gets its "is this for me?" check now, during the pause; the confirmed turn reuses it.
        addr = (self._addressee_task(uid, name, text, exclude_last=False)
                if followup and self.bot.addressee.enabled else None)
        task = asyncio.create_task(self._reply(ended_at, 0.0, [uid], pending=content, gate=gate, metrics=metrics,
                                               followup=followup, moods=[mood] if mood else None, addressee=addr),
                                   name="speculative-reply")
        self._spec = _Speculation(uid, future, content, gate, task, metrics)
        self.bot.counters["speculative started"] += 1
        log.debug("(speculating on %s: %s)", name, text)

    def _cancel_spec(self, reason: str) -> None:
        spec, self._spec = self._spec, None
        if spec is not None and not spec.gate.is_set():
            spec.task.cancel()
            log.debug("(speculative reply dropped: %s)", reason)

    def _named(self, text: str) -> bool:
        low = text.lower()
        return any(w.lower() in low for w in list(self.v.wake_words) or [self.cfg.bot.name])

    def _is_addressed(self, text: str, uid: int) -> bool:
        """Talking to the bot? Always in 'always' mode; otherwise when they say its name, or inside the
        (self-tuned) follow-up window after a reply. With followup_scope 'speaker' that window is only for
        the people it's been talking with - someone else chatting nearby needs the name to join in. If
        the last reply wasn't to anyone in particular (a greeting), anyone can answer it."""
        if self.v.response_mode != "wake_word" or self._named(text):
            return True
        now, window = time.monotonic(), self.tuner.followup()
        if now - self._last_reply_at >= window:
            return False
        if self.v.followup_scope == "anyone":
            return True
        active = {p for p, at in self._partners.items() if now - at < window}
        return uid in active or not active

    def _is_followup(self, text: str) -> bool:
        """Counts as talking to the bot only because it's inside the follow-up window (no name said)."""
        return self.v.response_mode == "wake_word" and not self._named(text)

    def _addressee_task(self, uid: int, name: str, text: str, exclude_last: bool) -> asyncio.Task:
        """The "is this follow-up for me?" check, shared by the speculative reply and the confirmed turn."""
        key = (uid, text)
        task = self._addr_checks.get(key)
        if task is None:
            recent = []
            for m in self.history[:-1] if exclude_last else self.history:
                if m["role"] == "assistant":
                    recent.append(f"{self.cfg.bot.name}: {m['content']}")
                else:
                    recent += [ln for ln in m["content"].splitlines() if ": " in ln and not ln.lstrip().startswith("[")]
            task = asyncio.create_task(self.bot.addressee.is_for_me(name, text, recent[-12:], self.humans_in_channel(),
                                                                    asked_them=uid in self._reply_to))
            if len(self._addr_checks) > 16:
                self._addr_checks.clear()
            self._addr_checks[key] = task
        return task

    async def _really_for_me(self, talking_to: list[tuple[Utterance, str]]) -> list[tuple[Utterance, str]]:
        """Drops follow-ups (no name said) that were really said to someone else ("is she roasting you?")."""
        if not self.bot.addressee.enabled:
            return talking_to
        checks = [self._addressee_task(u.user_id, u.name, t, exclude_last=True) if self._is_followup(t) else None
                  for u, t in talking_to]
        if not any(checks):
            return talking_to
        verdicts = [await c if c is not None else True for c in checks]
        for u, t in talking_to:
            self._addr_checks.pop((u.user_id, t), None)
        return [x for x, ok in zip(talking_to, verdicts) if ok]

    def _note_partners(self, talking_to: list[tuple[Utterance, str]]) -> None:
        """Whoever we just answered gets the follow-up window (see _is_addressed)."""
        now = time.monotonic()
        window = self.tuner.followup()
        self._partners = {p: at for p, at in self._partners.items() if now - at < window}
        for u, _ in talking_to:
            self._partners[u.user_id] = now

    def _context_note(self) -> str:
        channel = self.vc.channel
        members = [m for m in channel.members if not m.bot] if channel else []
        roster = self.bot.profiles.roster(members) if members else "- nobody but you"
        mine = self.bot.profiles.self_note(self.vc.guild)
        return f"People in the voice channel right now (display name, @username, roles):\n{roster}" + (
            f"\n{mine}" if mine else "")

    def _build_messages(self, pending: str | None = None) -> list[dict]:
        name = self.cfg.bot.name
        system = "\n\n".join([
            self.bot.persona(),
            self.v.system_prompt_suffix.replace("{name}", name).strip(),
            self.cfg.bot.time_prompt.strip(),
            self._context_note(),
        ])
        messages = [{"role": "system", "content": system}] + self.history
        if pending:  # a speculative reply: the turn isn't in history yet
            messages.append({"role": "user", "content": pending})
        if messages[-1]["role"] == "user":  # time, catch-up notes, length hint: this turn only, not history
            prefix = "\n".join(n for n in (now_note(), self.scene.note(), self.tuner.hint()) if n)
            messages[-1] = {"role": "user", "content": f"{prefix}\n{messages[-1]['content']}"}
        return messages

    def _trim_history(self) -> None:
        trim_history(self.history, int(self.cfg.bot.max_history_messages))

    async def _reply(self, ended_at: float, stt_ms: float, speaker_ids: list[int], pending: str | None = None,
                     gate: asyncio.Event | None = None, metrics: dict | None = None, note: str | None = None,
                     followup: bool = False, moods: list[asyncio.Future] | None = None,
                     addressee: asyncio.Task | None = None, protected: bool = False) -> None:
        """Generate and speak a reply. With `gate` (speculative) it runs ahead - search check, LLM, first
        TTS chunk - but holds the audio until the gate opens (the turn is confirmed over)."""
        if gate is None:
            self._busy = True
        self._barge_pending = False
        # Only the people this reply is for (and whoever it's in a conversation with) can talk it into stopping;
        # anyone else talking just turns it down. For nobody in particular (a greeting, an announcement): nobody
        # stops it that way - it's short, and bystanders' chatter or game audio used to cut it off.
        now = time.monotonic()
        self._reply_to = (frozenset(speaker_ids) | frozenset(
            p for p, at in self._partners.items() if now - at < self.tuner.followup())) if speaker_ids else frozenset({0})
        # Said in full, no matter what: the comeback line after being kicked, and the goodbye when asked to leave
        # (it used to wait for a gap, get superseded, and leave without a word).
        protected = protected or self.leave_after_reply
        if protected:
            self._reply_to = frozenset({0})  # nobody talks it into stopping (it still turns down while they talk)
        messages = self._build_messages(pending)
        speakers = {s.name for s in list(self._states.values())}  # list(): sink thread may add entries
        guard = SpeakerGuard(self.cfg.bot.name, set(self.humans_in_channel()) | speakers)
        repeats = RepeatGuard(recent_replies(self.history) + prompt_examples(messages[0]["content"]))
        turn = pending or (self.history[-1]["content"] if self.history and self.history[-1]["role"] == "user" else "")
        echo = EchoGuard([ln.split(": ", 1)[1] for ln in turn.splitlines()
                          if ": " in ln and not ln.lstrip().startswith("[")])
        # Memories from earlier calls that match what was just said (embedding on CPU, runs alongside the rest).
        memories = asyncio.create_task(self.bot.lore.recall(
            self.vc.guild.id, turn, [*self.humans_in_channel(), self.cfg.bot.name, *self.v.wake_words]))
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        generated: list[str] = []
        spoken: list[str] = []
        t_llm = time.perf_counter()
        metrics = {} if metrics is None else metrics
        metrics["stt_ms"] = stt_ms
        reads: list = []  # mood reads this reply acted on (scored on each person's next turn)
        if gate is None:
            metrics["turn_end_ms"] = (time.monotonic() - ended_at) * 1000

        def emit(chunk: str) -> bool:
            chunk = strip_stage(guard.strip_own(chunk if generated else guard.strip_label(chunk)))
            if not chunk:
                return True
            if not generated and echo.is_echo(chunk):
                log.info("(dropped the model's copy of what they said: %s)", chunk)
                return True
            if guard.is_other(chunk):
                log.info("(model started writing someone else's line - cut)")
                return False
            chunk, other = guard.split_other(chunk)
            if other:  # "What's up, guys? riley: Hey Static." - keep its part, stop there
                log.info("(model started writing someone else's line mid-sentence - cut)")
                if chunk and not repeats.is_repeat(chunk):
                    repeats.add(chunk)
                    generated.append(chunk)
                    sentences.put_nowait(chunk)
                return False
            if repeats.is_repeat(chunk):
                log.info("(repeat skipped: %s)", chunk)
                return True
            repeats.add(chunk)
            generated.append(chunk)
            sentences.put_nowait(chunk)
            return True

        async def produce():
            t_start = t_llm
            if addressee is not None:  # speculative follow-up: don't start a reply to a line meant for someone else
                if not await asyncio.shield(addressee):
                    return
                metrics["addressee_ms"] = (time.perf_counter() - t_llm) * 1000
                t_start = time.perf_counter()
            if note:  # a reminder/poll just ran: confirm it (this turn only, like search results)
                messages[-1] = {"role": "user", "content": f"{note}\n\n{messages[-1]['content']}"}
            elif self.cfg.search.voice and messages[-1]["role"] == "user":
                fillers = list(self.cfg.search.voice_fillers)
                say_filler = (lambda: sentences.put_nowait(random.choice(fillers))) if fillers else None
                found = await self.bot.search.lookup(messages[1:], say_filler, self._presence())
                metrics["search_ms"] = (time.perf_counter() - t_llm) * 1000
                # Not stored in history (history keeps what was said), but the last results stay around for a
                # few minutes so "who's that?" about a name from them doesn't get "never heard of her".
                blocks = self._recent_searches(found)
                if found:
                    blocks.append(found)
                if blocks:
                    messages[-1] = {"role": "user", "content": "\n\n".join(blocks + [messages[-1]["content"]])}
                t_start = time.perf_counter()
            try:
                lore_note = await memories
            except Exception as e:  # noqa: BLE001 - a memory is a nice-to-have
                log.warning("Lore recall failed: %s", e)
                lore_note = None
            if lore_note:
                messages[-1] = {"role": "user", "content": f"{lore_note}\n{messages[-1]['content']}"}
            if moods:  # started when the turn came in - normally done by now
                t_mood = time.perf_counter()
                reads[:] = [r for r in await asyncio.gather(*moods, return_exceptions=True)
                            if r is not None and not isinstance(r, BaseException)]
                metrics["mood_wait_ms"] = (time.perf_counter() - t_mood) * 1000
                if mood_note := self.bot.mood.choose(self.vc.guild.id, reads):
                    messages[-1] = {"role": "user", "content": f"{mood_note}\n{messages[-1]['content']}"}
                    log.info("🎭 %s", mood_note)
                t_start = time.perf_counter()
            think, actions, chunker = ThinkFilter(), ActionFilter(), SentenceChunker()
            async with aclosing(self.bot.llm.stream(messages, voice=True)) as tokens:
                async for tok in tokens:
                    if "llm_ttft_ms" not in metrics:
                        metrics["llm_ttft_ms"] = (time.perf_counter() - t_start) * 1000
                    text = actions.feed(think.feed(tok))
                    chunks = chunker.feed(text) if text else []
                    if chunks and "chunk1_ms" not in metrics:
                        metrics["chunk1_ms"] = (time.perf_counter() - t_start) * 1000
                    if not all(emit(c) for c in chunks):
                        return  # closing the stream stops generation server-side
            tail = actions.feed(think.flush()) + actions.flush()
            for c in chunker.feed(tail) + chunker.flush():
                if not emit(c):
                    return

        producer = asyncio.create_task(produce())
        producer.add_done_callback(lambda _: sentences.put_nowait(None))
        producer.add_done_callback(lambda _: memories.cancel())
        try:
            await self._speak_from_queue(sentences, spoken, ended_at, metrics, gate, protected=protected)
            await producer  # re-raise LLM errors
        except _Superseded:
            producer.cancel()
            self._owe_reply = True
            self._owed_to = [p for p in self._reply_to if p]
            self.bot.counters["replies redone after waiting"] += 1
            log.info("(waited for people to finish - answering everything together)")
            return
        except asyncio.CancelledError:
            producer.cancel()
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken) + " —"})
                self.bot.feed(self.vc.guild, "cut", self.cfg.bot.name, " ".join(spoken),
                              f"🔊 {self.vc.channel.name}" if self.vc.channel else "")
                if self._barge_pending:
                    self.tuner.interrupted(len(" ".join(spoken).split()), followup)
                    self.bot.mood.interrupted(self.vc.guild.id, reads)
                    # Kept for "go on" / "what were you saying?": what was said and what was still to come.
                    self._cut = (time.monotonic(), " ".join(spoken), " ".join(generated[len(spoken):]))
            elif self._barge_pending:  # stopped before a word came out: still resumable if it was only noise
                self._cut = (time.monotonic(), "", " ".join(generated))
            if self._barge_pending:
                self._owed_to = [p for p in self._reply_to if p]
            self._barge_pending = False
            raise
        finally:
            if self._reply_task is asyncio.current_task():
                self._busy = False
            if gate is None or gate.is_set():
                self._last_reply_at = time.monotonic()
                self.bot.profiles.activity(cancel=False)

        # History gets the cleaned text (no *actions*, no labels) so the model mimics clean speech.
        reply = " ".join(generated).strip()
        if reply and says_goodbye(reply):
            self.bye_at = time.monotonic()
        if reply:
            self._cut = None  # it got to finish: nothing to resume
            self.history.append({"role": "assistant", "content": reply})
            self._last_turn = time.monotonic()
            self._trim_history()
            self._update_stt_hint()
            self.bot.profiles.observe_reply(speaker_ids, f"{self.cfg.bot.name}: {reply}")
            self.bot.lore.observe(self.vc.guild, 0, f"{self.cfg.bot.name}: {reply}")
            self.scene.add(f"{self.cfg.bot.name}: {reply}")
            self._said.append((0, self.cfg.bot.name, reply, time.monotonic()))
            log.info("🤖 %s", reply)
            self.bot.feed(self.vc.guild, "reply", self.cfg.bot.name, reply,
                          f"🔊 {self.vc.channel.name}" if self.vc.channel else "")
            self.bot.post_transcript(f"🤖 **{self.cfg.bot.name}:** {reply}")
        if spoken:
            self.tuner.finished()
            self.bot.mood.replied(self.vc.guild.id, reads)
        self.last_metrics = metrics
        self.bot.voice_turns.append({**metrics, "speculative": gate is not None, "at": time.time()})
        self.bot.counters["voice replies"] += 1
        m = metrics.get
        how = "" if gate is None else " [speculative" + (
            f", ready {m('held_ms'):.0f}ms before the turn ended]" if "held_ms" in metrics else "]")
        log.info("Latency: end-of-speech -> audio=%.0fms%s | turn end %.0f, stt %.0f, search %.0f, "
                 "llm first token %.0f, first chunk %.0f, tts %.0f",
                 m("total_ms", 0), how, m("turn_end_ms", 0), m("stt_ms", 0), m("search_ms", 0),
                 m("llm_ttft_ms", 0), m("chunk1_ms", 0), m("tts_first_ms", 0))

    async def _wait_for_floor(self) -> bool:
        """Never start talking over someone: wait while anyone is mid-utterance (pauses included).
        False = someone said something new meanwhile, or is still going after floor_wait_s - drop this
        reply and answer everything together once they're done."""
        deadline = time.monotonic() + float(self.v.floor_wait_s)
        waited = False
        while any(st.speaking for uid, st in list(self._states.items()) if uid not in self._echoers):
            if time.monotonic() >= deadline:
                return False  # a stale reply after a long monologue would be odd: redo it with their words
            waited = True
            await asyncio.sleep(0.03)
        return not (waited and not self._queue.empty())

    async def _speak_from_queue(self, sentences: asyncio.Queue, spoken: list[str], ended_at: float | None,
                                metrics: dict | None = None, gate: asyncio.Event | None = None,
                                protected: bool = False) -> None:
        """protected: don't wait for a gap in the conversation (and never give up waiting) - just say it."""
        source = None
        done = asyncio.Event()
        try:
            while (s := await sentences.get()) is not None:
                speech = clean_for_speech(s)
                if not speech:
                    continue
                t = time.perf_counter()
                pcm = self.bot.tts_cache.get(speech) or await self.loop.run_in_executor(
                    self.bot.tts_executor, self.bot.tts.synthesize_discord, speech)
                if not pcm:
                    continue
                if source is None:
                    if metrics is not None:
                        metrics["tts_first_ms"] = (time.perf_counter() - t) * 1000
                    if gate is not None and not gate.is_set():
                        t_hold = time.monotonic()
                        await gate.wait()
                        if metrics is not None:
                            metrics["held_ms"] = (time.monotonic() - t_hold) * 1000
                    if not protected and not await self._wait_for_floor():
                        raise _Superseded
                    source = self._start_playback(done)
                    if metrics is not None and ended_at:
                        metrics["total_ms"] = (time.monotonic() - ended_at) * 1000
                source.feed(pcm)
                spoken.append(s)
            if source:
                source.finish()
                await done.wait()
        except asyncio.CancelledError:
            self._stop_playback()
            raise

    # ------------------------------------------------------------------ playback control

    def _start_playback(self, done: asyncio.Event) -> StreamingPCMSource:
        if self.vc.is_playing():
            self.vc.stop_playing()  # not vc.stop(): on a VoiceRecvClient that also stops receiving
        source = StreamingPCMSource()
        source.gain = self._gain
        if self.clips is not None:  # the bot's own voice goes in clips too (exactly as played, ducking included)
            bot_id = self.bot.user.id
            source.tap = lambda frame: self.clips.add(bot_id, frame, time.monotonic())

        def after(err):
            if err:
                log.error("Playback error: %s", err)
            self.loop.call_soon_threadsafe(done.set)

        self.vc.play(source, after=after)
        self._source = source
        return source

    def _stop_playback(self) -> None:
        if self._source:
            # Emptying the source ends playback on the player's next 20ms read, same as stopping it to the ear.
            self._source.clear()
            self._source = None
        elif self.vc.is_playing():
            self.vc.stop_playing()  # not vc.stop(): on a VoiceRecvClient that also stops receiving

    def interrupt(self) -> None:
        self._cancel_spec("interrupted")
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
        self._stop_playback()

    async def say(self, text: str) -> None:
        """Speak arbitrary text (used by /say)."""
        self.interrupt()
        q: asyncio.Queue[str | None] = asyncio.Queue()
        for s in split_sentences(text):
            q.put_nowait(s)
        q.put_nowait(None)
        self._busy = True
        task = self._reply_task = asyncio.create_task(self._speak_from_queue(q, [], None))

        def _done(_):
            if self._reply_task is task:
                self._busy = False

        task.add_done_callback(_done)
