"""Fluxer voice over LiveKit, behind the same calls VoiceSession makes on discord-ext-voice-recv's VoiceRecvClient
(listen / play / stop_playing / is_playing / is_connected / disconnect / .guild / .channel), so the whole voice
pipeline - VAD, speculative STT, barge-in, ducking, clips - runs on Fluxer unchanged.

Receive: LiveKit gives each person their own audio track. One task per track reads 20 ms frames (48 kHz stereo,
exactly Discord's PCM) into a queue; a single "fluxer-recv" thread hands them to the sink, like voice_recv's
decoder thread does on Discord (VAD stays off the event loop).
Send: one microphone track, published once at connect. A playback task pulls 20 ms frames from the
StreamingPCMSource (the same one Discord plays) into a small LiveKit buffer, so stopping is near-instant."""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import livekit.rtc as rtc
import numpy as np
import webrtcvad

from ..audio import DISCORD_SR, FRAME_BYTES, STT_SR, discord_to_16k_mono

if TYPE_CHECKING:
    from .objects import FxChannel, FxGuild

log = logging.getLogger("voicebot.fluxer.voice")


def _quiet_livekit_cleanup(unraisable, _default=sys.unraisablehook) -> None:
    """LiveKit's FfiHandle.__del__ asserts while handles are garbage-collected after a room closes (and at exit):
    harmless, but it filled the log with 'Exception ignored in FfiHandle.__del__ ... AssertionError' tracebacks."""
    if isinstance(unraisable.exc_value, AssertionError) and "FfiHandle" in repr(unraisable.object):
        return
    _default(unraisable)


sys.unraisablehook = _quiet_livekit_cleanup

CHANNELS = 2
FRAME_SAMPLES = FRAME_BYTES // 4  # 960 per channel

# Noise gate. Discord clients only send audio while someone talks, and the voice pipeline is built around that
# (a speaker "stops" when their packets stop). LiveKit sends every open mic all the time, room noise included,
# which kept people "mid-sentence" forever (the bot never got the floor) and fed Whisper noise ("Thank you.").
# So each person gets a gate: open when a frame is GATE_OPEN_DB above their own noise floor (and above
# GATE_MIN_DB), stay open GATE_HANG frames after the last loud one, and replay the GATE_PRE frames before it
# opened so the first syllable isn't clipped.
GATE_OPEN_DB = 12.0
GATE_MIN_DB = -52.0
GATE_HANG = 15   # 300 ms
GATE_PRE = 5     # 100 ms
# Loud isn't enough: clicks, bumps, breathing and fans passed the loudness gate and became Whisper's "Thank you."
# (dropped later, but only after they'd held up replies and once cut the bot off). So while the gate is closed, a
# loud frame only opens it if Silero VAD (the one faster-whisper ships, CPU, ~0.4 ms per check) hears a voice in
# the last 160 ms. Tested: speech 0.98-0.99, clicks/bumps/breathing/fan/mic rub 0.01-0.07. Once open it stays
# open while the sound goes on (+ GATE_HANG), so mid-sentence frames cost nothing. WebRTC VAD is the fallback.
GATE_SILERO = 0.5
GATE_VAD_MODE = 3
GATE_SPEECH = 2
GATE_REPORT_S = 300  # log who the gate held back, so noisy mics can be spotted

_silero = None  # shared by every gate (only used on the event loop)


def _silero_model():
    global _silero
    if _silero is None:
        try:
            from faster_whisper.vad import get_vad_model
            _silero = get_vad_model()
        except Exception as e:  # noqa: BLE001
            log.warning("Silero VAD unavailable for the Fluxer noise gate (%s) - using WebRTC VAD", e)
            _silero = False
    return _silero or None


@dataclass
class _Data:
    """What voice_recv hands the sink: .pcm = 20 ms of 48 kHz stereo s16le."""
    pcm: bytes


def uid_of(identity: str) -> int:
    """LiveKit identity 'user_<fluxer id>_<connection id>' -> the Fluxer user id (0 if it isn't one)."""
    parts = identity.split("_")
    try:
        return int(parts[1]) if len(parts) >= 2 and parts[0] == "user" else 0
    except ValueError:
        return 0


class NoiseGate:
    """One person's gate (see GATE_*): feed 20 ms frames, get back the frames to pass on (with the pre-roll when it
    opens) and whether this frame was loud noise held back."""

    def __init__(self):
        self.floor, self.hang = -60.0, 0
        self.pre: deque[bytes] = deque(maxlen=GATE_PRE)
        self.silero = _silero_model()
        self.vad = webrtcvad.Vad(GATE_VAD_MODE)
        self.votes: deque[bool] = deque(maxlen=3)  # WebRTC fallback: speech-or-not for the last loud frames
        self.recent: deque[np.ndarray] = deque(maxlen=8)  # last 160 ms at 16 kHz, for Silero

    def _voice(self, pcm16: bytes) -> bool:
        """Does the last bit of audio sound like someone talking?"""
        if self.silero is not None:
            audio = np.concatenate(self.recent)
            audio = audio[-(audio.size // 512) * 512:] if audio.size >= 512 else np.pad(audio, (0, 512 - audio.size))
            return float(np.max(self.silero(audio)[-3:])) >= GATE_SILERO
        self.votes.append(self.vad.is_speech(pcm16, STT_SR))
        return sum(self.votes) >= GATE_SPEECH

    def feed(self, data: bytes) -> tuple[list[bytes], bool]:
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        db = 10 * np.log10(float(np.mean(a * a)) / 32768.0 ** 2 + 1e-10)
        # noise floor: follows quiet frames down at once, creeps up ~1 dB/s through speech
        self.floor = db if db < self.floor else self.floor + 0.02
        loud = db > max(self.floor + GATE_OPEN_DB, GATE_MIN_DB)
        pcm16 = discord_to_16k_mono(data)
        self.recent.append(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0)
        # closed: open only for a voice; open: any loud frame keeps it open
        speech = loud and (self.hang > 0 or self._voice(pcm16))
        if speech:
            out = [*self.pre, data] if self.hang == 0 else [data]
            self.pre.clear()
            self.hang = GATE_HANG
            return out, False
        if self.hang > 0:
            self.hang -= 1
            return [data], False
        self.pre.append(data)
        return [], loud


class FxVoiceClient:
    def __init__(self, guild: "FxGuild", channel: "FxChannel", on_lost: Callable[["FxVoiceClient"], None]):
        self.guild = guild
        self.channel = channel
        self.room = rtc.Room()
        self._on_lost = on_lost
        self._connected = False
        self._closing = False
        self._sink = None
        self._after_listen = None
        self._readers: dict[str, asyncio.Task] = {}
        self._frames: queue.Queue = queue.Queue(maxsize=500)  # ~10 s of one speaker: drops beat unbounded lag
        self._recv_thread: threading.Thread | None = None
        self._source: rtc.AudioSource | None = None
        self._play_task: asyncio.Task | None = None
        self._playing = None
        self.loop = asyncio.get_running_loop()
        self.gate: dict[int, list[int]] = {}  # user id -> [frames let through, noisy frames held back]
        self._report_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ connection

    async def connect(self, endpoint: str, token: str) -> None:
        self.room.on("track_subscribed", self._track_subscribed)
        self.room.on("track_unsubscribed", lambda track, pub, p: self._stop_reader(pub.sid))
        self.room.on("participant_disconnected", self._participant_left)
        self.room.on("disconnected", self._disconnected)
        await self.room.connect(endpoint, token, rtc.RoomOptions(auto_subscribe=True))
        self._connected = True
        self._report_task = asyncio.create_task(self._gate_report(), name="fluxer-gate-report")
        self._source = rtc.AudioSource(DISCORD_SR, CHANNELS, queue_size_ms=100)
        track = rtc.LocalAudioTrack.create_audio_track("static", self._source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        log.info("LiveKit room %s joined (%d others here)", self.room.name, len(self.room.remote_participants))

    def _disconnected(self, reason=None) -> None:
        was = self._connected
        self._connected = False
        if was and not self._closing:
            log.warning("Lost the Fluxer voice connection in #%s (%s)", self.channel, reason)
            self._on_lost(self)

    def is_connected(self) -> bool:
        return self._connected

    async def _gate_report(self) -> None:
        while True:
            await asyncio.sleep(GATE_REPORT_S)
            noisy = {uid: c for uid, c in self.gate.items() if c[1] >= 50}  # >= 1 s of noise held back
            if noisy:
                parts = []
                for uid, (passed, held) in sorted(noisy.items(), key=lambda x: -x[1][1]):
                    m = self.guild.get_member(uid)
                    parts.append(f"{m.display_name if m else uid} {held * 20 / 1000:.0f}s noise held back "
                                 f"({passed * 20 / 1000:.0f}s speech let through)")
                log.info("🔇 Fluxer noise gate, last %d min: %s", GATE_REPORT_S // 60, "; ".join(parts))
            self.gate.clear()

    async def disconnect(self, force: bool = False) -> None:
        self._closing = True
        if self._report_task:
            self._report_task.cancel()
        self.stop_listening()
        self.stop_playing()
        if self._play_task:
            self._play_task.cancel()
        self._connected = False
        try:
            await self.room.disconnect()
        except Exception as e:  # noqa: BLE001
            log.debug("LiveKit disconnect: %s", e)
        if self._source is not None:
            try:
                await self._source.aclose()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ receive

    def listen(self, sink, after=None) -> None:
        self._sink, self._after_listen = sink, after
        if self._recv_thread is None or not self._recv_thread.is_alive():
            self._recv_thread = threading.Thread(target=self._recv_loop, name="fluxer-recv", daemon=True)
            self._recv_thread.start()
        for p in self.room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track is not None and pub.kind == rtc.TrackKind.KIND_AUDIO:
                    self._track_subscribed(pub.track, pub, p)

    def is_listening(self) -> bool:
        return self._sink is not None and self._recv_thread is not None and self._recv_thread.is_alive()

    def stop_listening(self) -> None:
        self._sink = None
        for sid in list(self._readers):
            self._stop_reader(sid)
        self._frames.put(None)  # wakes the thread so it can exit

    def _track_subscribed(self, track, pub, participant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or self._sink is None or pub.sid in self._readers:
            return
        if pub.source != rtc.TrackSource.SOURCE_MICROPHONE:
            # Screen share with sound: game/stream audio arrived as a second track from the same person and was
            # treated as their voice (game sounds as "speech", cutting the bot off). Only mics are listened to.
            log.info("Ignoring %s's %s audio (only microphones are listened to)", participant.name or participant.identity,
                     "screen-share" if pub.source == rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO else "non-mic")
            return
        uid = uid_of(participant.identity)
        if not uid:
            return
        self._readers[pub.sid] = asyncio.create_task(self._read(track, uid), name=f"fluxer-audio-{uid}")

    def _stop_reader(self, sid: str) -> None:
        task = self._readers.pop(sid, None)
        if task:
            task.cancel()

    def _participant_left(self, participant) -> None:
        for pub in participant.track_publications.values():
            self._stop_reader(pub.sid)

    async def _read(self, track, uid: int) -> None:
        stream = rtc.AudioStream(track, sample_rate=DISCORD_SR, num_channels=CHANNELS, frame_size_ms=20)
        gate = NoiseGate()
        counts = self.gate.setdefault(uid, [0, 0])
        try:
            async for ev in stream:
                data = bytes(ev.frame.data)
                if len(data) != FRAME_BYTES:
                    continue  # the sink expects Discord's 20 ms frames
                out, held = gate.feed(data)
                counts = self.gate.setdefault(uid, counts)
                counts[1] += held
                if not out:
                    continue
                counts[0] += len(out)
                for frame in out:
                    try:
                        self._frames.put_nowait((uid, frame))
                    except queue.Full:
                        pass
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("Reading Fluxer audio from %s failed", uid)
        finally:
            await stream.aclose()

    def _recv_loop(self) -> None:
        """The sink runs here (per-user VAD + resampling), never on the event loop."""
        error = None
        try:
            while True:
                item = self._frames.get()
                sink = self._sink
                if item is None or sink is None:
                    if sink is None:
                        break
                    continue
                uid, pcm = item
                member = self.guild.get_member(uid)
                if member is not None:
                    sink.write(member, _Data(pcm))
        except Exception as e:  # noqa: BLE001
            error = e
            log.exception("Fluxer voice receive failed")
        if self._after_listen:
            self._after_listen(error)

    # ------------------------------------------------------------------ send

    def play(self, source, after=None) -> None:
        self.stop_playing()
        self._playing = source
        self._play_task = asyncio.create_task(self._play(source, after), name="fluxer-playback")

    async def _play(self, source, after) -> None:
        error = None
        try:
            while self._playing is source and self._source is not None:
                frame = source.read()  # cheap: a slice of the TTS buffer (+ gain/clip tap)
                if not frame:
                    break
                await self._source.capture_frame(rtc.AudioFrame(frame, DISCORD_SR, CHANNELS, FRAME_SAMPLES))
            if self._source is not None and self._playing is source:
                await self._source.wait_for_playout()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            error = e
        finally:
            if self._playing is source:
                self._playing = None
            if after:
                after(error)

    def is_playing(self) -> bool:
        return self._playing is not None

    def stop_playing(self) -> None:
        self._playing = None
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
        if self._source is not None:
            self._source.clear_queue()

    stop = stop_playing  # nothing to trip over here: receiving is separate

    async def move_to(self, channel) -> None:
        raise NotImplementedError("rejoin instead")  # FluxerBot.join_channel handles moves
