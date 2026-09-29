"""'Static, clip that': the last N seconds of the voice channel - everyone plus the bot - as an MP3 in the chat.

A ring buffer holds a mono 48kHz mix of the last `clips.buffer_s` seconds, in RAM only (nothing touches disk
unless someone clips). Audio is placed on a 20ms slot timeline by arrival time. Discord only sends packets
while someone talks and delivers them with jitter, so each source continues from the slot after its previous
packet as long as that's close to real time - otherwise speech would stutter or overlap itself.
"""
from __future__ import annotations

import re
import threading

import numpy as np

from .audio import DISCORD_SR, FRAME_MS
from .reminders import parse_duration

SLOTS_PER_S = 1000 // FRAME_MS
SLOT_SAMPLES = DISCORD_SR * FRAME_MS // 1000  # 960 mono samples

_CLIP_RE = re.compile(r"\b(?:clip (?:that|it|this)|clip the last|make (?:a|that a) clip)\b", re.I)
_BARE_RE = re.compile(r"\W*(?:\w+\W+)?clip\W*", re.I)  # just "clip!" or "Static, clip."
_LAST_RE = re.compile(r"\blast ((?:\d+|a|an|one|two|three|half a?)?\s*(?:seconds?|secs?|minutes?|mins?))", re.I)
_WORDS = {"one": "1", "two": "2", "three": "3", "half a": "30 seconds", "half": "30 seconds"}


def clip_request(text: str, max_s: float, default_s: float) -> float | None:
    """'clip that' -> default_s; 'clip the last 10 seconds' -> 10; not a clip request -> None."""
    if not (_CLIP_RE.search(text or "") or _BARE_RE.fullmatch(text or "")):
        return None
    m = _LAST_RE.search(text)
    if not m:
        return default_s
    phrase = m.group(1).strip().lower()
    for word, digits in _WORDS.items():
        phrase = re.sub(rf"^{word}\b", digits, phrase)
    if phrase.startswith(("sec", "min")):
        phrase = f"1 {phrase}"
    if "30 seconds" in phrase:
        phrase = "30 seconds"
    seconds = parse_duration(phrase, default_s) or default_s
    return min(max(round(seconds), 3.0), max_s)


class ClipRecorder:
    """Thread-safe: fed from voice-recv's decoder thread (people) and the player thread (the bot)."""

    def __init__(self, seconds: float):
        self.size = int(seconds * SLOTS_PER_S)
        self.buf = np.zeros((self.size, SLOT_SAMPLES), dtype=np.int32)
        self.ids = np.full(self.size, -1, dtype=np.int64)
        self.who: list[set[int]] = [set() for _ in range(self.size)]
        self._next: dict[int, int] = {}
        self._lock = threading.Lock()

    def add(self, source: int, pcm48_stereo: bytes, now: float) -> None:
        a = np.frombuffer(pcm48_stereo, dtype=np.int16)
        if a.size < 2:
            return
        mono = a[: a.size - a.size % 2].reshape(-1, 2).astype(np.int32).sum(axis=1) // 2
        if mono.size != SLOT_SAMPLES:
            mono = np.resize(mono, SLOT_SAMPLES) if mono.size > SLOT_SAMPLES else np.pad(mono, (0, SLOT_SAMPLES - mono.size))
        slot_now = int(now * SLOTS_PER_S)
        silent = not mono.any()
        with self._lock:
            nxt = self._next.get(source)
            slot = nxt if nxt is not None and slot_now - 10 <= nxt <= slot_now + 25 else slot_now
            self._next[source] = slot + 1
            if silent:
                return
            i = slot % self.size
            if self.ids[i] != slot:
                self.buf[i] = 0
                self.ids[i] = slot
                self.who[i] = set()
            self.buf[i] += mono
            self.who[i].add(source)

    def clip(self, end: float, seconds: float) -> tuple[np.ndarray, set[int]]:
        """int16 mono 48kHz audio for [end - seconds, end] (monotonic clock), and who's audible in it.
        Silence before the first sound is trimmed; the level is normalized."""
        end_slot = int(end * SLOTS_PER_S)
        slots = np.arange(end_slot - int(min(seconds, self.size / SLOTS_PER_S) * SLOTS_PER_S), end_slot)
        idx = slots % self.size
        with self._lock:
            rows = self.buf[idx].copy()
            valid = self.ids[idx] == slots
            who = set().union(*(self.who[i] for i, ok in zip(idx, valid) if ok)) if valid.any() else set()
        rows[~valid] = 0
        audio = rows.reshape(-1)
        nz = np.flatnonzero(audio)
        if nz.size == 0:
            return np.zeros(0, dtype=np.int16), set()
        audio = audio[max(0, nz[0] - DISCORD_SR // 4):]  # keep 250ms before the first sound
        peak = int(np.abs(audio).max())
        gain = min(4.0, 0.9 * 32767 / peak) if peak else 1.0  # quiet mixes up, clipping mixes down
        return np.clip(audio * gain, -32768, 32767).astype(np.int16), who


def encode_mp3(audio: np.ndarray, bitrate: int = 96) -> bytes:
    """Blocking (~0.5s per minute of audio) - run in a thread."""
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate))
    enc.set_in_sample_rate(DISCORD_SR)
    enc.set_channels(1)
    enc.set_quality(2)
    return bytes(enc.encode(audio.tobytes()) + enc.flush())
