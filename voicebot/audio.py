"""Audio plumbing: Discord PCM <-> model formats, VAD, and a streaming playback source."""
from __future__ import annotations

import logging
import threading

import discord
import numpy as np
import soxr

log = logging.getLogger("voicebot.audio")

DISCORD_SR = 48000
FRAME_MS = 20
FRAME_BYTES = DISCORD_SR * FRAME_MS // 1000 * 2 * 2  # 20ms, stereo, s16le = 3840
SILENCE_FRAME = b"\x00" * FRAME_BYTES
STT_SR = 16000
STT_FRAME_BYTES = STT_SR * FRAME_MS // 1000 * 2  # 640


def discord_to_16k_mono(pcm: bytes) -> bytes:
    """48kHz stereo s16le -> 16kHz mono s16le. Averages 3 stereo samples per output sample
    (downmix + box-filter decimation), which is plenty for speech and costs almost nothing."""
    a = np.frombuffer(pcm, dtype=np.int16)
    trim = a.size % 6
    if trim:
        a = a[:-trim]
    return (a.reshape(-1, 6).astype(np.int32).sum(axis=1) // 6).astype(np.int16).tobytes()


def to_discord_pcm(audio: np.ndarray, sample_rate: int, volume: float = 1.0) -> bytes:
    """Mono float32 [-1, 1] audio at any rate -> 48kHz stereo s16le for Discord."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""
    if sample_rate != DISCORD_SR:
        audio = soxr.resample(audio, sample_rate, DISCORD_SR, quality="HQ")
    if volume != 1.0:
        audio = audio * volume
    i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return np.repeat(i16, 2).tobytes()  # duplicate each sample -> interleaved L/R


def int16_bytes_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class EnergyVAD:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return bool(a.size) and float(np.sqrt(np.mean(a * a))) > self.threshold


def make_vad(vcfg):
    if vcfg.vad == "webrtc":
        try:
            import webrtcvad

            return webrtcvad.Vad(int(vcfg.vad_aggressiveness))
        except Exception as e:  # noqa: BLE001
            log.warning("webrtcvad unavailable (%s) - falling back to energy VAD", e)
    return EnergyVAD(float(vcfg.energy_threshold))


class StreamingPCMSource(discord.AudioSource):
    """AudioSource fed incrementally while TTS is still generating.

    read() runs on discord.py's player thread every 20ms. While more audio is expected it returns
    silence on underrun (so playback doesn't end between sentences); after finish() it drains and ends.
    """

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0
        self._lock = threading.Lock()
        self._finished = False
        self.gain = 1.0  # set from other threads to duck the voice while someone talks over it
        self.tap = None  # optional callback(frame) seeing exactly what's played (for clips)

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._pos = 0
            self._finished = True

    def read(self) -> bytes:
        with self._lock:
            avail = len(self._buf) - self._pos
            if avail >= FRAME_BYTES:
                frame = bytes(self._buf[self._pos:self._pos + FRAME_BYTES])
                self._pos += FRAME_BYTES
                if self._pos > 1_000_000:  # compact occasionally instead of memmoving every frame
                    del self._buf[:self._pos]
                    self._pos = 0
            elif self._finished:
                if avail <= 0:
                    return b""
                frame = bytes(self._buf[self._pos:]) + b"\x00" * (FRAME_BYTES - avail)
                self._buf.clear()
                self._pos = 0
            else:
                return SILENCE_FRAME
        gain = self.gain
        if gain != 1.0:
            frame = (np.frombuffer(frame, dtype=np.int16) * gain).astype(np.int16).tobytes()
        if self.tap is not None:
            self.tap(frame)
        return frame

    def is_opus(self) -> bool:
        return False
