"""Speech-to-text backends. transcribe() takes 16kHz mono s16le PCM and returns text (blocking).
`hint` is optional context text (names in the conversation) that nudges Whisper's spelling."""
from __future__ import annotations

import io
import logging
import os
import re
import time
import wave

import numpy as np

from .audio import STT_SR, int16_bytes_to_float

log = logging.getLogger("voicebot.stt")

# Whisper likes to invent these on noise. Real short replies ("okay", "bye", "thank you") are kept:
# the VAD + min_speech_ms + no_speech_prob filters already stop most noise from reaching Whisper.
_HALLUCINATIONS = {
    "you", "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subtitles by the amaraorg community", "ill see you next time",
    "see you in the next video", "thank you for watching and ill see you next time",
}
_NORMALIZE = re.compile(r"[^\w\s]")


def _is_hallucination(text: str) -> bool:
    return _NORMALIZE.sub("", text.lower()).strip() in _HALLUCINATIONS or not re.search(r"\w", text)


def _is_looping(text: str) -> bool:
    """Whisper stuck repeating itself ("Arknights, Arknights, Arknights, ..."), which the name hints can set off
    on unclear audio. A word or short phrase that fills most of a long transcript isn't speech."""
    words = _NORMALIZE.sub(" ", text.lower()).split()
    if len(words) < 8:
        return False  # "no no no no no" is a real thing people say
    for n in (1, 2, 3, 4):
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        top = max(set(grams), key=grams.count)
        if grams.count(top) >= 4 and grams.count(top) * n >= len(words) * 0.5:
            return True
    return False


def _add_nvidia_dll_dirs() -> None:
    """On Windows, make pip-installed CUDA libs (nvidia-cublas-cu12 / nvidia-cudnn-cu12) loadable."""
    if os.name != "nt":
        return
    import importlib.util

    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ModuleNotFoundError, ValueError):
            continue
        for loc in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = os.path.join(loc, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class FasterWhisperSTT:
    def __init__(self, cfg):
        c = cfg.stt.faster_whisper
        self.language = cfg.stt.language or None
        self.beam_size = int(c.beam_size)
        self.initial_prompt = c.initial_prompt or None
        self.no_speech_threshold = float(c.no_speech_threshold)
        self.log_prob_threshold = float(c.log_prob_threshold)
        # Short transcripts ("Thank you.", "What?") are Whisper's favourite inventions on noise, and one of them in
        # the follow-up window gets a reply. large-v3-turbo's no_speech_prob is always ~0 and its confidence on
        # an invented "Thank you." (-0.3) is as good as on real speech, so those get a second opinion from
        # Silero VAD (ships with faster-whisper, ~3ms on CPU): real words peak at 0.75+, hum/hiss/clicks <0.15.
        self.short_words = int(c.get("short_words", 3))
        self.short_speech_prob = float(c.get("short_speech_prob", 0.3))
        self.short_log_prob = float(c.get("short_log_prob_threshold", -0.8))
        self._silero = None

        _add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(c.model, device=c.device, compute_type=c.compute_type,
                                      download_root=c.download_root)
            self.desc = f"faster-whisper {c.model} · {c.device} · {c.compute_type}"
            log.info("faster-whisper '%s' loaded on %s (%s)", c.model, c.device, c.compute_type)
        except Exception as e:  # noqa: BLE001
            if c.device == "cpu":
                raise
            log.warning("Could not load Whisper on %s (%s) - falling back to CPU int8 (slower)", c.device, e)
            self.model = WhisperModel(c.model, device="cpu", compute_type="int8", download_root=c.download_root)
            self.desc = f"faster-whisper {c.model} · cpu · int8 (GPU load failed)"

    def transcribe(self, pcm: bytes, hint: str = "") -> str:
        audio = int16_bytes_to_float(pcm)
        prompt = " ".join(p for p in (self.initial_prompt, hint) if p) or None
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=False,  # already segmented by our VAD
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt,
        )
        kept = [
            s for s in segments
            if s.no_speech_prob < self.no_speech_threshold and s.avg_logprob > self.log_prob_threshold
            and s.compression_ratio <= 2.4  # Whisper's own "this is a repetition loop" signal
        ]
        text = " ".join(s.text.strip() for s in kept).strip()
        if kept and len(_NORMALIZE.sub(" ", text).split()) <= self.short_words:
            voice = self._speech_prob(audio)
            logprob = min(s.avg_logprob for s in kept)
            if voice < self.short_speech_prob or logprob < self.short_log_prob:
                log.info("(dropped a doubtful short transcript: %r, voice %.2f, logprob %.2f)", text, voice, logprob)
                return ""
            log.debug("(short transcript kept: %r, voice %.2f, logprob %.2f)", text, voice, logprob)
        if prompt and _NORMALIZE.sub("", text.lower()).strip() in _NORMALIZE.sub("", prompt.lower()):
            return ""  # echoed the hint back on noise
        if _is_looping(text):
            log.info("(dropped a looping transcript: %.80s…)", text)
            return ""
        return "" if _is_hallucination(text) else text

    def _speech_prob(self, audio: np.ndarray) -> float:
        """Silero VAD's peak speech probability over the clip (0-1)."""
        try:
            if self._silero is None:
                from faster_whisper.vad import get_vad_model
                self._silero = get_vad_model()
            return float(np.max(self._silero(np.pad(audio, (0, (-audio.size) % 512)))))
        except Exception as e:  # noqa: BLE001 - never lose speech over the second opinion
            log.warning("Silero VAD check failed: %s", e)
            return 1.0

    def warmup(self) -> None:
        self.model.transcribe(np.zeros(STT_SR, dtype=np.float32), language=self.language, beam_size=1)
        self._speech_prob(np.zeros(STT_SR, dtype=np.float32))


class OpenAISTT:
    """Any OpenAI-compatible /v1/audio/transcriptions server (speaches, faster-whisper-server, whisper.cpp, ...)."""

    def __init__(self, cfg):
        from openai import OpenAI

        c = cfg.stt.openai
        self.client = OpenAI(base_url=c.base_url, api_key=c.api_key or "not-needed", timeout=float(c.timeout))
        self.model = c.model
        self.language = cfg.stt.language or None
        log.info("Remote STT: %s @ %s", c.model, c.base_url)

    def transcribe(self, pcm: bytes, hint: str = "") -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(STT_SR)
            w.writeframes(pcm)
        kwargs = {"language": self.language} if self.language else {}
        if hint:
            kwargs["prompt"] = hint
        r = self.client.audio.transcriptions.create(
            model=self.model, file=("speech.wav", buf.getvalue(), "audio/wav"), **kwargs
        )
        text = (r.text or "").strip()
        return "" if _is_hallucination(text) else text

    def warmup(self) -> None:
        pass


class TimedSTT:
    """Wraps an STT backend and keeps totals for /stats (calls run on the single STT thread)."""

    def __init__(self, inner):
        self.inner = inner
        self.desc = getattr(inner, "desc", None) or f"openai-compatible · {getattr(inner, 'model', '?')}"
        self.calls, self.busy_s, self.audio_s = 0, 0.0, 0.0

    def transcribe(self, pcm: bytes, hint: str = "") -> str:
        t0 = time.perf_counter()
        try:
            return self.inner.transcribe(pcm, hint)
        finally:
            self.calls += 1
            self.busy_s += time.perf_counter() - t0
            self.audio_s += len(pcm) / (STT_SR * 2)

    def warmup(self) -> None:
        self.inner.warmup()


def create_stt(cfg):
    return TimedSTT(FasterWhisperSTT(cfg) if cfg.stt.backend == "faster_whisper" else OpenAISTT(cfg))
