"""Text-to-speech backends. synthesize() returns (mono float32 audio, sample_rate) (blocking)."""
from __future__ import annotations

import io
import logging
import os
import time
import wave

import numpy as np

from .audio import int16_bytes_to_float, to_discord_pcm

log = logging.getLogger("voicebot.tts")


def _cuda_session(model_path: str):
    """ONNX Runtime session on the GPU, tuned for variable-length TTS inputs. None if CUDA is unavailable."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):  # ORT >= 1.21: load pip-installed CUDA/cuDNN libs
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            log.debug("ort.preload_dlls failed: %s", e)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        log.warning("onnxruntime has no CUDA provider (is onnxruntime-gpu installed?) - TTS will run on CPU")
        return None

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cuda_opts = {
        "device_id": 0,
        # The default (EXHAUSTIVE) re-benchmarks every conv for each new input length, which stalls
        # the first sentence of every new size. HEURISTIC picks an algorithm instantly.
        "cudnn_conv_algo_search": "HEURISTIC",
        # Grow the memory arena only as needed - VRAM is shared with Whisper and the LLM.
        "arena_extend_strategy": "kSameAsRequested",
    }
    try:
        return ort.InferenceSession(
            model_path, sess_options=so,
            providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not create CUDA session for %s (%s) - TTS will run on CPU", model_path, e)
        return None


class KokoroTTS:
    """Local Kokoro-82M via kokoro-onnx. Tens of ms per sentence on GPU, faster than real time on CPU."""

    def __init__(self, c):
        from kokoro_onnx import Kokoro

        for p in (c.model_path, c.voices_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Kokoro file missing: {p} (run: python download_models.py)")

        self.kokoro = None
        if c.use_cuda:
            session = _cuda_session(c.model_path)
            if session is not None:
                try:
                    self.kokoro = Kokoro.from_session(session, c.voices_path)
                except (AttributeError, TypeError) as e:
                    # older kokoro-onnx without from_session: let it build its own session on CUDA
                    log.debug("Kokoro.from_session unavailable (%s)", e)
                    os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
        if self.kokoro is None:
            self.kokoro = Kokoro(c.model_path, c.voices_path)

        self.voice, self.speed, self.lang = c.voice, float(c.speed), c.lang
        sess = getattr(self.kokoro, "sess", None)
        device = "GPU" if sess is not None and sess.get_providers()[0] == "CUDAExecutionProvider" else "CPU"
        self.desc = f"Kokoro · {self.voice} · {device}"
        log.info("Kokoro TTS loaded on %s (voice=%s)", device, self.voice)

    def synthesize(self, text: str):
        samples, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return np.asarray(samples, dtype=np.float32), sr


class PiperTTS:
    """Local Piper. Lowest latency, slightly more robotic."""

    def __init__(self, c):
        from piper import PiperVoice

        if not os.path.exists(c.model_path):
            raise FileNotFoundError(f"Piper voice missing: {c.model_path} (run: python download_models.py piper)")
        self.voice = PiperVoice.load(c.model_path, use_cuda=bool(c.use_cuda))
        self.length_scale = float(c.length_scale)
        log.info("Piper TTS loaded (%s)", c.model_path)

    def synthesize(self, text: str):
        sr = self.voice.config.sample_rate
        if hasattr(self.voice, "synthesize_stream_raw"):  # piper-tts < 1.3
            pcm = b"".join(self.voice.synthesize_stream_raw(text, length_scale=self.length_scale))
        else:  # piper-tts >= 1.3
            from piper import SynthesisConfig

            chunks = list(self.voice.synthesize(text, syn_config=SynthesisConfig(length_scale=self.length_scale)))
            pcm = b"".join(ch.audio_int16_bytes for ch in chunks)
            if chunks:
                sr = chunks[0].sample_rate
        return int16_bytes_to_float(pcm), sr


class OpenAICompatTTS:
    """Any OpenAI-compatible /v1/audio/speech server (Kokoro-FastAPI, openedai-speech, AllTalk, ...)."""

    def __init__(self, c):
        import httpx

        self.c = c
        self.http = httpx.Client(
            base_url=c.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {c.api_key or 'not-needed'}"},
            timeout=float(c.timeout),
        )
        log.info("Remote TTS: %s/%s @ %s", c.model, c.voice, c.base_url)

    def synthesize(self, text: str):
        fmt = self.c.response_format
        r = self.http.post("/audio/speech", json={
            "model": self.c.model, "voice": self.c.voice, "input": text,
            "response_format": fmt, "speed": float(self.c.speed),
        })
        r.raise_for_status()
        if fmt == "wav":
            with wave.open(io.BytesIO(r.content), "rb") as w:
                sr, ch = w.getframerate(), w.getnchannels()
                audio = int16_bytes_to_float(w.readframes(w.getnframes()))
            if ch > 1:
                audio = audio.reshape(-1, ch).mean(axis=1)
            return audio, sr
        return int16_bytes_to_float(r.content), int(self.c.sample_rate)


class TTS:
    def __init__(self, cfg):
        backend = cfg.tts.backend
        self.volume = float(cfg.tts.volume)
        self.engine = {"kokoro": KokoroTTS, "piper": PiperTTS, "openai": OpenAICompatTTS}[backend](cfg.tts[backend])
        self.desc = getattr(self.engine, "desc", backend)
        self.calls, self.chars, self.busy_s, self.audio_s = 0, 0, 0.0, 0.0  # for /stats

    def synthesize_discord(self, text: str) -> bytes:
        """Returns 48kHz stereo s16le PCM ready for Discord."""
        t0 = time.perf_counter()
        audio, sr = self.engine.synthesize(text)
        pcm = to_discord_pcm(audio, sr, self.volume)
        self.calls += 1
        self.chars += len(text)
        self.busy_s += time.perf_counter() - t0
        self.audio_s += len(pcm) / (48000 * 2 * 2)
        return pcm

    def warmup(self) -> None:
        # A few different lengths so GPU kernels/cuDNN plans for common sizes are ready.
        for text in ("Hi.", "Warming up the voice.", "This is a slightly longer sentence to warm up the speech model."):
            self.synthesize_discord(text)
        self.calls, self.chars, self.busy_s, self.audio_s = 0, 0, 0.0, 0.0
