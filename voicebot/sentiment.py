"""Mood and intent reading, so the bot reacts to how people feel - and learns which reactions work.

Per utterance/message (~55ms on CPU, run alongside the search check so it adds no latency):
  * emotion: a RoBERTa GoEmotions classifier (28 labels, int8 ONNX, CPU) grouped into angry / sad / anxious /
    confused / happy / amused / surprised / curious.
  * tone of voice (voice only): loudness and speaking rate compared with *that person's* usual (a running
    baseline), so someone who's always loud doesn't read as angry. Transcripts lose this completely.
  * intent: question / request / venting / teasing / sharing news - simple rules on the text + emotion.
The result becomes a one-line note on the newest turn ("[Mood: Eric sounds annoyed, louder than usual,
asking a question. Stay calm and get straight to the point.]") - never the system prompt (prompt cache).

Self-tuning (persisted in data/mood.json):
  * strategies: each mood has two ways to respond. Thompson sampling picks one; it's scored on the person's
    next turn (their mood held or improved = win; they talked over the reply or got more upset = loss).
  * per-person calibration: "I'm not mad" / "I'm just kidding" after a mood read raises that person's
    threshold for that emotion, so the bot stops misreading them; confirmed reads slowly lower it back.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("voicebot.mood")

GROUPS = {
    "angry": ("anger", "annoyance", "disgust", "disapproval"),
    "sad": ("sadness", "grief", "disappointment", "remorse"),
    "anxious": ("fear", "nervousness", "embarrassment"),
    "confused": ("confusion",),
    "happy": ("joy", "excitement", "love", "gratitude", "pride", "optimism", "relief", "admiration", "caring"),
    "amused": ("amusement",),
    "surprised": ("surprise", "realization"),
    "curious": ("curiosity",),
}
VALENCE = {"angry": -1.0, "sad": -1.0, "anxious": -0.6, "confused": -0.3, "neutral": 0.0, "curious": 0.1,
           "surprised": 0.2, "amused": 0.8, "happy": 1.0}
WORDING = {"angry": "annoyed", "sad": "down", "anxious": "worried", "confused": "confused", "happy": "happy",
           "amused": "amused", "surprised": "surprised", "curious": "curious"}
# Two ways to respond to each mood; the bandit learns which one works with this crowd.
STRATEGIES = {
    "angry": ("Stay calm and get straight to the point - no jokes.",
              "Briefly acknowledge the frustration, then help."),
    "sad": ("Be warm and gentle, keep it simple.",
            "Be supportive but a little upbeat."),
    "anxious": ("Be calm and reassuring.",
                "Be practical: offer one concrete helpful thing."),
    "confused": ("Explain it more simply, one step at a time.",
                 "Ask which part is confusing."),
    "happy": ("Match their energy.",
              "Be playful and hype them up."),
    "amused": ("Play along with the joke.",
               "Joke back and tease them a little."),
}
DEFAULT_THRESHOLD = 0.5

_QUESTION = re.compile(r"\?\s*$|^\W*(?:\w+,\s*)?(?:who|what|when|where|why|how|is|are|can|could|do|does|did|will|"
                       r"would|should|have|has)\b", re.I)
_REQUEST = re.compile(r"^\W*(?:\w+,?\s+)?(?:can you|could you|would you|please|tell me|play|show me|give me|help|"
                      r"remind|make|set|find|look up|search|explain|say)\b", re.I)
_I = re.compile(r"\b(?:i|i'm|im|i've|my|me|myself)\b", re.I)
_YOU = re.compile(r"\b(?:you|you're|youre|your|ya)\b", re.I)
_MISREAD = re.compile(r"\b(?:i'?m|i am) not (?:mad|angry|upset|sad|annoyed|pissed|crying|scared|worried|nervous|"
                      r"confused|mean)\b|\b(?:i'?m|i am) (?:fine|good|okay|ok|all good)\b|\b(?:just|only) "
                      r"(?:joking|kidding|messing)\b|\b(?:i'?m|i was) (?:joking|kidding)\b|\bnot serious\b", re.I)


@dataclass
class Read:
    user_id: int
    name: str
    group: str                      # emotion group or "neutral"
    score: float                    # 0-1 confidence of that group
    labels: list[tuple[str, float]]  # top raw emotions
    tone: list[str] = field(default_factory=list)  # "louder than usual", ...
    intent: str = ""
    misread: bool = False           # they just said "I'm not mad" etc.
    strategy: int | None = None     # which STRATEGIES entry the reply was told to use
    ms: float = 0.0
    guild_id: int = 0               # set when a reply acted on it
    at: float = 0.0

    @property
    def valence(self) -> float:
        return VALENCE.get(self.group, 0.0) * (0.5 + self.score / 2)

    def describe(self) -> str:
        """'Eric sounds really annoyed, talking louder than usual, asking a question' or '' if nothing to say."""
        bits = []
        if self.group != "neutral":
            how = "really " if self.score >= 0.85 else "a bit " if self.score < 0.6 else ""
            bits.append(f"sounds {how}{WORDING[self.group]}")
        bits += self.tone
        if bits and self.intent:
            bits.append(self.intent)
        return f"{self.name} {', '.join(bits)}" if bits else ""


class EmotionModel:
    """GoEmotions RoBERTa, int8 ONNX on CPU. Loaded once; call from one thread at a time."""

    def __init__(self, cfg):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        c = cfg.sentiment

        def get(f: str) -> str:  # local copy first: no network round-trips on every start
            try:
                return hf_hub_download(c.model_repo, f, cache_dir=c.cache_dir, local_files_only=True)
            except Exception:  # noqa: BLE001 - not downloaded yet
                return hf_hub_download(c.model_repo, f, cache_dir=c.cache_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(c.threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(get(c.model_file), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(get("onnx/tokenizer.json"))
        self.tok.enable_truncation(128)
        id2label = json.loads(Path(get("onnx/config.json")).read_text())["id2label"]
        self.labels = [id2label[str(i)] for i in range(len(id2label))]
        self.index = {name: i for i, name in enumerate(self.labels)}
        self.desc = f"{c.model_repo.split('/')[-1]} · int8 · CPU×{c.threads}"

    def predict(self, text: str) -> np.ndarray:
        e = self.tok.encode(text)
        logits = self.sess.run(None, {"input_ids": np.array([e.ids], dtype=np.int64),
                                      "attention_mask": np.array([e.attention_mask], dtype=np.int64)})[0][0]
        return 1 / (1 + np.exp(-logits))  # multi-label: independent sigmoids


def _voice_level(pcm16k: bytes) -> tuple[float, float] | None:
    """(mean dBFS of voiced 20ms frames, voiced seconds), or None if too short to judge."""
    a = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32)
    n = a.size // 320
    if n < 10:
        return None
    rms = np.sqrt(np.mean(a[: n * 320].reshape(n, 320) ** 2, axis=1)) + 1e-6
    db = 20 * np.log10(rms / 32768)
    voiced = db[db > max(-50.0, float(db.max()) - 30)]
    if voiced.size < 30:  # < 0.6s of actual speech
        return None
    return float(voiced.mean()), voiced.size * 0.02


class MoodReader:
    def __init__(self, bot):
        self.bot = bot
        self.cfg = bot.cfg.sentiment
        self.enabled = bool(self.cfg.enabled)
        self.model: EmotionModel | None = None
        self.path = Path(self.cfg.path)
        self._lock = threading.Lock()
        self._pending: dict[int, Read] = {}     # user id -> the read the last reply to them acted on
        self._dirty = 0
        self.counts: dict[str, int] = {}        # for /status
        self.total_ms = 0.0
        self.reads = 0
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault("users", {})
        self.state.setdefault("arms", {})

    def load(self) -> None:
        """Blocking (downloads ~120 MB on first run) - call from a thread at startup."""
        if not self.enabled:
            return
        try:
            self.model = EmotionModel(self.bot.cfg)
            self.model.predict("warming up")
            log.info("Mood reader loaded (%s)", self.model.desc)
        except Exception as e:  # noqa: BLE001
            self.enabled = False
            log.warning("Mood reader unavailable (%s) - carrying on without it", e)

    # ------------------------------------------------------------------ per-user state

    def _user(self, user_id: int) -> dict:
        return self.state["users"].setdefault(str(user_id), {"th": {}, "base": {}})

    def user_state(self, user_id: int) -> dict | None:
        """A copy of what's kept on this person (dashboard): tone-of-voice baseline + learned mood thresholds."""
        with self._lock:
            u = self.state["users"].get(str(user_id))
            return json.loads(json.dumps(u)) if u else None

    def forget(self, user_id: int) -> None:
        with self._lock:
            self.state["users"].pop(str(user_id), None)
            self._pending.pop(user_id, None)
        self.maybe_save(force=True)

    def threshold(self, user_id: int, group: str) -> float:
        return float(self._user(user_id)["th"].get(group, DEFAULT_THRESHOLD))

    def _tone(self, user_id: int, pcm: bytes | None, words: int, update: bool = True) -> list[str]:
        """Compare this utterance's loudness and pace with the speaker's running baseline, then update it."""
        if not pcm or not self.cfg.prosody:
            return []
        level = _voice_level(pcm)
        if level is None:
            return []
        db, voiced_s = level
        rate = words / voiced_s
        base = self._user(user_id)["base"]
        n = base.get("n", 0)
        out = []
        if n >= int(self.cfg.baseline_min):
            z_db = (db - base["db_m"]) / max(np.sqrt(base["db_v"]), 2.0)
            z_rate = (rate - base["rate_m"]) / max(np.sqrt(base["rate_v"]), 0.3)
            z = float(self.cfg.tone_z)
            if z_db > z:
                out.append("talking louder than usual")
            elif z_db < -z:
                out.append("talking quieter than usual")
            if z_rate > z:
                out.append("talking fast")
            elif z_rate < -z:
                out.append("talking slowly")
        if not update:
            return out
        # running mean/variance: exact average early on, then an exponential moving one
        alpha = max(0.08, 1 / (n + 1))
        for key, x in (("db", db), ("rate", rate)):
            m = base.get(f"{key}_m", x)
            v = base.get(f"{key}_v", 4.0 if key == "db" else 0.5)
            d = x - m
            base[f"{key}_m"] = m + alpha * d
            base[f"{key}_v"] = (1 - alpha) * (v + alpha * d * d)
        base["n"] = n + 1
        self._dirty += 1
        return out

    # ------------------------------------------------------------------ reading

    def read(self, user_id: int, name: str, text: str, pcm: bytes | None = None, to_bot: bool = True,
             observe: bool = True) -> Read | None:
        """Blocking (~55ms) - run in the mood executor. observe=False (speculative replies): judge it, but don't
        update the voice baseline or score the last reply - the confirmed turn will do that once."""
        if self.model is None or not text.strip():
            return None
        t0 = time.perf_counter()
        probs = self.model.predict(text)  # only ever called from the one mood thread: no lock needed
        with self._lock:  # state is shared with the event loop (replied/choose)
            scores = {g: max(float(probs[self.model.index[lab]]) for lab in labs) for g, labs in GROUPS.items()}
            group, score = max(scores.items(), key=lambda kv: kv[1])
            if group == "happy" and max(GROUPS["happy"], key=lambda lab: probs[self.model.index[lab]]) == "gratitude":
                score = 0.0  # "thanks" / "thank you" is politeness, not joy - don't hype them up for it
            if score < self.threshold(user_id, group):
                group, score = "neutral", float(probs[self.model.index["neutral"]])
            top = sorted(((self.model.labels[i], round(float(p), 2)) for i, p in enumerate(probs)),
                         key=lambda x: -x[1])[:3]
            r = Read(user_id, name, group, score, top, self._tone(user_id, pcm, len(text.split()), observe),
                     self._intent(text, group, to_bot), bool(_MISREAD.search(text)))
            r.ms = (time.perf_counter() - t0) * 1000
            if observe:
                self.reads += 1
                self.total_ms += r.ms
                self.counts[group] = self.counts.get(group, 0) + 1
                self._learn_from(r)
        return r

    @staticmethod
    def _intent(text: str, group: str, to_bot: bool) -> str:
        if _REQUEST.search(text):
            return "asking you to do something"
        if _QUESTION.search(text):
            return "asking a question"
        if group == "angry" and to_bot and _YOU.search(text):
            return "annoyed with you"
        if group in ("angry", "sad", "anxious") and _I.search(text):
            return "venting"
        if group == "amused" and to_bot and _YOU.search(text):
            return "teasing you"
        if group == "happy" and _I.search(text):
            return "sharing news"
        return ""

    def choose(self, guild_id: int, reads: list[Read]) -> str | None:
        """The mood note for this turn (None if everyone's neutral), picking a response strategy for the
        strongest non-neutral read."""
        described = [r for r in reads if r.describe()]
        if not described:
            return None
        lines = "; ".join(r.describe() for r in described[:3])
        main = max((r for r in reads if r.group in STRATEGIES), key=lambda r: r.score, default=None)
        advice = ""
        if main is not None and self.cfg.learn:
            with self._lock:
                main.strategy = self._pick(guild_id, main.group)
            advice = " " + STRATEGIES[main.group][main.strategy]
        return f"[Mood: {lines}.{advice} Don't mention this note or label their feelings.]"

    # ------------------------------------------------------------------ learning

    def _arms(self, guild_id: int, group: str) -> list[list[float]]:
        return self.state["arms"].setdefault(str(guild_id), {}).setdefault(group, [[1.0, 1.0], [1.0, 1.0]])

    def _pick(self, guild_id: int, group: str) -> int:
        """Thompson sampling: try both, favour whichever has worked better here."""
        return max(range(2), key=lambda i: random.betavariate(*self._arms(guild_id, group)[i]))

    def replied(self, guild_id: int, reads: list[Read] | None) -> None:
        """A reply that acted on these reads played; judge it on each person's next turn."""
        with self._lock:
            for r in reads or []:
                r.guild_id, r.at = guild_id, time.time()
                self._pending[r.user_id] = r

    def interrupted(self, guild_id: int, reads: list[Read] | None) -> None:
        """They talked over the reply: its strategy didn't land."""
        with self._lock:
            for r in reads or []:
                if r.strategy is not None:
                    self._score(guild_id, r, 0.0, "talked over")
                self._pending.pop(r.user_id, None)

    def _learn_from(self, now: Read) -> None:
        prev = self._pending.pop(now.user_id, None)
        if prev is None or time.time() - prev.at > float(self.cfg.feedback_window_s):
            return
        gid = prev.guild_id
        if now.misread and prev.group != "neutral":
            th = self._user(now.user_id)["th"]
            th[prev.group] = min(0.95, th.get(prev.group, DEFAULT_THRESHOLD) + 0.1)
            log.info("🎭 %s corrected a mood read (%s) → needs more to read them as %s now (%.2f)",
                     now.name, prev.group, prev.group, th[prev.group])
            self._dirty += 10
            self.maybe_save()
            return  # the read was wrong, not the strategy: don't score it
        if prev.group != "neutral":  # read stood uncorrected: ease their threshold back toward the default
            th = self._user(now.user_id)["th"]
            if th.get(prev.group, DEFAULT_THRESHOLD) > DEFAULT_THRESHOLD:
                th[prev.group] = max(DEFAULT_THRESHOLD, th[prev.group] - 0.02)
        if prev.strategy is not None:
            better = now.valence >= prev.valence - 0.05 or now.valence > 0.3
            self._score(gid, prev, 1.0 if better else 0.0, f"{prev.group} → {now.group}")

    def _score(self, guild_id: int, r: Read, reward: float, why: str) -> None:
        arm = self._arms(guild_id, r.group)[r.strategy]
        arm[0] += reward
        arm[1] += 1 - reward
        log.info("🎭 strategy for %s \"%s\" %s (%s) - now %.0f/%.0f", r.group, STRATEGIES[r.group][r.strategy][:40],
                 "worked" if reward else "didn't work", why, arm[0] - 1, arm[0] + arm[1] - 2)
        self._dirty += 10
        self.maybe_save()

    def maybe_save(self, force: bool = False) -> None:
        if self._dirty < 10 and not force:
            return
        self._dirty = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        except OSError as e:
            log.warning("Couldn't save mood state: %s", e)

    def summary(self, guild_id: int) -> list[str]:
        """Human-readable learned state for /tuning and /status."""
        out = []
        for group, arms in sorted(self.state["arms"].get(str(guild_id), {}).items()):
            if sum(a + b - 2 for a, b in arms) == 0:
                continue
            best = max(range(2), key=lambda i: arms[i][0] / (arms[i][0] + arms[i][1]))  # posterior mean
            out.append(f"{group}: " + " | ".join(
                f"{'★' if i == best else ' '}\"{STRATEGIES[group][i]}\" {int(a - 1)}/{int(a + b - 2)}"
                for i, (a, b) in enumerate(arms)))
        return out

    def calibrated(self) -> int:
        """How many people have learned (non-default) thresholds."""
        return sum(1 for u in self.state["users"].values() if any(v != DEFAULT_THRESHOLD for v in u["th"].values()))
