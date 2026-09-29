"""Streaming text helpers: strip <think> blocks and *actions*, chunk into sentences, clean text for
TTS, and conversational heuristics (end-of-turn, backchannels, speaker labels)."""
from __future__ import annotations

import re
from datetime import datetime


def _partial_suffix(s: str, tag: str) -> int:
    """Length of the longest suffix of s that is a prefix of tag."""
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:n]):
            return n
    return 0


class SpanFilter:
    """Removes open...close spans from a token stream, correctly handling tags split across tokens."""

    def __init__(self, open_tag: str, close_tag: str):
        self.open, self.close = open_tag, close_tag
        self.buf = ""
        self.inside = False
        self.inner = ""  # text of the open span so far
        self._ctx = ""  # the last bit of text let through, so keep() can see what comes before a span

    def keep(self, inner: str, before: str) -> bool:
        """Whether a finished span's text should be kept after all (only its markers removed)."""
        return False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while True:
            if self.inside:
                i = self.buf.find(self.close)
                if i < 0:
                    keep = _partial_suffix(self.buf, self.close)
                    self.inner += self.buf[: len(self.buf) - keep]
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                inner, self.inner = self.inner + self.buf[:i], ""
                if self.keep(inner, self._ctx + "".join(out)):
                    out.append(inner)
                self.buf = self.buf[i + len(self.close):]
                self.inside = False
            else:
                i = self.buf.find(self.open)
                if i >= 0:
                    out.append(self.buf[:i])
                    self.buf = self.buf[i + len(self.open):]
                    self.inside = True
                    continue
                hold = _partial_suffix(self.buf, self.open)
                out.append(self.buf[: len(self.buf) - hold])
                self.buf = self.buf[len(self.buf) - hold:]
                break
        result = "".join(out)
        self._ctx = (self._ctx + result)[-40:]
        return result

    def flush(self) -> str:
        rest = "" if self.inside else self.buf
        self.buf = self.inner = self._ctx = ""
        self.inside = False
        return rest


class ThinkFilter(SpanFilter):
    """Removes <think>...</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)."""

    def __init__(self):
        super().__init__("<think>", "</think>")


_ACTION_WORDS = {
    "sighs", "sigh", "laughs", "laugh", "grins", "grin", "smirks", "smirk", "shrugs", "shrug", "chuckles",
    "chuckle", "giggles", "giggle", "winks", "wink", "nods", "nod", "leans", "lean", "yawns", "yawn", "gasps",
    "gasp", "snorts", "snort", "scoffs", "scoff", "facepalms", "coughs", "cough", "smiles", "smile", "blinks",
    "pauses", "pause", "whispers", "rolls", "clears", "stretches", "groans", "groan", "huffs", "hums", "claps",
    "waves", "wave", "sips", "munches", "stares", "glares", "raises", "tilts", "crosses", "snickers", "cackles",
    "scratches", "shakes", "taps", "adjusts", "fidgets", "beams", "pouts", "rubs", "cracks", "sniffs", "blushes",
    "gestures", "points", "looks", "turns", "slams", "throws", "puts", "holds", "grabs", "sits", "stands",
}
_NAME_GLUE = {"of", "the", "and", "a", "an", "in", "on", "to", "for", "de", "von", "&", "-"}


class ActionFilter(SpanFilter):
    """Removes roleplay stage directions like *leans back and grins* so TTS doesn't read them out.
    Markdown **bold** survives: the empty ** spans are removed and the inner text is kept. So do names and
    emphasis in asterisks (*Purrchena*, *Dreamscape of Wind and Snow*, *so*), which would leave a hole."""

    def __init__(self):
        super().__init__("*", "*")

    def keep(self, inner: str, before: str) -> bool:
        words = inner.split()
        if not words or words[0].lower().strip(",.!?") in _ACTION_WORDS:
            return False
        # Inside a sentence ("your whole routine is just *and you get cut off*?") it's a quote or emphasis:
        # cutting it would leave a hole. Stage directions sit between sentences, or start with an action word.
        prev = before.rstrip()
        if prev and prev[-1] not in ".!?:;\n" and len(words) <= 14:
            return True  # (the first word isn't an action word - checked above)
        if len(words) > 6:
            return False
        if len(words) == 1 and words[0].islower():  # emphasis: "that's *so* bad" (not "*sighs*", "*smiling*")
            w = words[0].strip(",.!?")
            return not w.endswith(("s", "ing", "ed"))
        return words[0][0].isupper() and all(w[0].isupper() or w[0].isdigit() or w.lower() in _NAME_GLUE
                                             for w in words)


_CAPS = re.compile(r"[A-Z][\w'’-]*(?:(?::[ \t]*|[ \t]+)[A-Z][\w'’-]*)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NOT_NAMES = {"I", "I'm", "I'll", "I've", "I'd", "OK", "Okay", "Oh", "Yeah", "Nah", "Hey"}


def name_hints(text: str) -> list[str]:
    """Capitalised names in a reply ("Purrchena", "Arknights: Endfield"), for Whisper's spelling hint.
    A capital at the start of a sentence only counts when it's part of a multi-word name."""
    out = []
    for sent in _SENTENCE.split(text):
        start = len(sent) - len(sent.lstrip(" \t\"'“‘(-"))
        for m in _CAPS.finditer(sent):
            name = re.sub(r"['’]s$", "", m.group().strip(" :-'’")).strip(" :-'’")
            if name in _NOT_NAMES or (m.start() == start and " " not in name):
                continue
            out.append(name)
    return out


_CONTINUE = re.compile(
    r"(?:(?:oh|ok(?:ay)?|sorry|no|nah|wait|yeah|alright|my bad)[,.!]?\s+)*(?:(?:hey\s+)?\w+[,.!]?\s+)??"
    r"(?:go on|keep going|continue|carry on|go ahead|proceed|you can continue|please continue|"
    r"finish (?:it|that|what you were saying|your (?:thought|sentence|point))|"
    r"what were you (?:saying|gonna say|going to say)|you were saying|as you were saying)"
    r"(?:\s+(?:please|then|now))*(?:,?\s*\w+)?[\s.!?]*", re.I)


def continue_request(text: str) -> bool:
    """A sentence asking the bot to carry on after being cut off ("Sorry, go on", "Static, keep going",
    "What were you saying?"). Whole sentences only."""
    return any(_CONTINUE.fullmatch(s.strip()) for s in _SENTENCE.split(text) if s.strip())


_LEAVE_TAIL = (
    r"(?:(?:you\s+can|you\s+may|you\s+should|go\s+ahead\s+and|please|just|now|time\s+to|can\s+you|could\s+you)\s+)*"
    r"(?:leave|disconnect|hop\s+off)"
    r"(?:\s+(?:now|the\s+(?:call|channel|vc|voice\s+(?:chat|channel))|please))*"
    r"(?:,?\s*%s)?[\s.!?]*")
_LEAVE_HEAD = r"(?:(?:ok(?:ay)?|alright|all right|cool|thanks?|thank you|well|so|yeah|anyway)[,.!]?\s+)*"


# Blunter ways to send it away - only with its name ("get out" alone is often said to a friend mid-game).
_BEGONE = r"(?:get\s+(?:the\s+(?:fuck|hell)\s+)?out(?:\s+of\s+here)?|go\s+away|beat\s+it|get\s+lost|piss\s+off|scram|bounce)"


# "how about you do me a favor and leave?", "can you leave", "why don't you just leave already" - a sentence ending
# in "leave" that isn't about someone else leaving ("I have to leave", "we should leave"). Only used for lines
# that are already talking to the bot.
_ASK_LEAVE = re.compile(r"(?<!\bi )(?<!\bwe )(?<!\bto )(?<!\bgotta )(?<!\bgonna )(?<!\bshould )\bleave"
                        r"(?:\s+(?:now|already|please|the\s+(?:call|channel|vc)))*[\s.!?]*$", re.I)


def leave_request(text: str, names=()) -> bool:
    """A sentence that tells the bot to leave the call ("You can leave now.", "Static leave", "Static, disconnect",
    "Static, get the fuck out of here"). Whole sentences only, so "I have to leave soon" or "leave it" don't count.
    `names` = what the bot is called (its name / wake words), which may come before or after with or without a comma."""
    alt = "|".join(re.escape(n) for n in names if n) or r"(?!x)x"
    pattern = re.compile(_LEAVE_HEAD + r"(?:(?:hey\s+)?(?:%s)[,.!]?\s*)?" % alt + _LEAVE_TAIL % f"(?:{alt})", re.I)
    begone = re.compile(_LEAVE_HEAD + rf"(?:(?:hey\s+)?(?:{alt})[,.!]?\s+{_BEGONE}|{_BEGONE},?\s+(?:{alt}))[\s.!?]*", re.I)
    return any(pattern.fullmatch(s.strip()) or begone.fullmatch(s.strip()) or _asks_leave(s.strip())
               for s in _SENTENCE.split(text) if s.strip())


def _asks_leave(sentence: str) -> bool:
    low = sentence.lower()
    if re.search(r"\b(?:i|we|i'?m|we'?re|he|she|they|gotta|have to|need to|want to|about to)\b[^.!?]*\bleave", low) \
            and not re.search(r"\b(?:you|u)\b", low):
        return False
    return bool(_ASK_LEAVE.search(sentence)) and bool(re.search(r"\b(?:you|u|just|please|can|could|how about|why don'?t)\b", low))


_BYE = re.compile(r"(?:^|[,.!?]\s*)later\b(?![ \t]+(?:in|on|than|today|tonight|this))|\b(?:bye|goodbye|catch (?:y'?all|you|ya)|peace(?: out)?|i'?m out|i'?ll (?:bounce|dip|head out|go)|"
                  r"bouncing|see (?:y'?all|ya|you))\b", re.I)


def says_goodbye(text: str) -> bool:
    return bool(_BYE.search(text))


_QUIET = re.compile(r"\b(?:shut\s+(?:the\s+(?:fuck|hell)\s+)?up|stfu|be\s+quiet|nobody(?:'s|\s+is|\s+was)\s+talking\s+to\s+you|"
                    r"no\s*(?:one|body)\s+asked(?:\s+you)?|stop\s+(?:talking|responding|answering)|not\s+talking\s+to\s+you|"
                    r"(?:quit|stop)\s+(?:butting|jumping)\s+in)\b", re.I)


def quiet_request(text: str) -> bool:
    """"Shut up", "nobody's talking to you", "stop responding": the bot should stop joining in without its name."""
    return bool(_QUIET.search(text))


_LEADING_LABEL = re.compile(r"^\s*([A-Z][\w'’.-]*(?:[ \t]+[A-Z][\w'’.-]*){0,2})[ \t]*:[ \t]+")
# A stage direction on its own line at the start of a reply: "(Implied owner)\n\nYeah..."
_LEADING_ASIDE = re.compile(r"^\s*\([^()\n]{1,60}\)[ \t]*(?:\n+|$)")
_QUOTED = re.compile(r'["“]([^"”\n]{12,})["”]')


def prompt_examples(system_prompt: str) -> list[str]:
    """Quoted example lines in the persona. Fed to RepeatGuard as already said, so the model can't
    just recite them."""
    return _QUOTED.findall(system_prompt)


class SpeakerGuard:
    """Roleplay/uncensored models often label their own lines ("Static: ...") or keep going and write
    the next line for a human ("Alice: ..."). Strip the former, cut the reply at the latter."""

    def __init__(self, bot_name: str, other_names):
        own = [n for n in {bot_name, "assistant", "AI", "bot"} if n]
        bot_low = bot_name.lower()
        others = {n for n in other_names if n and n.lower() != bot_low} | {"user", "human"}
        self._own = re.compile(r"^[ \t]*(?:%s)[ \t]*:[ \t]*" % "|".join(map(re.escape, own)), re.I | re.M)
        alt = "|".join(map(re.escape, sorted(others, key=len, reverse=True)))
        self._other_start = re.compile(r"^\s*(?:%s)\s*:" % alt, re.I)
        self._other_line = re.compile(r"\n[ \t]*(?:%s)[ \t]*:" % alt, re.I)
        # ...or on the same line, after a sentence: "What's up, guys? riley: Hey Static." (spoken aloud before)
        self._other_mid = re.compile(r"(?<=[.!?…])[ \t]+(?:%s)[ \t]*:" % alt, re.I)
        # Usernames are lowercase, which _LEADING_LABEL (capitalised names) doesn't catch.
        known = "|".join(map(re.escape, sorted(set(own) | others, key=len, reverse=True)))
        self._known_label = re.compile(r"^\s*(?:%s)[ \t]*:[ \t]*" % known, re.I)
        # Talking to itself: "Yo, Static!" / "Hey Static, you good?" / "Nice one, Static." in answer to a line that
        # used its name - people noticed ("why is it talking in third person?"). Only the name used as a greeting or
        # address goes; "I'm Static" and "Static's the name" stay.
        me = re.escape(bot_name)
        self._self_open = re.compile(
            rf"(^|(?<=[.!?])\s+)((?:hey|yo|hi|hello|sup|oh|ayy?e?|what'?s up)\b[ \t]*,?[ \t]*)?{me}[ \t]*[,!.][ \t]*",
            re.I)
        self._self_close = re.compile(rf",[ \t]*{me}\b[ \t]*(?=[.!?]|$)", re.I)

    def strip_own(self, text: str) -> str:
        text = self._own.sub("", text)
        fixed = self._self_open.sub(self._self_greeting, self._self_close.sub("", text))
        if fixed != text:  # a sentence may now start lowercase ("Static, you there?" -> "you there?")
            fixed = re.sub(r"(^|[.!?][ \t]+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), fixed)
        return fixed

    @staticmethod
    def _self_greeting(m: re.Match) -> str:
        """Keep the greeting word, drop its own name: "Yo, Static!" -> "Yo!" / "Static, you there?" -> "You there?"."""
        lead, greet = m.group(1), (m.group(2) or "").strip().rstrip(",")
        return f"{lead}{greet.capitalize()}! " if greet else lead

    def strip_label(self, text: str) -> str:
        """Drop a 'Name:' label at the very start of a reply. Whatever name it is (its own, the person
        it's answering, one copied from history), the line is still the model's own - only later
        lines labelled with someone else's name are it writing their part."""
        stripped = self._known_label.sub("", text, count=1)
        if stripped == text:
            stripped = _LEADING_LABEL.sub("", text, count=1)
        return _LEADING_ASIDE.sub("", stripped, count=1)

    def is_other(self, chunk: str) -> bool:
        return bool(self._other_start.match(chunk))

    def split_other(self, chunk: str) -> tuple[str, bool]:
        """(the part before someone else's "Name:" in the middle of a chunk, whether there was one)."""
        m = self._other_mid.search(chunk)
        return (chunk[: m.start()].strip(), True) if m else (chunk, False)

    def clean(self, text: str) -> str:
        """For complete replies: cut at the first line spoken 'as' someone else, strip own labels."""
        s = "\n" + self.strip_label(text.lstrip())
        m = self._other_line.search(s) or self._other_mid.search(s)
        if m:
            s = s[: m.start()]
        return self.strip_own(s.lstrip("\n")).strip()


_WORDS = re.compile(r"[a-z']+")


def _norm_words(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


class EchoGuard:
    """Small models sometimes write the chat out like a script: they first copy the message they're answering
    ("alice: you're great") and only then give their own line ("Static: thanks"). The label filters strip
    both names, which would leave their words in the reply as if the bot said them. This drops that copy."""

    def __init__(self, said: list[str]):
        self.target = _norm_words(" ".join(said))
        self.pos = 0
        self.active = bool(self.target)

    def is_echo(self, piece: str) -> bool:
        """For the start of a reply, piece by piece (labels already stripped): True while it's still copying."""
        if not self.active:
            return False
        p, t = _norm_words(piece), self.target
        end = self.pos + len(p)
        if p and t.startswith(p, self.pos) and (end == len(t) or t[end] == " "):
            self.pos = end + 1
            return True
        self.active = False
        return False

    def strip(self, text: str, label=None) -> str:
        """A whole reply without the copied lines at its start. `label` strips a "Name:" label from a line."""
        lines = text.split("\n")
        while lines:
            line = lines[0].strip()
            if not line:
                lines.pop(0)
                continue
            bare = label(line) if label else line
            sentences = split_sentences(bare) or [bare]
            if not all(self.is_echo(x) for x in sentences):
                break
            lines.pop(0)
        return "\n".join(lines).strip()


class RepeatGuard:
    """Small models loop: they reuse their own openers, jokes and whole sentences from earlier turns
    (sampler penalties only look at the last few dozen tokens, so they don't catch it). Flags a new
    sentence when it's the same as, or mostly made of word triples from, something said recently."""

    def __init__(self, recent_replies, overlap: float = 0.6):
        self.overlap = overlap
        self.sentences: set[str] = set()
        self.grams: set[tuple] = set()
        for reply in recent_replies:
            for s in split_sentences(reply):
                self.add(s)

    @staticmethod
    def _words(text: str) -> list[str]:
        return _WORDS.findall(text.lower().replace("’", "'"))

    def add(self, sentence: str) -> None:
        w = self._words(sentence)
        self.sentences.add(" ".join(w))
        self.grams.update(zip(w, w[1:], w[2:]))

    def is_repeat(self, sentence: str) -> bool:
        w = self._words(sentence)
        if not w:
            return False
        if " ".join(w) in self.sentences:
            return True
        grams = list(zip(w, w[1:], w[2:]))
        return len(grams) >= 3 and sum(g in self.grams for g in grams) / len(grams) >= self.overlap

    def filter(self, text: str) -> tuple[str, list[str]]:
        """For complete replies: (text without repeated sentences, the dropped ones)."""
        kept, dropped = [], []
        for s in split_sentences(text):
            if self.is_repeat(s):
                dropped.append(s)
            else:
                kept.append(s)
                self.add(s)
        return (text if not dropped else " ".join(kept)), dropped


def recent_replies(history: list[dict], n: int = 6) -> list[str]:
    return [m["content"] for m in history if m["role"] == "assistant"][-n:]


_CONTINUATION_WORDS = {
    "and", "but", "or", "so", "because", "cause", "like", "um", "uh", "umm", "uhh", "er", "the", "a", "an",
    "to", "of", "with", "if", "when", "that", "which", "who", "is", "was", "are", "my", "your", "then",
    "about", "for", "in", "on", "at", "just", "i", "i'm", "we", "you", "it's", "kinda", "maybe",
}
_TERMINAL = re.compile(r"[.!?][\"')\]]*\s*$")
_TRAILING_OFF = re.compile(r"(?:\.\.\.|…|[,;:\-—])\s*$")


def looks_finished(text: str) -> bool | None:
    """Rough end-of-turn guess from a transcript: True = sounds complete, False = mid-thought,
    None = can't tell. Used to end turns sooner, or to wait longer before replying."""
    t = text.strip()
    if not t:
        return None
    if _TRAILING_OFF.search(t):
        return False
    words = _WORDS.findall(t.lower())
    if words and words[-1] in _CONTINUATION_WORDS:
        return False
    return True if _TERMINAL.search(t) else None


_BACKCHANNEL_WORDS = {
    "yeah", "yea", "ya", "yep", "yup", "yes", "mhm", "mm", "mmm", "hmm", "mmhm", "uh", "huh", "um", "right",
    "okay", "ok", "sure", "true", "nice", "wow", "ha", "haha", "hahaha", "lol", "oh", "ah", "cool", "exactly",
    "totally", "gotcha", "damn", "word", "facts", "no", "way", "really",
}


def is_backchannel(text: str) -> bool:
    """'yeah', 'mhm', 'uh huh', 'oh nice' - listener noises that shouldn't get their own reply."""
    words = _WORDS.findall(text.lower())
    return 0 < len(words) <= 3 and all(w in _BACKCHANNEL_WORDS for w in words)


_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|\n+")
_SOFT_BREAKS = (", ", "; ", ": ", " - ", " — ")
_SOFT_BREAK_RE = re.compile(r"[,;:]\s+| [-—] ")


class SentenceChunker:
    """Accumulates streamed text and emits speakable chunks as soon as a sentence ends.

    The first chunk may also end at a clause break (comma etc.) once it has first_soft_min chars,
    so audio starts before the whole first sentence has been generated.
    """

    def __init__(self, min_chars: int = 8, soft_max: int = 160, first_soft_min: int = 20):
        self.buf = ""
        self.min_chars = min_chars
        self.soft_max = soft_max
        self.first_soft_min = first_soft_min
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = None
            for m in _BOUNDARY.finditer(self.buf):
                if m.end() >= self.min_chars:
                    cut = m.end()
                    break
            if cut is None and not self.emitted and self.first_soft_min:
                m = _SOFT_BREAK_RE.search(self.buf, self.first_soft_min)
                if m:
                    cut = m.end()
            if cut is None and len(self.buf) > self.soft_max:
                idx = max(self.buf.rfind(p, 0, self.soft_max) for p in _SOFT_BREAKS)
                if idx <= 0:
                    idx = self.buf.rfind(" ", 0, self.soft_max)
                if idx > 0:
                    cut = idx + 1
            if cut is None:
                break
            chunk = self.buf[:cut].strip()
            self.buf = self.buf[cut:]
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        chunk = self.buf.strip()
        self.buf = ""
        return [chunk] if chunk else []


def split_sentences(text: str) -> list[str]:
    ch = SentenceChunker()
    return ch.feed(text + " ") + ch.flush()


_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_DISCORD_MENTION = re.compile(r"<[@#][!&]?\d+>|<a?:(\w+):\d+>")
_MD_CHARS = re.compile(r"[*_`#>~|]+")
_WS = re.compile(r"\s+")
_HAS_WORD = re.compile(r"\w")


# "(Wait for a beat)", "(laughs)", "(rolls eyes)": stage directions in brackets are never meant to be said.
_STAGE = re.compile(r"\((?:[\w' ]{0,20}\b)?(?:beat|pause[sd]?|wait|laugh(?:s|ing)?|sigh(?:s|ing)?|smirk(?:s|ing)?|"
                    r"grin(?:s|ning)?|shrug(?:s|ging)?|chuckl(?:es?|ing)|giggl(?:es?|ing)|rolls? (?:my |her )?eyes|"
                    r"scoff(?:s|ing)?|snort(?:s|ing)?|clears? (?:my |her )?throat|dramatic(?:ally)?)\b[\w' ]{0,20}\)", re.I)


def strip_stage(text: str) -> str:
    """'(a slight smirk) You know...' -> 'You know...'. clean_for_speech already keeps these out of the audio; the
    voice reply strips them before history too, so the model doesn't learn to keep writing them."""
    return _WS.sub(" ", _STAGE.sub(" ", text)).strip()


def clean_for_speech(text: str) -> str:
    text = _STAGE.sub(" ", text)
    text = _CODE_BLOCK.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" a link ", text)
    text = _DISCORD_MENTION.sub(lambda m: m.group(1) or "", text)
    text = _MD_CHARS.sub("", text)
    text = _EMOJI.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text if _HAS_WORD.search(text) else ""


def trim_history(history: list[dict], max_messages: int) -> None:
    """Drop old messages in chunks (down to 75%) instead of one per turn. Dropping one message every
    turn shifts the whole prompt, so the LLM server's prefix cache never hits and every reply
    re-reads the full history; trimming in chunks keeps the prefix stable most turns."""
    if len(history) > max_messages:
        del history[: len(history) - int(max_messages * 0.75)]


def split_message(text: str, limit: int = 1990) -> list[str]:
    """Split text to fit Discord's 2000-char message limit."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def now_note() -> str:
    """The current date/time for the newest message of a request (never the system prompt: that has to
    stay identical between turns for the LLM server's prompt cache)."""
    now = datetime.now().astimezone()
    return f"[Right now it's {now:%A, %B} {now.day}, {now:%Y}, {now.hour % 12 or 12}:{now:%M %p %Z}]"
