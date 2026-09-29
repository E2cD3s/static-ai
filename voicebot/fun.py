"""Game-night helpers done in Python, never by the model: coin flips, dice, picking someone/something, a random
number, splitting the call into teams. Like calc.py: found with patterns (no LLM call), and handed to the model as
a [Random ...] note for that turn so it announces the real result in its own words instead of inventing one.

"flip a coin", "roll a d20", "roll 2d6+3", "roll three dice", "pick someone", "who goes first", "pick between
tacos, pizza or sushi", "random number between 1 and 50", "split us into two teams". People for "pick someone" /
teams are the humans in the voice call (or whoever the caller passes in).
"""
from __future__ import annotations

import random
import re

_rng = random.SystemRandom()

_NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "a couple": 2, "a pair of": 2}
_NUM_WORD = r"(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten)"


def _n(word: str | None, default: int = 1) -> int:
    if not word:
        return default
    word = word.lower().strip()
    return int(word) if word.isdigit() else _NUMS.get(word, default)


# ---------------------------------------------------------------- coin

_COIN = re.compile(r"\b(?:flip|toss)(?:\s+(?:a|the|us\s+a|me\s+a))?\s+coins?\b|\bheads\s+or\s+tails\b|\b(?:do|give\s+(?:me|us)|let'?s\s+do)\s+a\s+(?:quick\s+)?coin\s+(?:flip|toss)\b", re.I)


def coin(text: str) -> str | None:
    if not _COIN.search(text):
        return None
    return f"flipped a coin: {_rng.choice(['heads', 'tails'])}"


# ---------------------------------------------------------------- dice

_DND = re.compile(r"\b(?P<n>\d+)?\s*d\s*(?P<s>\d{1,3})(?:\s*(?P<sign>[+-])\s*(?P<mod>\d+))?\b", re.I)
_ROLL = re.compile(r"\broll(?:\s+(?:me|us))?\b", re.I)
_DICE = re.compile(rf"\b(?P<n>{_NUM_WORD})?\s*(?:(?P<s>\d+)[- ]sided\s+)?(?:dice|die)\b", re.I)


def dice(text: str) -> str | None:
    if not _ROLL.search(text):
        return None
    n, sides, mod = 1, 6, 0
    if m := _DND.search(text):
        n, sides = int(m["n"] or 1), int(m["s"])
        mod = int(m["mod"] or 0) * (-1 if m["sign"] == "-" else 1)
    elif m := _DICE.search(text):
        n, sides = _n(m["n"], 1 if "die" in m.group(0).lower() else 2), int(m["s"] or 6)
    else:
        return None
    if not (1 <= n <= 20 and 2 <= sides <= 1000):
        return None
    rolls = [_rng.randint(1, sides) for _ in range(n)]
    total = sum(rolls) + mod
    what = f"{n}d{sides}" + (f"{mod:+d}" if mod else "")
    if n == 1 and not mod:
        return f"rolled a d{sides}: {rolls[0]}"
    parts = " + ".join(map(str, rolls)) + (f" {'+' if mod > 0 else '-'} {abs(mod)}" if mod else "")
    return f"rolled {what}: {parts} = {total}"


# ---------------------------------------------------------------- random number

_NUMBER = re.compile(r"\b(?:random|pick\s+a|choose\s+a|give\s+me\s+a)\s+number\s+(?:between|from)\s+(?P<a>-?\d+)\s+(?:and|to|-)\s+(?P<b>-?\d+)", re.I)


def number(text: str) -> str | None:
    if not (m := _NUMBER.search(text)):
        return None
    a, b = sorted((int(m["a"]), int(m["b"])))
    if b - a > 10**9:
        return None
    return f"drew a random number between {a} and {b}: {_rng.randint(a, b)}"


# ---------------------------------------------------------------- pick one

_PICK_PERSON = re.compile(
    r"\b(?:pick|choose|select)\s+(?:a\s+)?(?:random\s+)?(?:someone|somebody|anyone|person|player|one\s+of\s+us)\b"
    r"|\bwho\s+(?:should\s+)?(?:goes|go|starts|start|picks|pick|plays|play)\s+first\b"
    r"|\brandom\s+(?:person|player)\b", re.I)
_PICK_FROM = re.compile(r"\b(?:pick|choose|decide)\s+(?:between|from|one\s+of|for\s+(?:me|us)\s*(?:between|from)?)\s*[:,]?\s*(?P<list>.+)$", re.I)


def _options(raw: str) -> list[str]:
    raw = re.sub(r"[?.!]+$", "", raw.strip())
    parts = re.split(r"\s*,\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+", raw)
    return [p.strip(" \"'") for p in parts if p.strip(" \"'")][:12]


def pick(text: str, people: list[str], speaker: str = "") -> str | None:
    if _PICK_PERSON.search(text):
        if len(people) < 2:
            return "wanted to pick someone, but there's nobody else here to pick from - say so"
        return f"picked at random from {', '.join(people)}: {_rng.choice(people)}"
    if m := _PICK_FROM.search(text):
        opts = _options(m["list"])
        if len(opts) >= 2:
            return f"picked at random from {', '.join(opts)}: {_rng.choice(opts)}"
    return None


# ---------------------------------------------------------------- teams

_TEAMS = re.compile(rf"\b(?:split|divide|break|sort|put)\s+(?:us|everyone|them|people|the\s+group)\s+(?:up\s+)?into\s+(?P<n>{_NUM_WORD})\s+teams?\b"
                    rf"|\b(?:make|pick|random(?:ize)?)\s+(?P<n2>{_NUM_WORD})?\s*(?:random\s+)?teams\b", re.I)


def teams(text: str, people: list[str]) -> str | None:
    if not (m := _TEAMS.search(text)):
        return None
    n = _n(m["n"] or m["n2"], 2)
    if n < 2 or len(people) < n:
        return f"wanted {n} teams, but only {len(people)} people are here ({', '.join(people) or 'nobody'}) - say so"
    shuffled = people[:]
    _rng.shuffle(shuffled)
    split = [shuffled[i::n] for i in range(n)]
    return "split into teams: " + "; ".join(f"team {i + 1}: {', '.join(t)}" for i, t in enumerate(split))


def note(text: str, people: list[str] | None = None, speaker: str = "") -> str | None:
    """The [Random ...] note for a line, or None. people = who's in the call (humans), for picks and teams."""
    people = [p for p in dict.fromkeys(people or []) if p]
    for fn in (lambda t: teams(t, people), lambda t: pick(t, people, speaker), number, dice, coin):
        try:
            found = fn(text)
        except Exception:  # noqa: BLE001 - a helper, never a reason for a turn to fail
            found = None
        if found:
            return (f"[Random result - really random, already done for you: you {found}. Announce exactly this "
                    "result in your own words, keep it short; don't re-roll, change or add to it.]")
    return None
