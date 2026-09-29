"""Server member lookup: "what roles does Casey have?", "check my roles", "who is X in this Discord?".

The prompt only describes the people in the voice channel, and the web search knows nothing about this server,
so questions about other members got game trivia instead. This finds the member (cached members, then Discord's
name search, which matches usernames and nicknames by prefix and works without the Server Members intent) and
hands the model a short fact sheet for that one turn.
"""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata

import discord

from .profiles import member_presence, member_roles

log = logging.getLogger("voicebot.members")

# The question is about this server's people or roles ("rolls": speech-to-text's usual spelling of "roles").
_ABOUT_SERVER = re.compile(r"\b(?:roles?|rolls?|discord|server|members?|users?|usernames?|joined|nicknames?)\b", re.I)
_ABOUT_ME = re.compile(r"\b(?:my|mine)\b[^.?!]*\b(?:roles?|rolls?)\b|\b(?:roles?|rolls?)\b[^.?!]*\b(?:do|have|did) i\b", re.I)
_LOOKUP = re.compile(r"\b(?:look(?:\s+\w+)?\s+up|find|search|check|who(?:'s| is| are)|what roles|which roles)\b", re.I)
_WORD = re.compile(r"[^\W_][\w.'-]{2,}", re.U)
_STOP = set("""
the and can you look up find search check roles role rolls roll discord server this that what who whos who's their them
they give please tell have has does did list user users member members named called name names static some with about
are for from here there his her him she he its it's is of to do get got all any one like just okay ok yeah yes need want
show see know goofy ass characters character weird person people guy girl dude bro in on me my mine our your yours
we us an a be been being was were will would could should also again then than so if or not no nah hey yo lol
right now today tell tells telling what's whats where when why how much many more most very really thing things
""".split())


def _norm(text: str) -> str:
    """'🔥𝓒á𝓼𝓮𝔂✨' -> 'casey' (fancy fonts and accents fold to plain letters; emoji and symbols drop)."""
    folded = unicodedata.normalize("NFKD", text)
    return "".join(c for c in folded if c.isascii() and c.isalnum()).lower()


def _names(member) -> set[str]:
    return {n for n in (_norm(member.name), _norm(getattr(member, "display_name", "")),
                        _norm(getattr(member, "global_name", None) or ""), _norm(getattr(member, "nick", None) or ""))
            if n}


def _matches(word: str, member) -> bool:
    return any(word == n or (len(word) >= 4 and word in n) for n in _names(member))


def describe(member, guild) -> str:
    roles = member_roles(member)
    parts = [f"roles ({len(roles)}): {', '.join(roles)}" if roles else "no roles"]
    if guild.owner_id == member.id:
        parts.append("server owner")
    elif getattr(getattr(member, "guild_permissions", None), "administrator", False):
        parts.append("admin")
    if getattr(member, "joined_at", None):
        parts.append(f"joined the server {member.joined_at:%b %d, %Y}")
    if getattr(member, "voice", None) and member.voice.channel:
        parts.append(f"in voice channel {member.voice.channel.name}")
    if doing := member_presence(member):
        parts.append("; ".join(doing))
    shown = member.display_name if _norm(member.display_name) == _norm(member.name) else f"{member.display_name} / @{member.name}"
    return f"- {shown}: " + "; ".join(parts)


async def lookup(guild, speaker, text: str, skip_names=()) -> str | None:
    """A [bracketed] note with the members the text asks about, or None if it isn't about members."""
    if guild is None or not _ABOUT_SERVER.search(text):
        return None
    if not (_ABOUT_ME.search(text) or _LOOKUP.search(text) or re.search(r"\b(?:roles?|rolls?)\b", text, re.I)):
        return None  # "the minecraft server is down" mentions the server but asks about nobody
    skip = {_norm(n) for n in skip_names}
    found: dict[int, object] = {}
    if _ABOUT_ME.search(text) and speaker is not None:
        found[speaker.id] = speaker
    words = [w for w in (_norm(x) for x in _WORD.findall(text)) if len(w) >= 3 and w not in _STOP and w not in skip]
    missing = []
    for w in dict.fromkeys(words):
        hits = [m for m in guild.members if not m.bot and _matches(w, m)]
        if hits:
            found.update((m.id, m) for m in hits[:3])
        else:
            missing.append(w)
    for w in missing[:3]:  # not cached: ask Discord (prefix match on username / nickname)
        try:
            hits = await asyncio.wait_for(guild.query_members(query=w, limit=5), timeout=3)
        except (discord.HTTPException, asyncio.TimeoutError) as e:
            log.warning("Member search for %r failed: %s", w, e)
            continue
        found.update((m.id, m) for m in hits if not m.bot and _matches(w, m))
    if found:
        people = list(found.values())[:4]
        log.info("👥 member lookup: %s", ", ".join(m.name for m in people))
        return ("[Discord member info, from this server (not the web). Answer from it, and if they ask for roles, "
                "read them all out:\n" + "\n".join(describe(m, guild) for m in people) + "]")
    if _LOOKUP.search(text) and words:
        log.info("👥 member lookup: nobody matches %s", words[:3])
        return (f"[You looked in this Discord server for \"{' '.join(words[:3])}\" and found no member by that name. "
                "Say you couldn't find them here - don't guess or make anything up.]")
    return None


# ---------------------------------------------------------------- roasts

_ROAST = re.compile(r"\b(?:roast(?:ing)?|flame|diss|clown on|cook|talk (?:shit|smack|trash|crap)(?: about| to| on)?|"
                    r"make fun of|rip on|go off on|trash talk)\b", re.I)
_ROAST_ME = re.compile(r"\b(?:roast|flame|diss|cook|clown on|make fun of|rip on|go off on)\s+(?:me|myself)\b|"
                       r"\b(?:about|to|on) me\b", re.I)
# The opposite bit: "rizz up jordan", "hype him up", "compliment sam", "flirt with her".
_HYPE = re.compile(r"\b(?:rizz(?:\s+up)?|hype\s+up|gas\s+up|glaze|compliment|flirt\s+with|shoot\s+your\s+shot\s+(?:at|with))\b|"
                   r"\b(?:hype|gas|rizz)\s+(?:\w+\s+)?up\b", re.I)
_HYPE_ME = re.compile(r"\b(?:rizz|hype|gas|glaze|compliment|flirt\s+with)\s+(?:me|myself)\b|\b(?:rizz|hype|gas)\s+me\s+up\b", re.I)
_ROAST_WORDS = set("roast roasting flame diss clown cook talk shit smack trash crap make fun rip off hard "
                   "rizz hype gas glaze compliment flirt shoot your shot up".split())


def roast_request(text: str) -> str | None:
    """ "Static, roast jordan", "talk shit about him", "cook this guy", "roast me" -> "roast";
    "rizz up jordan", "hype him up", "compliment me" -> "hype"; else None."""
    if _ROAST.search(text):
        return "roast"
    if _HYPE.search(text):
        return "hype"
    return None


def roast_target(pool, text: str, requester, recent_speakers: list[str]):
    """Who to roast: a name in the request (the call first, then the whole server), "me" = whoever asked, or for
    "him" / "this guy" / no name, the last other person who spoke."""
    if _ROAST_ME.search(text) or _HYPE_ME.search(text):
        return requester
    m = _ROAST.search(text) or _HYPE.search(text)
    tail = text[m.end():] if m else text
    for w in (_norm(x) for x in _WORD.findall(tail)):
        if len(w) < 3 or w in _STOP or w in _ROAST_WORDS:
            continue
        hits = [p for p in pool if not p.bot and _matches(w, p)]
        if hits:
            return hits[0]
    rid = getattr(requester, "id", None)
    for name in reversed(recent_speakers):
        hit = next((p for p in pool if not p.bot and p.display_name == name and p.id != rid), None)
        if hit is not None:
            return hit
    return None


def roast_note(requester_name: str, target, notes: str, lines: list[str], avatar: str = "",
               mode: str = "roast") -> str:
    """The note that makes a small model actually roast (or rizz) instead of hedging: who, what to aim at, how."""
    doing = "; ".join(member_presence(target))
    ammo = [f"what you know about them: {notes[:400]}" if notes else "",
            f"what they're doing right now: {doing}" if doing else "",
            f"their profile picture: {avatar[:200]}" if avatar else "",
            "their recent lines: " + " / ".join(f'"{x}"' for x in lines[-3:]) if lines else ""]
    ammo = "; ".join(a for a in ammo if a)
    who = "themselves" if getattr(target, "display_name", "") == requester_name else target.display_name
    if mode == "hype":
        return (f"[{requester_name} wants you to rizz up {who}. Talk straight to {target.display_name}: two or three "
                "short, smooth, cheesy-on-purpose lines - pickup-line energy with your usual attitude"
                + (f" - material: {ammo}" if ammo else "") + ". Build every line on something real about them "
                "(pick one or two details, don't list them). Plain words a person would say out loud - no made-up "
                "compliments like 'built-in' or 'aesthetic', no hedging, and don't make it creepy.]")
    return (f"[{requester_name} wants you to roast {who}. Go hard: two or three short, savage, specific lines aimed "
            f"right at {target.display_name}" + (f" - ammo: {ammo}" if ammo else "") + ". Specific beats generic: "
            "build every line on something real about them (from the ammo, or what they just said). No compliments, no hedging, "
            "no 'I guess', and don't tell someone else to do it. It's banter between friends - make the room laugh.]")
