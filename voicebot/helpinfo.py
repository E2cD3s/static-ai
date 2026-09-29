"""What the bot can do, for people: the public /help page on the dashboard and the /help command.
Built from the live config and the registered slash commands, so it never drifts from what's actually on.
`platforms` says whether the Discord and/or Fluxer bot runs (Fluxer uses ! commands - see fluxer/bot.py)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from discord import app_commands

from .fluxer import help as fluxer_help

if TYPE_CHECKING:
    from .bot import VoiceBot

# Who can use a command, where it isn't everyone. Anything else is open to all.
_WHO = {"llm": "owner", "models": "owner", "say": "admins", "reset": "admins", "tuning": "admins"}
# Order + grouping on the help page / in /help.
GROUPS = [
    ("Voice", ["join", "leave", "stop", "clip"]),
    ("Reminders & polls", ["remind", "reminders", "poll"]),
    ("Fun", ["quote"]),
    ("Your data", ["profile", "forget", "profiling", "link", "unlink"]),
    ("Info", ["status", "help"]),
    ("Server admins", ["say", "reset", "tuning"]),
    ("Owner", ["llm", "models"]),
]


def commands(bot: "VoiceBot") -> list[dict]:
    out = []
    for cmd in bot.tree.get_commands():
        if not isinstance(cmd, app_commands.Command):
            continue
        params = [{"name": p.name, "description": p.description if p.description != "…" else "",
                   "required": p.required} for p in cmd.parameters]
        group = next((g for g, names in GROUPS if cmd.name in names), "Other")
        out.append({"name": cmd.name, "description": cmd.description, "params": params,
                    "who": _WHO.get(cmd.name, "everyone"), "group": group})
    order = [n for _, names in GROUPS for n in names]
    return sorted(out, key=lambda c: (order.index(c["name"]) if c["name"] in order else len(order), c["name"]))


def info(bot: "VoiceBot") -> dict:
    cfg, v, d = bot.cfg, bot.cfg.voice, bot.cfg.discord
    u = bot.user
    wake = list(v.wake_words) or [cfg.bot.name]
    return {
        "name": cfg.bot.name,
        "user": u.name if u else None,
        "avatar": u.display_avatar.url if u else None,
        "voice": {
            "mode": v.response_mode,                      # wake_word | always
            "wake_words": wake,
            "followup_s": float(v.wake_word_followup_s),  # the configured start; it self-tunes per server
            "followup_scope": v.followup_scope,           # speaker | anyone
            "barge_in": bool(v.barge_in),
            "greet_on_join": bool(v.greet_on_join),
            "leave_on_request": bool(v.leave_on_request),
            "boomerang": bool(v.get("boomerang")),
            "auto_leave_when_empty": bool(v.auto_leave_when_empty),
            "barge_in_s": float(v.barge_in_ms) / 1000,
            "barge_in_scope": v.get("barge_in_scope", "speaker"),
            "resume_s": float(v.get("resume_after_cut_s") or 0),
        },
        "text": {
            "mentions": bool(d.respond_to_mentions), "replies": bool(d.respond_to_replies),
            "name": bool(d.respond_to_name), "threads": bool(d.respond_in_threads),
            "dms": bool(d.respond_in_dms), "channels": len(d.text_channel_ids),
            "context_messages": int(d.context_messages),
        },
        "features": {
            "search": bool(bot.search.enabled),
            "clips": bool(cfg.clips.enabled), "clip_buffer_s": float(cfg.clips.buffer_s),
            "clip_default_s": float(cfg.clips.default_s),
            "reminders": bool(bot.planner.enabled),
            "profiles": bool(cfg.profiles.enabled),
            "lore": bool(bot.lore.enabled and bot.lore.model is not None),
            "calc": True,
            "activities": bool(cfg.profiles.include_activities),
            "vision": bool(cfg.vision.enabled),
            "mood": bool(bot.mood.enabled and bot.mood.model is not None),
            "tuning": bool(bot.tuning.enabled),
            "check_in": bool(bot.mood.enabled and bot.mood.model is not None
                             and (cfg.sentiment.get("check_in") or {}).get("enabled")),
            "check_in_cooldown_min": float((cfg.sentiment.get("check_in") or {}).get("cooldown_min", 10)),
            "fun": bool(cfg.fun.enabled),
            "weather": bool(bot.weather.enabled),
            "weather_home": bool(str(cfg.weather.get("default_location") or "").strip()),
            "quotes": bool(cfg.quotes.enabled),
            "links": bool(bot.links.enabled) and cfg.platform.mode == "both",
        },
        "platforms": {
            "discord": cfg.platform.mode in ("discord", "both"),
            "fluxer": cfg.platform.mode in ("fluxer", "both"),
            "fluxer_prefix": str(cfg.fluxer.prefix or "!"),
            "fluxer_host": "/".join(str(cfg.fluxer.api_url).split("/", 3)[2:3]) or None,
        },
        # The Fluxer bot answers text on its own settings (fluxer.*) and has ! commands instead of slash commands.
        "fluxer_text": {
            "mentions": bool(cfg.fluxer.respond_to_mentions), "replies": bool(cfg.fluxer.respond_to_replies),
            "name": bool(cfg.fluxer.respond_to_name), "threads": False, "dms": bool(cfg.fluxer.respond_in_dms),
            "channels": len(cfg.fluxer.text_channel_ids), "context_messages": int(cfg.fluxer.context_messages),
        },
        "fluxer_commands": fluxer_help.as_dicts(str(cfg.fluxer.prefix or "!")),
        "models": {
            "llm": bot.llm.model_for(bot.llm.voice_endpoint),
            "stt": getattr(bot.stt, "desc", None),
            "tts": getattr(bot.tts, "desc", None),
        },
        "commands": commands(bot),
        "url": (cfg.dashboard.public_url or "").rstrip("/") + "/help" if cfg.dashboard.public_url else None,
    }
