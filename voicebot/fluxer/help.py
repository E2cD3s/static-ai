"""The Fluxer bot's ! commands, described once: used by !help (fluxer/bot.py) and the web /help page's Fluxer tab
(helpinfo.py). Keep it in step with the _cmd_* handlers in fluxer/bot.py. Groups match helpinfo.GROUPS."""
from __future__ import annotations

# (name, arguments, description, who, group)
COMMANDS: list[tuple[str, str, str, str, str]] = [
    ("join", "", "Join your voice channel and start listening", "everyone", "Voice"),
    ("leave", "", "Leave the voice channel", "everyone", "Voice"),
    ("stop", "", "Stop talking", "everyone", "Voice"),
    ("clip", "[seconds]", "Post the last few seconds of the voice channel as an MP3", "everyone", "Voice"),
    ("remind", "in 20 minutes to check the oven", "Set a reminder (add \"everyone\" for the whole call)", "everyone", "Reminders & polls"),
    ("reminders", "[cancel <id>]", "Your pending reminders (by DM), or cancel one", "everyone", "Reminders & polls"),
    ("poll", "question | choice, choice | 10 minutes", "Start a reaction poll - the result is announced in voice", "everyone", "Reminders & polls"),
    ("quote", "[words]", "A random quote, or search; reply to a message with it to save that message", "everyone", "Fun"),
    ("profile", "[@someone]", "What the bot remembers about you, by DM (admins: about anyone)", "everyone", "Your data"),
    ("forget", "[@someone]", "Delete everything the bot remembers about you (admins: about anyone)", "everyone", "Your data"),
    ("profiling", "on|off", "Turn off (or back on) the bot building a profile of you", "everyone", "Your data"),
    ("link", "[code]", "Link your Discord account (get a code, or enter one from Discord's /link)", "everyone", "Your data"),
    ("unlink", "", "Unlink your Discord and Fluxer accounts", "everyone", "Your data"),
    ("status", "", "Uptime, models and voice reply times", "everyone", "Info"),
    ("help", "", "How to talk to the bot and its commands", "everyone", "Info"),
    ("say", "<text>", "Make the bot say something in the voice channel", "admins", "Server admins"),
    ("reset", "", "Clear the bot's conversation memory here", "admins", "Server admins"),
    ("tuning", "[reset]", "How the bot has tuned its voice replies (by DM)", "admins", "Server admins"),
    ("llm", "<endpoint> [model] [text|voice|both]", "Switch the LLM (shared with the Discord bot)", "owner", "Owner"),
    ("models", "[endpoint]", "List models available on an endpoint", "owner", "Owner"),
]


def as_dicts(prefix: str) -> list[dict]:
    """For the web help page (same shape as helpinfo.commands, plus the prefix and example arguments)."""
    return [{"name": n, "args": a, "description": d, "who": w, "group": g, "prefix": prefix,
             "params": []} for n, a, d, w, g in COMMANDS]
