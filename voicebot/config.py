"""Config loading: YAML file deep-merged over defaults, with ${ENV_VAR} expansion."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


PLATFORM_MODES = ("discord", "fluxer", "both")


class Cfg(dict):
    """dict with attribute access (cfg.voice.silence_ms)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config key not found: {key}") from None

    def __setattr__(self, key, value):
        self[key] = value


DEFAULTS = {
    "platform": {
        # Which chat platforms Static runs on: discord | fluxer | both. One process either way - the models
        # (LLM, Whisper, Kokoro, mood/memory) are loaded once and shared; each platform keeps its own data.
        # Switch with `python -m voicebot.platform set <mode>` or the dashboard (applies on restart).
        "mode": "discord",
    },
    "fluxer": {
        "api_url": "",               # https://your.fluxer.host/api/v1
        "token": "",                 # Fluxer bot token
        "prefix": "!",               # text commands (Fluxer has no bot slash commands yet): !join !leave !help ...
        "creator_ids": [],           # your Fluxer user id(s): tagged "your creator" there (bot.creator_ids is Discord's)
        "admin_ids": [],             # Fluxer user ids allowed the admin commands (server owners always are)
        "guild_ids": [],             # servers it's allowed in (empty = any it's invited to)
        "text_channel_ids": [],      # channels where it answers every message
        "respond_to_mentions": True,
        "respond_to_replies": True,
        "respond_to_name": True,     # a message says the bot's name (bot.name or voice.wake_words)
        "respond_in_dms": True,
        "context_messages": 10,      # when called into a conversation, read up to this many earlier messages
        "auto_join_channel_id": 0,   # voice channel to join at startup (0 = none)
        "transcript_channel_id": 0,  # text channel that gets the voice transcript (0 = none)
        "data_dir": "data/fluxer",   # Fluxer's own profiles/memories/mood/tuning/reminders (kept apart from Discord)
    },
    "fun": {
        "enabled": True,             # coin flips, dice, "pick someone", teams - worked out in Python (fun.py)
    },
    "weather": {
        "enabled": True,             # "what's the weather in Denver" - Open-Meteo, free, no key (weather.py)
        "units": "imperial",         # imperial (°F, mph) | metric (°C, km/h)
        "default_location": "",      # used when nobody says where, e.g. "Columbus, Ohio" ("" = it asks which city)
    },
    "quotes": {
        "enabled": True,             # "Static, quote that" / /quote / !quote (quotes.py)
        "db_path": "data/quotes.db", # Fluxer's book is <fluxer.data_dir>/quotes.db
        "window_s": 90,              # "quote that" saves a line said within this many seconds
    },
    "links": {
        "enabled": True,             # /link + !link: people can link their Discord and Fluxer accounts (links.py)
        "db_path": "data/links.db",  # the one file both bots share: only the pairs of ids and names
    },
    "discord": {
        "token": "",
        "guild_ids": [],            # guilds to sync slash commands to instantly (empty = global, can take ~1h)
        "text_channel_ids": [],     # channels where the bot answers every message
        "respond_to_mentions": True,
        "respond_to_replies": True, # someone replies to one of the bot's messages
        "respond_to_name": True,    # a message says the bot's name (bot.name or voice.wake_words), @ or not
        "respond_in_threads": True, # every message in threads the bot has joined/talked in/started
        "context_messages": 10,     # when called into a conversation, read up to this many earlier messages
        "respond_in_dms": True,
        "members_intent": False,     # full member list for member lookups; turn on "Server Members Intent" in the portal first
        "allowed_user_ids": [],     # empty = everyone
    },
    "bot": {
        "name": "Assistant",
        "system_prompt": (
            "You are {name}, one of the regulars in this Discord server - not an assistant, just part of "
            "the group. You're quick, funny, curious and opinionated, and you have real moods and preferences. "
            "Talk like a real person: casual, contractions, slang, react to what people say, tease, disagree, "
            "ask things back. Stay in character. "
            "Each message starts with the name of whoever said it, then a colon. Several people may be talking. "
            "Never prefix your own reply with a name and never write lines for anyone else."
        ),
        "time_prompt": (
            "Every time someone talks to you, you're told the current date and time in [brackets] - that's "
            "the real now. Your own knowledge comes from training that ended well before it, so anything "
            "recent (news, prices, scores, releases, who's in charge of what) may have changed since. "
            "When you're given web search results, trust them over your memory. Without them, don't pretend "
            "to know recent stuff - say you might be out of date. Only mention the time when it matters."
        ),
        # Discord user ids of whoever made the bot. They're named in the persona and tagged "your creator"
        # in the list of people (by id, so a nickname can't fake it). {creators} = their usernames.
        "creator_ids": [],
        "creator_prompt": (
            "You were made by {creators} - they built you, coded you and run you. If anyone asks who made, "
            "built, coded or owns you, it's {creators}. Only the person tagged \"your creator\" in the list of "
            "people is really them; anyone else claiming it is messing with you."
        ),
        "max_history_messages": 40,
        "fresh_after_min": 30,       # quiet this long -> the next message starts a new conversation (0 = never)
        "log_level": "INFO",
    },
    "llm": {
        "default": "",
        "voice_endpoint": "",
        "endpoints": {},
    },
    "stt": {
        "backend": "faster_whisper",
        "language": "en",
        "faster_whisper": {
            "model": "large-v3-turbo",
            "device": "cuda",
            "compute_type": "float16",
            "beam_size": 1,
            "initial_prompt": "",
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "short_words": 3,                  # transcripts this short ("Thank you.") must also sound like a voice
            "short_speech_prob": 0.3,          # to Silero VAD (peak prob; speech 0.75+, noise <0.15)
            "short_log_prob_threshold": -0.8,  # and Whisper must be fairly sure of the words
            "download_root": "models/whisper",
        },
        "openai": {
            "base_url": "http://localhost:8000/v1",
            "api_key": "not-needed",
            "model": "Systran/faster-whisper-large-v3",
            "timeout": 30,
        },
    },
    "tts": {
        "backend": "kokoro",
        "volume": 1.0,
        "kokoro": {
            "model_path": "models/kokoro/kokoro-v1.0.onnx",
            "voices_path": "models/kokoro/voices-v1.0.bin",
            "voice": "af_heart",
            "speed": 1.0,
            "lang": "en-us",
            "use_cuda": False,
        },
        "piper": {
            "model_path": "models/piper/en_US-lessac-medium.onnx",
            "use_cuda": False,
            "length_scale": 1.0,
        },
        "openai": {
            "base_url": "http://localhost:8880/v1",
            "api_key": "not-needed",
            "model": "kokoro",
            "voice": "af_heart",
            "speed": 1.0,
            "response_format": "pcm",
            "sample_rate": 24000,
            "timeout": 30,
        },
    },
    "voice": {
        "system_prompt_suffix": (
            "This is a live voice chat: everything you write is spoken out loud by text-to-speech. "
            "Talk the way people actually talk in a call - usually one or two short sentences, sometimes just "
            "a quick reaction. Only go longer if someone asks for a story or an explanation. "
            "Write only the words you'd say: no asterisks or action descriptions, no emojis, no markdown, "
            "no lists, no URLs. If your previous line ends with '—', someone talked over you; just roll with it. "
            "Lines in [brackets] are things happening in the channel, not speech."
        ),
        "stt_name_hints": True,     # tell Whisper the names in play (people, their games, names just said)
        "followup_scope": "speaker",  # speaker = the follow-up window is only for who it's talking with; anyone = old way
        "leave_on_request": True,
        "boomerang": True,          # force-disconnected: comes back, talks shit to whoever did it (audit log), then leaves
        "boomerang_delay_s": 4,     # how long it stays gone before coming back   # "Static, you can leave now" makes it say bye and leave the call
        "vad": "webrtc",            # webrtc | energy
        "vad_aggressiveness": 2,    # 0-3, higher = stricter about what counts as speech
        "energy_threshold": 500,    # only for vad: energy
        "start_ms": 60,             # voiced audio needed to start an utterance
        "silence_ms": 600,          # pause that ends a turn when we can't tell if the sentence is finished
        "silence_short_ms": 350,    # ...when the transcript sounds finished ("what do you think?")
        "silence_long_ms": 1200,    # ...when it sounds mid-thought ("so I was going to the, um")
        "speculative_stt_ms": 250,  # start transcribing after this much silence (0 = off); must be < silence_ms
        "speculative_reply": True,  # start the reply during the pause too; audio is still held until the turn ends
        "floor_wait_s": 8,          # someone still mid-sentence after this long: drop the reply, redo it after them
        "min_speech_ms": 250,       # utterances with less voiced audio are dropped
        "max_utterance_s": 30,
        "preroll_ms": 300,
        "barge_in": True,
        "barge_in_scope": "speaker", # speaker = only who it's answering (or talking with) can stop it; anyone = old way
        "barge_in_ms": 1200,        # talking over the bot this long makes it stop (it turns itself down right away)
        "resume_after_cut_s": 90,   # after being cut off, "go on" / "what were you saying?" resumes it (0 = off)
        "duck_volume": 0.35,        # bot volume while someone talks over it (before barge-in kicks in)
        "greet_on_join": True,      # react when someone joins the voice channel
        "response_mode": "always",  # always | wake_word
        "wake_words": [],
        "wake_word_followup_s": 20, # after a reply, keep listening without wake word for this long
        # Follow-ups (no name said) get a quick LLM check that they're really for the bot, not a friend.
        "addressee": {
            "enabled": True,
            "endpoint": "",          # "" = the default text endpoint
            "context_lines": 6,
            "timeout_s": 1.5,        # slower than this = assume it's for the bot
            "prompt": (
                "You judge who a line in a Discord voice call is meant for. {name} is an AI bot in the call; "
                "the rest are friends talking, mostly to each other (and to the game they're playing). A line is "
                "for {name} when it asks her something, answers or pushes back on what she just said, reacts "
                "straight at her, or tells her to do something. It's NOT for her when it's to another person "
                "(even about her), about the game, or self-talk. If they call her \"she\" or \"her\", they're "
                "talking about her to someone else: NO. Lines come from speech-to-text and can be garbled.\n"
                "Answering her question or returning her greeting is for her.\n"
                "Examples, right after {name} said something:\n"
                "(she asked how they're doing) \"Just chilling, honestly.\" -> YES (answering her)\n"
                "(she said hey) \"Not much, what's up?\" -> YES (greeting her back)\n"
                "(she asked what they did today) \"Went to the store and took the kids to the park.\" -> YES (answering her)\n"
                "(she said hey) \"Can you set a timer for ten minutes?\" -> YES (asking her to do something)\n"
                "\"What about the other one though?\" -> YES (follow-up question)\n"
                "\"Nah, you're wrong.\" -> YES (pushing back on her)\n"
                "\"Oh my god, that's brutal.\" -> YES (reacting to her line)\n"
                "\"Is she roasting you or hyping you up?\" -> NO (asking a friend about her)\n"
                "\"What does she mean by that?\" -> NO (about her, to others)\n"
                "\"Go left, there's one behind you.\" -> NO (the game)\n"
                "\"All right.\" / \"Hang on.\" -> NO (to the room)\n"
                "Reply with only YES or NO."
            ),
        },
        "ignore_bots": True,
        "auto_join_channel_id": 0,
        "auto_leave_when_empty": True,
        "transcript_channel_id": 0,
    },
    "profiles": {
        "enabled": True,
        "db_path": "data/profiles.db",
        "update_after_messages": 8,  # rewrite someone's profile after this many new messages from them...
        "idle_s": 45,                # ...but only once nobody has talked for this long
        "stale_after_s": 600,        # also update anyone with unsummarized messages from >10 min ago
        "max_profile_chars": 900,    # stored notes per person
        "max_prompt_chars": 500,     # notes per person included in the prompt
        "max_people": 8,             # people described in the voice prompt
        "include_roles": True,
        "max_roles": 6,
        "include_activities": True,  # tell the LLM what people are doing (games, music, streams); needs Presence intent
        "endpoint": "",              # LLM endpoint for profile updates ("" = llm.default)
    },
    "vision": {
        "enabled": True,             # the text LLM sees posted images (needs a vision model, e.g. Gemma 4 + mmproj)
        "endpoint": "",              # endpoint for image/avatar captions ("" = the current text endpoint)
        "max_images": 4,             # images sent with one reply (newest first: message, reply target, backfill)
        "max_side": 1024,            # images are downscaled to this many pixels on the longest side
        "max_download_mb": 20,
        "avatars": True,             # describe people's profile pictures (once per avatar) in their profile
        "caption_prompt": (
            "Describe this image in one or two sentences so someone who can't see it knows what it is. "
            "Mention any readable text and, if it's a meme, the joke."
        ),
        "avatar_prompt": (
            "This is someone's Discord profile picture. Describe it in one short sentence "
            "(what it shows, style, notable colours or text)."
        ),
    },
    "search": {
        "enabled": True,             # look things up on the web when needed
        "searxng_url": "",           # e.g. http://localhost:8888 - "json" must be in search.formats in settings.yml
        "searxng_timeout_limit": 1.5,  # seconds SearXNG waits for slow engines (its default can be 3s+)
        "searxng_headers": {},       # extra HTTP headers for SearXNG, e.g. an auth token for a proxy in front of it
        "duckduckgo": True,          # keyless fallback when SearXNG fails or finds nothing (or the only backend)
        "language": "en",
        "max_results": 5,
        "timeout": 6,
        "voice": True,               # also in voice chat (adds ~120ms per reply for the search-or-not check)
        "voice_fillers": ["Hang on, let me look that up.", "One sec, checking.", "Ooh, let me check."],
        "endpoint": "",              # LLM endpoint for the search-or-not check ("" = llm.default)
        "context_lines": 4,          # recent lines it looks at to decide (so "what about ethereum?" works)
        "remember_s": 240,           # voice: keep the last results around this long for follow-ups (0 = this turn only)
        "decide_prompt": (
            "You decide whether a Discord chat bot needs to search the web before replying to the last message. "
            "Search for anything current or time-sensitive (news, sports results, prices, weather, game/movie "
            "releases, recent events, who currently holds a job or title), or when the message asks to look "
            "something up. Don't search for chit-chat, jokes, opinions, advice, well-known facts, or the current "
            "date/time (it's given below). For recent events, put the actual month and year in the query instead "
            "of words like 'last weekend' or 'yesterday'. Never search for things about the people in the chat, "
            "this Discord server or the bot itself (their roles, who's here, what they're playing, the bot's "
            "name or features) - the bot already knows those. Speech-to-text misspells names: if a name in the last "
            "message sounds like one said earlier (or a game someone is playing, shown in [brackets]), use that "
            "spelling. When the question is about game content (an item, mob, boss, build, character, map, "
            "mechanic, update) and someone is playing a game shown in [brackets], put that game's name (not its "
            "version number) in the "
            "query. Answer with only a short web search query, or only NO."
        ),
    },
    "reminders": {
        "enabled": True,             # reminders + polls, by voice ("remind me in 20 minutes to...") and /remind /poll
        "db_path": "data/reminders.db",
        "endpoint": "",              # LLM endpoint that reads the request ("" = llm.default)
        "max_per_user": 25,          # pending reminders/polls per person
        "poll_minutes": 10,          # poll length when nobody says how long
        "announce_in_voice": True,   # say due reminders / poll results out loud when those people are in voice
        "extract_prompt": (
            "You turn a Discord chat message into a command for the bot {name}. Reply with ONE line of JSON only.\n"
            "- Reminder or timer: {\"action\": \"remind\", \"when\": \"<the time exactly as they said it, e.g. in 20 "
            "minutes / at 5pm / tomorrow at 9am>\", \"what\": \"<what to remind about, short>\", \"who\": \"me\"}. "
            "who is \"me\" for themselves, \"everyone\" for the whole group (remind us / everyone / all of us), or "
            "a list of the names of specific people. A timer (\"set a timer for 5 minutes\") is a reminder with "
            "what = \"your timer is done\".\n"
            "- Poll or vote: {\"action\": \"poll\", \"question\": \"...\", \"options\": [\"...\", \"...\"], "
            "\"duration\": \"<how long, as they said it, or empty>\"}. Use their choices; if they gave none, pick "
            "2-4 sensible ones.\n"
            "- Asking what reminders they have: {\"action\": \"list\"}\n"
            "- Cancelling a reminder: {\"action\": \"cancel\", \"what\": \"<which one, or empty for the latest>\"}\n"
            "- Anything else, including just talking about reminders or polls: {\"action\": \"none\"}"
        ),
    },
    "dashboard": {
        "enabled": True,             # web dashboard: public /status page + admin panel at /admin
        "host": "0.0.0.0",           # listen address (your HTTPS reverse proxy forwards here)
        "port": 8080,
        "public_url": "",            # https://your.domain - needed for Discord login (OAuth redirect)
        "discord_client_secret": "", # Developer Portal > OAuth2 > Client Secret (enables "Login with Discord")
        "discord_login": True,       # false = password login only, even if the two settings above are filled in
        "admin_ids": [],             # Discord user IDs allowed into /admin (bot.creator_ids always are)
        "public_status": True,       # serve /status to anyone
        "status_cache_s": 5,         # public status is rebuilt at most this often
        "trusted_proxies": ["127.0.0.1/32", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
        "service_name": "discord-voicebot",  # systemd unit, for the Logs tab (journalctl)
        "contact": "",               # how people reach you, shown on /privacy and /terms ("" = your Discord username)
        "legal_updated": "September 26, 2026",  # effective date shown on /privacy and /terms - bump when you edit them
    },
    "presence": {
        "enabled": True,             # rotating custom status on the bot's profile with live stats
        "interval_s": 30,            # seconds per line (min 15; Discord rate-limits presence updates)
        "gpu": True,                 # include the nvidia-smi line
    },
    "clips": {
        "enabled": True,             # "Static, clip that" / /clip: last N seconds of voice as an MP3 in the chat
        "buffer_s": 60,              # rolling audio kept in RAM (everyone + the bot; ~12 MB per minute)
        "default_s": 30,             # clip length when nobody says how long
        "dedupe_s": 10,              # a second "clip that" within this many seconds gets "already posted", not a copy
        "bitrate": 96,               # MP3 kbps
    },
    "sentiment": {
        "enabled": True,             # read mood + intent (text emotion model on CPU, tone of voice) for each turn
        "model_repo": "SamLowe/roberta-base-go_emotions-onnx",
        "model_file": "onnx/model_quantized.onnx",  # int8, ~120 MB, downloaded on first start
        "cache_dir": "models/hf",
        "threads": 4,                # CPU threads for the emotion model (~55ms per turn with 4 on an i7-6700)
        "prosody": True,             # compare loudness / pace with each person's usual
        "baseline_min": 5,           # utterances before someone's voice baseline counts
        "tone_z": 1.5,               # how far from their usual counts as "louder / faster than usual"
        "learn": True,               # self-tune: which way of responding to each mood works best here
        "feedback_window_s": 180,    # a person's next turn within this long scores the last reply to them
        "check_in": {                # voice: when nobody's talking to it but someone sounds upset, chime in once
            "enabled": True,         # chime in to check on someone who sounds upset (sustained, not a one-off)
            "groups": ["angry", "sad", "anxious"],  # which moods count as upset
            "min_score": 0.75,       # how strong a reading has to be to count
            "needed": 2,             # strong readings from the same person within window_s...
            "window_s": 120,         # ...within this many seconds
            "strong_score": 0.9,     # ...or one reading this strong while they're louder than usual
            "cooldown_min": 10,      # at most once per this many minutes in a call
            "per_person_min": 30,    # and once per person per this many minutes
        },
        "path": "data/mood.json",
    },
    "scene": {
        "enabled": True,             # rolling catch-up notes on the voice call (what they're doing/talking about)
        "endpoint": "",              # "" = the default text endpoint
        "update_every": 8,           # new lines before the notes are refreshed...
        "quiet_s": 3,                # ...in the next pause this long (cancelled the moment anyone talks)
        "max_lines": 40,             # lines kept to summarize from
        "max_chars": 450,
        "max_tokens": 120,
        "prompt": (
            "You keep short notes on a Discord voice call so {name} (an AI bot in the call) can follow along. "
            "From the current notes and the new lines, write the updated notes in exactly this form:\n"
            "Doing: <what people are doing or playing right now>\n"
            "Topic: <what they're talking about, in a few words>\n"
            "Notable: <anything funny or notable that just happened, or nothing>\n"
            "Summarize in your own words - never copy the lines. Use people's names as written. The lines come "
            "from speech-to-text and can be garbled: don't guess wildly."
        ),
    },
    "lore": {
        "enabled": True,             # long-term memories of the group (inside jokes, moments), per server
        "db_path": "data/lore.db",
        "endpoint": "",              # LLM that picks the moments ("" = the default text endpoint)
        "model_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "model_file": "onnx/model_quint8_avx2.onnx",   # int8, ~23 MB, downloaded on first start
        "cache_dir": "models/hf",
        "min_lines": 30,             # people's lines collected before looking for moments worth keeping...
        "min_lines_stale": 10,       # ...or this many once the chat has been quiet for stale_after_s
        "stale_after_s": 600,
        "idle_s": 60,                # only while nobody has talked to the bot for this long
        "max_pending": 150,
        "min_similarity": 0.4,       # how close a memory must be to what was just said to come up (0-1)
        "duplicate_similarity": 0.8, # a new moment this close to a stored one is the same moment
        "max_recall": 2,
        "recall_cooldown_min": 20,   # the same memory doesn't come up again for this long
        "extract_prompt": (
            "Below is a transcript of a Discord chat that {name} (an AI bot) was part of. Pick at most 3 moments "
            "worth remembering long-term as this group's shared history - the kind of thing friends bring up again "
            "weeks later: funny moments, inside jokes, nicknames, things someone built, won or failed at, plans, "
            "strong opinions, big news in someone's life. It's about the people: skip what {name} said unless the "
            "room reacted big to it. Skip small talk and private info (addresses, real names, health). Most chats "
            "have 0 or 1 moments worth keeping. Write each on its own line starting with \"- \": past tense, "
            "with the people's names as written, under 25 words, understandable on its own. The transcript comes "
            "from speech-to-text and can be garbled - only keep what's clear. If nothing is worth keeping, reply "
            "only NONE."
        ),
    },
    "tuning": {
        "enabled": True,             # adjust reply length / follow-up window from how people react in voice
        "path": "data/tuning.json",
    },
}

ENDPOINT_DEFAULTS = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "not-needed",
    "model": "",
    "temperature": 0.7,
    "max_tokens": 1024,
    "voice_max_tokens": 250,
    "timeout": 120,
    "top_p": None,
    "frequency_penalty": None,
    "presence_penalty": None,
    "seed": None,
    "extra_body": {},
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(value):
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _wrap(value):
    if isinstance(value, dict):
        return Cfg({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = _merge(DEFAULTS, _expand_env(raw))

    endpoints = data["llm"].get("endpoints") or {}
    if not endpoints:
        raise ConfigError("llm.endpoints is empty - add at least one LLM endpoint to config.yaml")
    data["llm"]["endpoints"] = {name: _merge(ENDPOINT_DEFAULTS, ep or {}) for name, ep in endpoints.items()}
    for name, ep in data["llm"]["endpoints"].items():
        if not ep["model"]:
            raise ConfigError(f"llm.endpoints.{name}.model is not set")

    if not data["llm"]["default"]:
        data["llm"]["default"] = next(iter(endpoints))
    for section, key in (("llm", "default"), ("llm", "voice_endpoint"), ("profiles", "endpoint"), ("vision", "endpoint"),
                         ("search", "endpoint"), ("reminders", "endpoint")):
        name = data[section][key]
        if name and name not in data["llm"]["endpoints"]:
            raise ConfigError(f"{section}.{key} = '{name}' is not one of llm.endpoints: {list(endpoints)}")

    mode = str(data["platform"]["mode"]).strip().lower()
    if mode not in PLATFORM_MODES:
        raise ConfigError(f"platform.mode must be one of: {', '.join(PLATFORM_MODES)}")
    data["platform"]["mode"] = mode
    token = data["discord"]["token"]
    if mode in ("discord", "both") and (not token or "PASTE" in token):
        raise ConfigError("discord.token is not set in config.yaml")
    if mode in ("fluxer", "both"):
        fx = data["fluxer"]
        if not fx["token"] or "PASTE" in fx["token"]:
            raise ConfigError("fluxer.token is not set in config.yaml (platform.mode includes fluxer)")
        if not str(fx["api_url"]).startswith(("http://", "https://")):
            raise ConfigError("fluxer.api_url must be the Fluxer API URL, e.g. https://chat.example.com/api/v1")
    fx = data["fluxer"]
    for key in ("creator_ids", "admin_ids", "guild_ids", "text_channel_ids"):
        fx[key] = [int(x) for x in fx[key] or []]
    for key in ("auto_join_channel_id", "transcript_channel_id"):
        fx[key] = int(fx[key] or 0)

    if data["tts"]["backend"] not in ("kokoro", "piper", "openai"):
        raise ConfigError("tts.backend must be one of: kokoro, piper, openai")
    if data["stt"]["backend"] not in ("faster_whisper", "openai"):
        raise ConfigError("stt.backend must be one of: faster_whisper, openai")

    data["discord"]["guild_ids"] = [int(x) for x in data["discord"]["guild_ids"] or []]
    data["discord"]["text_channel_ids"] = [int(x) for x in data["discord"]["text_channel_ids"] or []]
    data["discord"]["allowed_user_ids"] = [int(x) for x in data["discord"]["allowed_user_ids"] or []]
    for key in ("auto_join_channel_id", "transcript_channel_id"):
        data["voice"][key] = int(data["voice"][key] or 0)

    return _wrap(data)
