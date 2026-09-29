# Static - Discord (+ Fluxer) voice LLM bot

A bot ("Static") that talks in voice channels and text chat, fully local, on Discord and optionally on a
self-hosted Fluxer server (same process, `platform.mode: discord|fluxer|both`):
voice → per-user VAD → faster-whisper (GPU) → Ollama LLM (streamed) → Kokoro TTS (GPU) → voice.

## Layout

- `main.py` - entry point (`python main.py [config.yaml]`); chdirs to the project root.
- `voicebot/bot.py` - discord.Client, slash commands, text chat, join/leave events. Patches
  `Member/BaseUser.display_name` to return the account username everywhere (nicknames vary per server and are
  often fancy-font/emoji). People are tracked by user id; names are only labels.
- `voicebot/voice_session.py` - the live voice loop: VAD + adaptive end-of-turn (speculative STT),
  barge-in/ducking, backchannel filter, streaming LLM → sentence chunks → TTS → playback.
- `voicebot/llm.py` - OpenAI-compatible client (Ollama at `localhost:11434/v1`), message normalization.
- `voicebot/stt.py`, `voicebot/tts.py`, `voicebot/audio.py` - speech models and PCM plumbing.
- `voicebot/text_utils.py` - streaming filters (`<think>`, `*actions*` - names/quotes/mid-sentence emphasis kept),
  sentence chunker, `SpeakerGuard` (strips "Static:" labels, stops the model writing other people's lines),
  `EchoGuard` (drops the model's copy of the message it's answering), leave / "go on" phrase detection, heuristics.
- `voicebot/members.py` - server member lookup ("what roles does X have?", "list my roles"): cached members with
  fancy-font folding, then `guild.query_members`; injects a fact sheet for that turn instead of a web search.
- `voicebot/helpinfo.py` - live facts (name, wake words, features, slash commands + who can use them) for the
  `/help` page and the `/help` embed.
- `voicebot/reminders.py` - reminders + polls (voice/text requests read by a small LLM JSON call, `/remind`
  `/reminders` `/poll`), stored in `data/reminders.db`, fired by one scheduler task. Times parsed by `dateparser`.
- `voicebot/stats.py` - `/status` embeds (LLM token/speed accounting from `LLMRouter.usage`, STT/TTS realtime
  factors, latency percentiles, nvidia-smi, psutil, Ollama `/api/ps`). Modules keep their own counters for it.
- `voicebot/web/` - aiohttp dashboard in the bot process (`dashboard:` in config): public `/status` (structured JSON from
  `snapshot.py` drawn by `static/board.js`; server/channel names are stripped from the public copy), `/help` (user
  guide: prose in `help.html`, live values from `/api/help`), `/terms`, `/privacy` (`{{placeholders}}` filled
  server-side), `/linked-role` (Discord Linked Roles OAuth, metadata registered on first use). Shared header menu +
  footer on every page. The admin Overview adds per-server learned
  tuning/mood strategies from `snapshot.learned`). One theme for every page in `static/style.css`, taken from the bot's avatar (dark only), admin `/admin` (servers/voice/leave/invite, People = everything stored per user + forget/opt-out, settings editor that writes
  `config.yaml` with ruamel keeping comments + validates with `load_config` + applies in place, logs, restart =
  exit 75 so systemd restarts). Auth: Discord OAuth2 (`admin_ids` + `bot.creator_ids`) or a local password
  (`python -m voicebot.web set-password`), HMAC-signed cookies, state in `data/dashboard.json`.
- `voicebot/presence.py` - rotating custom status on the bot's profile (live LLM/voice/GPU stats, `presence:`
  in config). Bots can't have full Rich Presence (no images/buttons), only this line.
- `voicebot/clips.py` - rolling RAM buffer of the voice channel mix (people + bot via `StreamingPCMSource.tap`)
  for "clip that" / `/clip`, MP3 via lameenc. `voicebot/tuning.py` - self-tuning reply length + follow-up window
  (per guild, `data/tuning.json`, `/tuning`); hints go on the newest turn, never the system prompt.
- `voicebot/sentiment.py` - mood/intent per turn: GoEmotions int8 ONNX on CPU (`mood_executor`), tone of voice
  vs per-user baseline, Thompson-sampled response strategies + per-user thresholds learned in `data/mood.json`.
- `voicebot/profiles.py` - per-user identity/roles + long-term LLM-written profiles in `data/profiles.db` (SQLite).
- `voicebot/addressee.py` - "is this follow-up for me?": follow-ups without the name get a YES/NO LLM check
  (started during the pause alongside the speculative reply). Logged `↪ not for me`.
- `voicebot/calc.py` - arithmetic / unit conversion / time-in-a-city worked out in Python, handed over as a note (🧮).
- `voicebot/fun.py` - coin / dice / random number / pick someone / teams, real randomness, same note pattern (🎲).
- `voicebot/weather.py` - Open-Meteo (no key) forecasts, place + day parsed from the line, 10-min cache (🌤).
  One `Weather` on the Discord bot, shared by Fluxer. `weather.default_location` for "what's the weather?".
- `voicebot/quotes.py` - quote book: "quote that" in voice (VoiceSession keeps `_said`, the recent lines),
  `/quote` + "Save as quote" message menu, `!quote` on Fluxer. One book per platform (💬).
- `voicebot/links.py` - opt-in Discord<->Fluxer account links (`/link` <-> `!link` 6-digit code, 10 min).
  `data/links.db` is the only data both bots share; linked people get the other bot's notes via `profiles.linked`.
- `voicebot/scene.py` - per-call catch-up notes (Doing/Topic/Notable), refreshed in pauses (🧭), newest turn only.
- `voicebot/lore.py` - server memories (📜): LLM picks 0-3 moments from idle chat, MiniLM int8 ONNX embeddings on
  CPU, recalled by similarity per turn. `data/lore.db`; /forget and opt-out delete a person's memories.
- `voicebot/replay.py` + `evals/cases.yaml` - replay test set against the live Ollama:
  `.venv/bin/python -m voicebot.replay [-k id] [-v]`. Run it before/after prompt changes; add a case per bug.
- `voicebot/fluxer/` - Static on Fluxer (self-hosted, `platform.mode: discord|fluxer|both`, switch with
  `python -m voicebot.platform set <mode>` or the dashboard header). Same process: VoiceBot hosts the shared models
  (and the dashboard) even when Discord is off; FluxerBot borrows them and keeps its own data in `data/fluxer/`
  (`_fluxer_cfg` overlays the paths). `objects.py` = Discord-shaped wrappers so shared code runs unchanged,
  `voice.py` = LiveKit behind VoiceRecvClient's calls (+ a per-speaker noise gate: LiveKit sends open mics
  continuously), `bot.py` = text chat (a copy of Discord's on_message - mirror chat changes there) + `!` commands.
  `help.py` = the `!` command list (name/args/description), used by `!help` and the web /help page's Fluxer tab.
- `voicebot/platform.py` - CLI: `python -m voicebot.platform [set discord|fluxer|both] [--restart]`.
- `voicebot/config.py` - `DEFAULTS` deep-merged under `config.yaml`. New settings go in `DEFAULTS`
  (so existing configs keep working) and are documented in `config.example.yaml`.
- `install.sh` - one-click Ubuntu installer (idempotent; safe to re-run). `run.sh` - what the service runs.

## This machine

- Ubuntu 26.04, i7-6700, 8 GB RAM, RTX 2070 SUPER (8 GB VRAM), NVIDIA driver 595.
- Project at `~/static_bot`. Python env: `.venv` is **Python 3.12 made with uv** (system Python is 3.14,
  which kokoro-onnx doesn't support). Install packages with
  `~/.local/bin/uv pip install --python .venv/bin/python ...` - there is no plain pip workflow.
- Services: `discord-voicebot` (the bot, runs `run.sh`) and `ollama`. LLM: `igorls/gemma-4-E4B-it-heretic-GGUF:Q4_K_M`
  (endpoint `gemma`; the previous `huihui_ai/qwen3.5-abliterated:4B` is still configured as `uncensored`).
- `config.yaml` holds the Discord bot token. Never print it, paste it into output, or commit it (it's gitignored).

## Operating it

- Which bots run: `.venv/bin/python -m voicebot.platform` (show) / `set both --restart`, or the dashboard's Platforms
  tab. In `discord` mode main.py still runs plain `bot.run()`; `fluxer`/`both` go through `main.run_multi`.
  Fluxer log lines: `Logged in to Fluxer as ...`, `Fluxer server: <name> (N members)`, `💬 Fluxer [...]`.

- Restart after code/config changes: `sudo systemctl restart discord-voicebot`
- Logs: `journalctl -u discord-voicebot -n 100 --no-pager` (🎙 = what it heard, 🤖 = what it said,
  `Latency:` lines per voice turn). Follow live with `-f`.
- VRAM: `nvidia-smi`; the LLM must show `100% GPU` in `ollama ps` or replies get several times slower.
- No unit tests, but `python -m voicebot.replay` checks prompts/model behaviour (addressee, calculator, search,
  replies). Verify changes by running it, restarting the service and checking the logs for a clean start
  (`Logged in as`, `Kokoro TTS loaded on GPU`, `LLM endpoint ... ready`, `Lore memory ready`) and a real exchange.

## Gotchas

- **`VoiceRecvClient.stop()` also stops receiving.** Use `vc.stop_playing()` to stop audio; calling `vc.stop()` on
  barge-in made the bot deaf until the watchdog restarted listening ("Voice receive stopped ... listening again").
- **Privileged intents:** `discord.members_intent` / presences need the matching switch in the Developer Portal, or
  the login fails with PrivilegedIntentsRequired. Turn it on in the portal first, then in config.
- **Whisper name hints** (`voice.stt_name_hints`) can make Whisper loop a hinted word on unclear audio; the STT drops
  transcripts with a high compression ratio or one word/phrase filling most of them (`stt._is_looping`).
- **onnxruntime**: kokoro-onnx pulls in CPU `onnxruntime`, which shares a package folder with
  `onnxruntime-gpu`. After any `uv pip install -r requirements.txt`, redo the swap (uninstall
  `onnxruntime`, `--reinstall --no-deps onnxruntime-gpu`) or TTS silently falls back to CPU.
  Re-running `bash install.sh` handles this.
- **Voice receive / DAVE**: Discord end-to-end encrypts voice. Receiving needs zacker150's forks of
  `discord-ext-voice-recv` + `discord.py`, pinned by commit in `requirements.txt`. Don't swap back to PyPI versions.
- **Latency**: the system prompt must stay stable between turns (Ollama reuses the cached prompt prefix).
  Don't put per-turn changing values (timestamps, counters) in it; history is trimmed in chunks for the same reason.
- **One GPU for everything**: Whisper + Kokoro (~2 GB) + the LLM share 8 GB. Keep LLMs under ~4.5 GB on disk
  (Gemma E-models are the exception: per-layer embeddings stay in RAM, so check `ollama ps`/`nvidia-smi`).
- The model is small (4B); fix behaviour through prompts in `config.yaml`/`DEFAULTS` and the text filters,
  and keep replies short for voice. Don't quote the bad output in a prompt ("not vague stuff like 'you're
  predictable'") - it copies it. Few-shot examples work better than rules.
- large-v3-turbo's `no_speech_prob` is always ~0: short transcripts are checked with Silero VAD instead
  (`stt.faster_whisper.short_*`; drops logged "dropped a doubtful short transcript").
- **Fluxer**: GUILD_CREATE only carries members in voice (+ the bot): `FluxerBot._load_members` fetches the full list
  over REST, and Fluxer rejects `after=0` (400 Invalid form body) - only send `after` when paging. LiveKit sends open
  mics continuously; the per-speaker noise gate in `fluxer/voice.py` stands in for Discord's client-side gating.
  Fluxer text chat is a copy of Discord's `on_message` (fluxer/bot.py): mirror chat changes there.
- **Addressee check**: if the bot's last line asked a question, the next line counts as for it with no LLM call
  (the 4B model called plain answers "NO"). Add a replay case for every miss (`evals/cases.yaml`, kind addressee).
- Every extra LLM call (search check, addressee, scene, lore, profiles) shares one Ollama slot
  (`OLLAMA_NUM_PARALLEL=1`): background ones must pause on live activity (`profiles.activity()`).
