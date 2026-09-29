# Static on Fluxer - plan (done)

**Status (2026-09-26): shipped.** Running in `platform.mode: both`. All phases below are done, plus: feature parity
(every command as a `!` command, reaction polls, images/avatars, boomerang), the dashboard Platforms tab (turn each
bot on/off) with Discord/Fluxer separation across Servers/Live/People/Settings, and the shared extras (quote book,
game-night helpers, weather, opt-in account linking). User-facing docs: README "Fluxer" section and the /help page;
developer notes: CLAUDE.md. Next idea under discussion: an opt-in Discord <-> Fluxer chat-channel bridge.

The original plan, kept for the record:

Goal: a second Static that lives on the self-hosted Fluxer server (https://chat.example.com), does the same
things as the Discord bot (voice + text), and cannot break the Discord bot.

## What the voice test proved (2026-09-26, `~/fluxer_test/voice_test.py`)

- Bot logs in over Fluxer's gateway (`/api/v1`, `wss://chat.example.com/gateway`), joins voice via
  VOICE_STATE_UPDATE -> VOICE_SERVER_UPDATE (LiveKit endpoint + token), connects to `sfu.example.com` in ~1 s.
- Voice is **not** E2E encrypted. Each person arrives as their **own audio track** (48 kHz) - exactly what the
  per-user VAD needs, no forked libraries.
- Publishing audio works (beep heard in the channel).
- `fluxer.py 0.4.2` + `livekit 1.1.20` install into the bot's `.venv` cleanly (dry run: 5 new packages, nothing
  existing upgraded or downgraded).

## Key decision: one process, two frontends

Today: 6.1 of 8 GB VRAM used (Ollama 4.45 GB, bot 1.7 GB), ~2.5 GB RAM available. A second bot process would load
its own Whisper + Kokoro + ONNX models (~1.7 GB VRAM + CUDA context, ~1.5 GB RAM) - it doesn't fit.

So the existing process gets a second, independent frontend:

```
                 ┌── Discord frontend (unchanged: bot.py, slash commands, voice_recv)
shared core ─────┤
(Whisper, Kokoro,└── Fluxer frontend (new: voicebot/fluxer/, prefix commands, LiveKit voice)
 LLM router, text
 filters, mood, lore, reminders, calc, search ...)
```

Isolation rules (so Fluxer can't hurt Discord):
- `platform.mode: discord` by default - the bot runs exactly the code path it runs today.
- The Fluxer frontend runs as one supervised asyncio task: any exception is logged and it reconnects with backoff.
  It never touches the Discord client, never calls `close()`, never exits the process.
- If `fluxer.py`/`livekit` fail to import, log once and skip Fluxer.
- Fluxer data kept separate (own guild ids; profiles/lore/mood/tuning are already keyed per guild/user).
- Live voice on either platform counts as "activity" so background LLM jobs pause for both
  (Ollama has one slot; a reply on one platform can briefly queue behind the other).

## Phases (each ends with a check before moving on)

### 0. Safety net
- `git init` + baseline commit (config.yaml already gitignored) so any step can be rolled back in seconds.
- Run `python -m voicebot.replay` and save the baseline result.

### 1. Voice transport seam - Discord only, no behaviour change  *(the only step that edits Discord code)*
- `VoiceSession` currently takes a discord `VoiceRecvClient`, gets audio from `VoiceSink.write(user, pcm)` and
  plays a `StreamingPCMSource`. Put a small `VoiceTransport` interface between them:
  start/stop listening -> `on_audio(user, pcm48)`, `play(source)`, `stop_playing()`, `is_playing()`,
  `humans_in_channel()`, member info.
- `DiscordTransport` wraps the existing code 1:1 (keeps the `stop_playing()` not `stop()` gotcha, watchdog, DAVE).
- A tiny `Person` shape (id, name, is_bot, activities) where VoiceSession/profiles read discord user objects.
- Check: replay evals match baseline, restart when nobody's in a call, one real Discord voice exchange,
  clean start in the logs.

### 2. Fluxer text chat
- `voicebot/fluxer/client.py` - gateway + REST via `fluxer.py` (vendor/patch it if it's missing something).
- `voicebot/fluxer/chat.py` - mentions / name / replies -> same LLM, filters, search, calc, reminders,
  profiles, lore as Discord text chat (calls the shared modules; the Discord `on_message` is left alone).
- Prefix commands (Fluxer has no bot slash commands yet): `!join !leave !stop !help !status !remind
  !reminders !poll !clip !profile !reset` (admins: `!say !llm !tuning`). Voice requests work as today.
- Config: `fluxer:` section in `DEFAULTS` + `config.example.yaml` (enabled, api_url, token, prefix, admin ids,
  allowed guilds). Token lives only in `config.yaml`.
- Check: chat with it on Fluxer; Discord unaffected.

### 3. Fluxer voice
- `LiveKitTransport`: one `AudioStream` per remote participant -> 20 ms 48 kHz frames -> `on_audio`;
  playback task reads the same `StreamingPCMSource` every 20 ms -> `AudioSource.capture_frame`.
  Barge-in/ducking, clips tap, backchannel, speculative STT all reuse VoiceSession unchanged.
- Map LiveKit identities (`user_<id>_<session>`) to Fluxer users; follow joins/leaves; auto-leave when empty;
  reconnect when the 10-min LiveKit token/room drops.
- Check: real conversation on Fluxer, then Discord + Fluxer calls at the same time (watch latency lines).

### 4. Extras (as wanted)
- Reminder/poll delivery on Fluxer (DMs, channel pings; polls via reactions if Fluxer has no native polls).
- `!status` / `!help` output, dashboard shows Fluxer servers + voice, presence line, images (vision).
- A replay/eval case or two for Fluxer-specific text handling.

## Not coming over (for now)
- Slash commands (until Fluxer ships them - then swap prefix commands over).
- Discord Linked Roles, Spotify/activity-based profile notes unless Fluxer exposes presences.

## Decisions (2026-09-26)
- **Everything separated** per platform: memories, profiles, lore, mood, tuning, reminders, muted list, logs labels.
- **Same persona** (name, personality, prompts) on both.
- **Platform mode switch**: `platform.mode: discord | fluxer | both` - switchable from the CLI
  (`python -m voicebot.platform set <mode>`) and the web panel; applied with a restart (exit 75).
  Both at once works because the models are shared in one process.
- Prefix `!` (default, configurable). Fluxer admins: server owner + configured ids (still need the user's Fluxer id).
