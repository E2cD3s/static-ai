# Discord + Fluxer Voice LLM Bot (fully local)

Talk to an LLM live in a Discord voice channel, and optionally on a self-hosted [Fluxer](https://fluxer.app)
server too (see [Fluxer](#fluxer-optional-second-platform)). Everything runs on-prem:

```
Discord voice ──► per-user VAD ──► faster-whisper (GPU) ──► Ollama / LM Studio (streaming)
                                                                   │ sentence-by-sentence
Discord voice ◄── streaming playback ◄── Kokoro / Piper TTS ◄──────┘
```

**Why it's fast**
- Each speaker is segmented separately with WebRTC VAD. The turn ends after `silence_ms` of quiet.
- **Speculative STT:** transcription starts 250 ms into a pause, so the text is usually ready the moment your turn ends. If you keep talking, that result is thrown away.
- STT runs in-process on the GPU (`large-v3-turbo`, greedy decoding, trailing silence trimmed).
- The LLM reply is streamed and cut into sentences. The first chunk is cut at the first comma, so audio starts before the first sentence is finished. It's synthesized and playing while the rest is still being generated.
- Kokoro TTS runs on the GPU. cuDNN's per-shape benchmarking is disabled so new sentence lengths never stall, and several lengths are warmed up at startup.
- STT and TTS run on separate threads, so they never block each other or the Discord event loop.
- All models (and the LLM) are loaded and warmed up at startup, so the first reply isn't slow.
- Barge-in: when the person it's answering talks over it for about a second, it stops speaking and cancels the LLM request.
- `<think>` blocks from reasoning models are stripped before anything is spoken.

Each turn is logged with its latency:
`Latency: stt=120ms llm_ttft=180ms tts_first=90ms | end-of-speech -> audio=1010ms`
(this total includes the `silence_ms` wait).

## Setup (Windows)

1. Install **Python 3.12** from python.org and tick *Add to PATH*.
2. Download the code (`git clone https://github.com/E2cD3s/static-ai.git`, or *Code → Download ZIP* on GitHub), then run `setup.bat`. It creates `.venv`, installs the dependencies and CUDA libraries, and downloads Kokoro.
3. Create the bot at https://discord.com/developers/applications:
   - **Bot** tab → Reset Token → paste it into `config.yaml`.
   - **Bot** tab → enable **Message Content Intent**.
   - **OAuth2 → URL Generator** → scopes `bot` and `applications.commands`. Permissions: View Channels, Send Messages, Read Message History, Connect, Speak, Use Voice Activity. Open the generated URL to invite the bot.
4. Edit `config.yaml`: add the token, your LLM endpoints, and your server ID under `guild_ids` so slash commands appear instantly.
5. Run `run.bat`.

## Setup (Ubuntu, one click, recommended for a dedicated box)

1. Install **Ubuntu Server 24.04 LTS** on the machine.
2. Create the bot at https://discord.com/developers/applications: **New Application** → **Bot** → **Reset Token**, and copy the token.
3. Get the code onto the machine and run the installer:
   ```bash
   git clone https://github.com/E2cD3s/static-ai.git static_bot
   cd static_bot
   bash install.sh
   ```

The installer asks for the bot token, your server ID, a bot name and your time zone. It checks the token with Discord and tells you if Message Content Intent still needs turning on. Then it does the rest by itself:

- system packages and the NVIDIA driver (if a driver is needed, it reboots and carries on automatically; follow along with `tail -f install.log`)
- the Python environment, with GPU speech models (Whisper + Kokoro)
- Ollama, tuned for a shared 8 GB GPU, plus the uncensored LLM
- `config.yaml` and a service that starts the bot on every boot

At the end, it prints the link for inviting the bot to your server. Re-running it is safe.

- Live logs: `journalctl -u discord-voicebot -f`
- After editing `config.yaml`: `sudo systemctl restart discord-voicebot`

### The model

`huihui_ai/qwen3.5-abliterated:4B` is an uncensored ("abliterated") build of Qwen 3.5 4B, released March 2026 and 3.3 GB at Q4. It's the newest model that fits the GPU alongside the speech models while staying fully on the GPU. Its "thinking" mode is turned off in the config (`reasoning_effort: "none"`), which saves seconds per reply.

To try another model, `ollama pull` it and add an endpoint in `config.yaml` (there's an example), then `/llm <name>`. Keep model files under about 4.5 GB on an 8 GB card. Gemma 4 E4B (e.g. a "heretic" GGUF build) also works well and is fast; its per-layer embeddings stay in system RAM, so check `ollama ps` / `nvidia-smi` rather than the file size.

### Everything on one 8 GB GPU (e.g. an RTX 2070)

| Component | Runs on | Approx. VRAM |
|---|---|---|
| Bot process CUDA context (shared by Whisper and Kokoro) | GPU | ~0.4 GB |
| Whisper large-v3-turbo (`int8_float16`) | GPU | ~1.1 GB |
| Kokoro TTS (onnxruntime-gpu) | GPU | ~0.6 GB |
| qwen3.5-abliterated 4B Q4 + 6k context | GPU | ~4 GB |

The bot loads Whisper and Kokoro first and warms up the LLM last, so Ollama sizes itself to the VRAM that's left. Once it's running, check `ollama ps`: the PROCESSOR column must say **100% GPU**. If it shows any CPU percentage, the model didn't fit, and replies will be several times slower.

At startup, the log should say `Kokoro TTS loaded on GPU`.

## Commands

| Command | | Who |
|---|---|---|
| `/join` / `/leave` | Join or leave your voice channel | everyone |
| `/stop` | Make the bot stop talking | everyone |
| `/clip [seconds]` | Post the last few seconds of the call as an MP3 | everyone |
| `/quote [search] [delete]` | A random quote from the server's quote book, search it, or delete one | everyone (delete: admins or whoever saved it) |
| *Save as quote* (message menu) | Right-click a message > Apps > Save as quote | everyone |
| `/remind` / `/reminders` / `/poll` | Reminders (DM or group ping) and Discord polls | everyone |
| `/status` | LLM speed and tokens, GPU, latency, pipeline | everyone |
| `/help` | How to talk to it, what it can do, the commands | everyone |
| `/profile [user]` | What the bot remembers about you | everyone (others: admins) |
| `/forget [user]` | Delete a stored profile | everyone (others: admins) |
| `/profiling enabled` | Opt out of (or back into) profiling | everyone, for themselves |
| `/link [code]` / `/unlink` | Link (or unlink) your Discord and Fluxer accounts | everyone, for themselves |
| `/say text` | Make the bot say something in voice | server admins |
| `/reset` | Clear the conversation memory in a channel | server admins (anyone in DMs) |
| `/tuning [reset]` | What it learned about reply length / follow-ups | server admins |
| `/llm endpoint [model] [scope]` / `/models` | Switch or list LLMs while running | owner (`bot.creator_ids`, `dashboard.admin_ids`) |

"Server admins" means Manage Server or Administrator. Those commands are also hidden from everyone else in Discord's
command list (server owners can change that under Server Settings > Integrations; the bot checks again either way).

Most things also work by just asking, in voice or text: "Static, remind me in 20 minutes to…", "make a poll: pizza or
tacos", "clip that", "quote that", "what roles does Casey have?", "Static, roast jordan" (aimed with what it knows about
them, what they're playing and their recent lines), "flip a coin", "roll 2d6", "split us into two teams", "what's the
weather in Denver tomorrow?", "Static, you can leave now".

In text channels, the bot answers @mentions, replies to its messages, messages that say its name, threads it's in,
DMs, and every message in `text_channel_ids`. @mentions inside a message are passed to the model as names.

## Quote book, game night, weather

- **Quote book** (`quotes:`): "Static, quote that" (or a bare "quote that") saves the last line someone else said in
  voice; "quote what riley said" / "quote me" pick the line; it's posted in the chat. "Static, give us a random quote"
  reads one out. `/quote` shows a random one or searches, and the *Save as quote* message menu saves a text message.
  Each platform has its own book (`data/quotes.db`, Fluxer's in `data/fluxer/`). `/forget` doesn't remove quotes
  (they're the group's); admins or whoever saved one can delete it.
- **Game night** (`fun:`): coin flips, dice (`d20`, `2d6+3`, "three dice"), a random number, "pick someone" / "who goes
  first", "pick between tacos, pizza or sushi", "split us into two teams" (from the people in the call). Worked out in
  Python with real randomness and handed to the model as a note, so it announces the actual result (`voicebot/fun.py`).
- **Weather** (`weather:`): "what's the weather in Denver tomorrow?", "is it gonna rain in Chicago today?". Live
  forecasts from [Open-Meteo](https://open-meteo.com) (free, no key; only the place name leaves the machine), today to
  a week out. `weather.units` (imperial/metric), and `weather.default_location` for questions that don't name a place.

## LLM server tips

- **Ollama:** `base_url` must end in `/v1`. On the server, set `OLLAMA_KEEP_ALIVE=-1` so the model stays loaded, `OLLAMA_HOST=0.0.0.0` so it's reachable over the LAN, and `OLLAMA_FLASH_ATTENTION=1`.
- **LM Studio:** start the server (Developer tab) with *Serve on local network* turned on. `model` must match the identifier LM Studio shows; `/models` lists them.
- Voice needs a fast time-to-first-token. 3B–8B models (llama3.2:3b, llama3.1:8b, qwen2.5:7b, gemma3) feel conversational. You can set `voice_endpoint` to a small model for voice and keep a larger one for text.
- For reasoning models, turn thinking off. On Ollama, add `extra_body: {reasoning_effort: "none"}` to the endpoint. The bot strips `<think>` blocks, but the model still spends time generating them.

## Natural conversation

- **Persona:** `bot.system_prompt` is a character, not an assistant. Rewrite it freely; it has more effect on how natural the bot feels than anything else.
- **Adaptive turn-taking:** during a pause, the bot looks at your partial transcript. "What do you think?" gets an answer after 350 ms. "So I was going to the, um..." gets up to 1.2 s so you aren't cut off.
- **Overlap:** talk over the bot and it drops its volume right away. Only after about a second of real talking
  (`voice.barge_in_ms`) does it stop, and only for the person it's answering or already talking with
  (`voice.barge_in_scope`); anyone else just turns it down. Listener noises like "yeah", "mhm", or "haha" don't interrupt
  it and don't get a reply. Cut it off by accident? "Go on" / "what were you saying?" within `voice.resume_after_cut_s`
  makes it finish the thought.
- **Follow-ups:** after it answers you, you can keep talking without its name for `wake_word_followup_s` (self-tuned
  per server, 12 s minimum). With `voice.followup_scope: speaker` that window is only for the people it's talking with;
  others say its name to join in. A quick YES/NO LLM check (`voice.addressee`) skips follow-ups that were really said
  to someone else ("is she roasting you?"); an answer to a question it just asked always counts as for it.
- **Leaving:** "Static, leave" / "you can go ahead and leave" makes it say bye and go (`voice.leave_on_request`).
  Force-disconnect it instead (right-click > Disconnect) and it comes straight back, talks some shit to whoever did it
  (read from the audit log, so give it *View Audit Log*), then leaves on its own (`voice.boomerang`). A second kick
  within 10 minutes and it stays gone.
- **Checking in:** when nobody's talking to it but someone sounds genuinely upset for a while (the mood reader, plus a
  louder-than-usual voice), it chimes in once to check on them, working in what they're playing or just said.
  Rare on purpose: see `sentiment.check_in` for the thresholds and cooldowns.
- **Awareness:** the bot knows the day, the time of day, and who's in the channel. It greets people when it joins and notices when they come and go.
- **Roleplay-model cleanup:** `*action text*` is never spoken (names, quotes and emphasis in asterisks are kept),
  "Static:" labels are stripped, a copy of the message it's answering at the start of a reply is dropped, and
  generation stops if the model starts writing another person's lines.
- **Hearing names right:** Whisper is told the names in play (people in the call, the games they're playing, names the
  bot just said) so "Arknights: Endfield" doesn't come out as "Arknight's infield" (`voice.stt_name_hints`). Transcripts
  where Whisper gets stuck repeating a word are dropped.
- **Search follow-ups:** the last web search results stay available for `search.remember_s`, so "who's that?" about
  something it just found works.
- **Time zone:** set the server's time zone so the time of day is right: `sudo timedatectl set-timezone America/New_York`.

## Knowing people

- **Identity:** every reply knows who it's talking to: display name, @username, roles (highest first), and whether they're the server owner or an admin. In voice, that covers everyone in the channel. It also knows its own roles.
- **Member lookups:** "what roles does Casey have?", "list my roles", "who is X in this server?" look the member up in
  the server (never the web), including people not in the call and fancy-font names. That needs `discord.members_intent`
  and the *Server Members Intent* switched on in the Developer Portal (turn it on in the portal first, or login fails).
- **Profiles:** each person gets long-term notes (who they are, interests, opinions, what's going on in their life, running jokes, how they like to be talked to), stored in `data/profiles.db`.
  - After someone has said about 8 new things, the LLM rewrites their notes from the old notes plus the new conversation.
  - Updates only run once the chat has been quiet for 45 s. Anyone speaking cancels an in-flight update, so profiles never slow down a live reply.
  - The notes go into the prompt the next time that person talks. A roleplay-model conversation might produce: *"plays Valorant most nights; night-shift nurse; roasts Static about its taste in music"*.
- **Privacy controls:**
  - `/profile` shows a person what the bot has stored about them.
  - `/forget` deletes it.
  - `/profiling enabled:False` opts them out and deletes what was stored.
  - Server managers can use `/profile` and `/forget` on anyone.

  Tell your server about these commands. People should know the bot remembers them, and Discord's developer policy expects bots that store user data to say so.
- **Linked accounts** (`links:`, both platforms running): someone can link their Discord and Fluxer accounts - `/link`
  gives a 6-digit code to type as `!link <code>` on Fluxer within 10 minutes (or the other way round). Each bot then
  knows it's the same person and adds the other bot's notes on them to their roster line; nothing else is shared.
  `/unlink`, `!unlink`, `/forget` (either side) and the dashboard's Forget remove the link. Stored in `data/links.db`,
  the one file both bots share.

## Fluxer (optional second platform)

[Fluxer](https://fluxer.app) is an open-source, self-hostable Discord-like chat app with voice over LiveKit. The same
bot can run there too, in the same process, sharing the models (so it costs no extra VRAM) but keeping its own people
data. `platform.mode` picks what runs: `discord` (default), `fluxer`, or `both`.

**Set up**
1. On your Fluxer server, create a bot application and copy its token. Voice needs LiveKit enabled on the Fluxer server.
2. Invite the bot to your Fluxer server (the invite link in Fluxer's bot settings, or
   `<fluxer>/oauth2/authorize?client_id=<app id>&scope=bot&permissions=3214336`).
3. In `config.yaml`, fill the `fluxer:` section: `api_url` (`https://your.fluxer.host/api/v1`), `token`, and your Fluxer
   user id in `creator_ids` (see `config.example.yaml` for the rest).
4. Switch: `.venv/bin/python -m voicebot.platform set both --restart`, or the **Platforms** tab in the dashboard.
   The switch checks the config first (it refuses `fluxer` without a token).

**How it differs from Discord**
- Fluxer has no bot slash commands yet, so commands use a prefix (`fluxer.prefix`, default `!`): `!join` `!leave`
  `!stop` `!clip` `!quote` `!remind` `!reminders [cancel <id>]` `!poll question | a, b, c | 10 minutes` `!profile`
  `!forget` `!profiling on|off` `!link` `!unlink` `!status` `!help`; admins `!say` `!reset` `!tuning`; owner `!llm` `!models`.
  Everything you can ask out loud or in chat works the same.
- Answers that are private (ephemeral) on Discord arrive by DM. Polls are reaction polls (1️⃣ 2️⃣ …), counted when they
  close, posted and announced in voice. Voice channels have no chat, so clips/quotes/reminders set in voice post in the
  channel `!join` was typed in.
- Its data lives in `fluxer.data_dir` (`data/fluxer/`: profiles, memories, mood, tuning, reminders, quotes, muted
  servers). Linked accounts are the only thing shared.
- LiveKit sends every open mic continuously (Discord only sends while you talk), so each person gets a noise gate
  (`voicebot/fluxer/voice.py`) - without it the bot never gets a turn to speak.
- If the Fluxer connection drops, it reconnects by itself; a Fluxer problem never takes the Discord bot down.

## Website

The bot serves a small site (`dashboard:` in config) at `dashboard.public_url`:

- `/status` - public live stats (no server or channel names; Discord/Fluxer lights, voice calls tagged by platform).
  `/help` - the user guide, built from the live config and command list, with a Discord / Fluxer switch when both
  run (`/help?p=fluxer` opens on Fluxer's `!` commands). `/terms` and `/privacy` - terms of service and privacy policy (contact: `dashboard.contact`,
  effective date: `dashboard.legal_updated`).
- `/linked-role` - Discord's *Linked Roles Verification URL*. Servers can then require "has talked to Static", "times
  talked" or "days known". Needs `dashboard.discord_client_secret` and `<public_url>/auth/linked-role` as an OAuth2 redirect.
- `/admin` - Overview, **Platforms** (turn the Discord / Fluxer bot on or off, apply with a restart), Live, Servers,
  People, Settings (writes `config.yaml`, keeps comments; grouped into Platforms / Discord only / Fluxer only / Shared),
  Logs, restart. With both bots on, Servers, Live and People get a Discord / Fluxer switch. Log in with Discord
  (`dashboard.discord_client_secret` + `<public_url>/auth/callback` as a redirect) or a local password
  (`python -m voicebot.web set-password`).

The Developer Portal fields: Terms of Service URL `<public_url>/terms`, Privacy Policy URL `<public_url>/privacy`,
Linked Roles Verification URL `<public_url>/linked-role` (optional).

**Your own theme:** every page loads `voicebot/web/static/local.css` after `style.css`, if it exists. Put CSS there to
re-theme your install (the easiest start is overriding the colour and font variables in `:root` at the top of
`style.css`). It's git-ignored, so it survives updates and stays out of commits. Changes show on the next page reload,
no restart needed. The page's Content-Security-Policy allows fonts from Google Fonts and images from the site itself
or `data:` URIs.

## Tuning

- **Snappier replies:** lower `voice.silence_short_ms` and `silence_ms`. Too low and it will cut people off.
- **It cuts you off mid-thought:** raise `silence_long_ms` to 1500+.
- **Background noise triggers it:** raise `vad_aggressiveness` to 3 or `min_speech_ms` to 400.
- **Group calls:** set `response_mode: wake_word` so the bot only answers when addressed by name.
- **Short on VRAM:** `compute_type: int8_float16`, or `model: distil-large-v3` / `small.en`.
- **No usable GPU for TTS:** use `tts.backend: piper` (`pip install piper-tts`, then `python download_models.py piper`). It's the fastest option on CPU.
- **STT/TTS on another machine:** set the `openai` backends. Any OpenAI-compatible server works, e.g. speaches for STT or Kokoro-FastAPI for TTS.

## Troubleshooting

- **The bot joins and `/say` works, but it never hears anyone.** Since March 2026, Discord end-to-end encrypts all voice (DAVE). The PyPI `discord-ext-voice-recv` can't decrypt it, so `requirements.txt` pins zacker150's fork, which can. Make sure you're on the current `requirements.txt` and re-run `bash install.sh`. In wake-word mode, also check that you're saying its name; the log shows every transcript (🎙).
- **"Could not load Whisper on cuda".** The pip CUDA libraries didn't load. Update your NVIDIA driver, or reinstall `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`. Until then, the bot falls back to the CPU.
- **Whisper hallucinations ("Thank you.") on noise:** these are filtered already. Raise `min_speech_ms` if you still get them.
- **Fluxer: `!join` does nothing / "Fluxer didn't answer the voice join".** The bot isn't in that server, or lacks
  Connect/Speak in the channel. Check the Servers tab in the dashboard (Fluxer group).
- **Fluxer: it hears people but never answers / waits forever.** Someone's mic noise is getting past the noise gate:
  raise `GATE_OPEN_DB` in `voicebot/fluxer/voice.py`, or have them use push-to-talk / noise suppression.
- **Fluxer: `Gateway connection error ... Reconnecting in 5s` in the log.** Normal at boot if Fluxer starts slower
  than the bot; it keeps retrying.

## License

MIT - see [LICENSE](LICENSE).
