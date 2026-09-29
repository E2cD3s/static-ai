#!/usr/bin/env bash
# =============================================================================================
#  One-click installer: Discord voice LLM bot on Ubuntu + NVIDIA GPU
#
#      bash install.sh
#
#  Asks a few quick questions (Discord token, server ID, bot name, time zone), then does
#  everything else by itself:
#    - system packages + NVIDIA driver (if one is needed it reboots and carries on automatically)
#    - Python environment, Whisper (speech-to-text) and Kokoro (text-to-speech), both on the GPU
#    - Ollama, tuned for a shared 8GB GPU, plus the uncensored LLM
#    - config.yaml, time zone, and a service that starts the bot on every boot
#  Safe to re-run: finished steps are skipped or refreshed.
# =============================================================================================
set -euo pipefail

# Uncensored ("abliterated") Qwen 3.5 4B - 3.3GB, fits the GPU next to Whisper + Kokoro.
MODEL="huihui_ai/qwen3.5-abliterated:4B"
SERVICE="discord-voicebot"
RESUME_UNIT="voicebot-install-resume"
# Bot invite permissions: View Channels, Send Messages, Read Message History, Connect, Speak, Use Voice Activity
PERMS=36768768

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
STATE="$DIR/.install_state"
LOG="$DIR/install.log"
RESUME=0
[ "${1:-}" = "--resume" ] && RESUME=1

exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n    Full log: %s\n' "$*" "$LOG"; exit 1; }

# ---------------------------------------------------------------------------------- who runs what
# Phase 1 runs as your user (sudo for system steps). After a driver reboot, phase 2 runs from a
# one-shot systemd unit as root, so user-owned steps drop back to your user via as_user.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=()
    # shellcheck disable=SC1090
    [ -f "$STATE" ] && . "$STATE"
    INSTALL_USER="${INSTALL_USER:-${SUDO_USER:-}}"
    if [ -z "$INSTALL_USER" ] || [ "$INSTALL_USER" = "root" ]; then
        die "Run this as your normal user, not root:  bash install.sh"
    fi
else
    SUDO=(sudo)
    INSTALL_USER="$USER"
fi
as_user() { if [ "$(id -u)" -eq 0 ]; then sudo -u "$INSTALL_USER" -H "$@"; else "$@"; fi; }
as_root() { "${SUDO[@]}" "$@"; }

save_state() {
    {
        printf 'INSTALL_USER=%q\n' "$INSTALL_USER"
        printf 'TZ_CHOICE=%q\n' "${TZ_CHOICE:-}"
        printf 'APP_ID=%q\n' "${APP_ID:-}"
    } > "$STATE"
}

# ---------------------------------------------------------------------------------- Discord checks
discord_check() {  # prints: OK <app_id> <has_message_content_intent 0|1> <bot username>
    DISCORD_TOKEN="$1" python3 - <<'PY'
import json, os, sys, urllib.request
tok = os.environ["DISCORD_TOKEN"]
def get(path):
    req = urllib.request.Request("https://discord.com/api/v10" + path, headers={
        "Authorization": "Bot " + tok, "User-Agent": "DiscordBot (voicebot-installer, 1.0)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)
try:
    me = get("/users/@me")
except Exception as e:
    print(f"FAIL {e}")
    sys.exit(1)
try:
    flags = get("/applications/@me").get("flags", 0)
except Exception:
    flags = 0
intent = 1 if flags & ((1 << 18) | (1 << 19)) else 0   # GATEWAY_MESSAGE_CONTENT (or _LIMITED)
print(f"OK {me['id']} {intent} {me['username']}")
PY
}

ask_questions() {
    say "Quick setup questions (everything after this is automatic)"
    echo "    Bot token: https://discord.com/developers/applications -> New Application -> Bot -> Reset Token"
    while true; do
        read -rsp "    Paste your bot token (input hidden): " TOKEN; echo
        [ -z "$TOKEN" ] && continue
        if RESULT="$(discord_check "$TOKEN")"; then
            read -r _ APP_ID HAS_INTENT BOT_USER <<< "$RESULT"
            ok "Token works - bot account: $BOT_USER"
            break
        fi
        warn "Discord rejected that token (or no internet). Try again."
    done

    while [ "$HAS_INTENT" != "1" ]; do
        warn "The bot needs MESSAGE CONTENT INTENT turned on, or it can't log in."
        echo "    Open: https://discord.com/developers/applications/$APP_ID/bot"
        echo "    Scroll to 'Privileged Gateway Intents', enable MESSAGE CONTENT INTENT, click Save."
        read -rp "    Press Enter when done (or type s to skip): " ans
        [ "$ans" = "s" ] && break
        read -r _ _ HAS_INTENT _ <<< "$(discord_check "$TOKEN" || echo "FAIL 0 0 x")"
    done
    [ "$HAS_INTENT" = "1" ] && ok "Message Content intent is on"

    while true; do
        read -rp "    Server ID (right-click your server -> Copy Server ID; Enter to skip): " GUILD_ID
        [[ -z "$GUILD_ID" || "$GUILD_ID" =~ ^[0-9]{15,21}$ ]] && break
        warn "That doesn't look like a server ID (a long number). Enable Developer Mode in Discord settings -> Advanced."
    done

    read -rp "    Bot name [Static]: " BOT_NAME
    BOT_NAME="${BOT_NAME:-Static}"

    local cur_tz
    cur_tz="$(timedatectl show -p Timezone --value 2>/dev/null || echo Etc/UTC)"
    while true; do
        read -rp "    Your time zone, e.g. America/New_York [$cur_tz]: " TZ_CHOICE
        TZ_CHOICE="${TZ_CHOICE:-$cur_tz}"
        timedatectl list-timezones 2>/dev/null | grep -qx "$TZ_CHOICE" && break
        warn "Unknown time zone '$TZ_CHOICE' (list them with: timedatectl list-timezones)"
    done
}

write_config() {
    TOKEN="$TOKEN" GUILD_ID="$GUILD_ID" BOT_NAME="$BOT_NAME" python3 - <<'PY'
import os, re
env = os.environ
t = open("config.example.yaml", encoding="utf-8").read()
t = t.replace("PASTE_YOUR_BOT_TOKEN_HERE", env["TOKEN"])
if env["GUILD_ID"]:
    t = re.sub(r"^(  guild_ids: )\[\]", lambda m: m.group(1) + "[" + env["GUILD_ID"] + "]", t, count=1, flags=re.M)
name = env["BOT_NAME"].replace('"', "").strip() or "Static"
t = re.sub(r'^(  name: )".*?"', lambda m: f'{m.group(1)}"{name}"', t, count=1, flags=re.M)
if name.lower() != "static":  # the example already lists "static" plus its common mishearings
    t = re.sub(r"^(  wake_words: )\[.*?\]", lambda m: f'{m.group(1)}["{name.lower()}"]', t, count=1, flags=re.M)
with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(t)
PY
    chmod 600 config.yaml  # contains your token
    ok "Wrote config.yaml (token stored there, readable only by you)"
}

# ================================================================================ PHASE 1 (interactive)
if [ "$RESUME" -eq 0 ]; then
    printf '\n\033[1mDiscord Voice LLM Bot - one-click install\033[0m  (log: %s)\n' "$LOG"
    . /etc/os-release
    [ "${ID:-}" = "ubuntu" ] || warn "Made for Ubuntu (found ${PRETTY_NAME:-unknown}) - continuing anyway"

    say "Getting sudo (your password, once)"
    sudo -v || die "sudo is required"
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true; sleep 45; done ) 2>/dev/null &
    SUDO_KEEPALIVE=$!
    trap 'kill "$SUDO_KEEPALIVE" 2>/dev/null || true' EXIT

    if [ ! -f config.yaml ] || grep -q "PASTE_YOUR_BOT_TOKEN_HERE" config.yaml; then
        ask_questions
        write_config
    else
        ok "Using existing config.yaml"
        TZ_CHOICE="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
        TOKEN="$(python3 -c "import re;print(re.search(r'^  token: \"(.*)\"', open('config.yaml').read(), re.M).group(1))" 2>/dev/null || true)"
        [ -n "$TOKEN" ] && read -r _ APP_ID _ _ <<< "$(discord_check "$TOKEN" || echo "FAIL 0 0 x")"
    fi
    save_state

    say "Installing system packages"
    as_root apt-get update -y
    as_root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y \
        python3 python3-venv python3-pip libopus0 curl ca-certificates pciutils ubuntu-drivers-common git
    ok "System packages installed"

    if [ -n "${TZ_CHOICE:-}" ]; then
        as_root timedatectl set-timezone "$TZ_CHOICE" && ok "Time zone: $TZ_CHOICE"
    fi

    lspci | grep -qi nvidia || die "No NVIDIA GPU detected (lspci shows none)."

    if ! nvidia-smi >/dev/null 2>&1; then
        say "Installing the NVIDIA driver"
        if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then
            warn "Secure Boot is on. Ubuntu's signed NVIDIA driver normally works with it; if the GPU"
            warn "isn't detected after the reboot, turn Secure Boot off in the BIOS and re-run install.sh."
        fi
        # Non-interactive: no hidden debconf/needrestart questions (output goes through the log pipe,
        # which can hide prompts and make the install look frozen).
        as_root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a ubuntu-drivers install
        ok "Driver installed - a reboot is needed to load it"

        as_root tee "/etc/systemd/system/$RESUME_UNIT.service" >/dev/null <<EOF
[Unit]
Description=Finish Discord voice bot install after the NVIDIA driver reboot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/bash "$DIR/install.sh" --resume
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        as_root systemctl daemon-reload
        as_root systemctl enable "$RESUME_UNIT" >/dev/null 2>&1
        say "Rebooting in 20 seconds. Setup continues automatically after the reboot."
        echo "    Follow along after logging back in:  tail -f \"$LOG\""
        echo "    (Ctrl+C now to reboot later yourself - it will still continue on next boot.)"
        sleep 20
        as_root reboot
        exit 0
    fi
fi

# ================================================================================ PHASE 2 (automatic)
if [ "$RESUME" -eq 1 ]; then
    printf '\n\033[1m--- Resuming install after reboot (%s) ---\033[0m\n' "$(date)"
    systemctl disable "$RESUME_UNIT" >/dev/null 2>&1 || true
    rm -f "/etc/systemd/system/$RESUME_UNIT.service"
    systemctl daemon-reload
fi

say "Checking the internet connection"
# After the driver reboot this runs very early in boot, before DNS is always ready - wait for it.
for _ in $(seq 1 90); do
    getent hosts ollama.com >/dev/null 2>&1 && getent hosts astral.sh >/dev/null 2>&1 && break
    sleep 2
done
getent hosts ollama.com >/dev/null 2>&1 || die "No internet/DNS (can't resolve ollama.com). Check the network, then re-run: bash install.sh"
ok "Online"

say "Checking the GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || die "NVIDIA driver isn't working. Try: sudo ubuntu-drivers install && sudo reboot, then re-run install.sh"
ok "GPU ready"

say "Setting up Python environment (a few minutes - big CUDA libraries)"
# The bot gets its own Python 3.12 via uv, whatever the system ships (Ubuntu 26.04 has 3.14,
# which kokoro-onnx doesn't support yet). uv is also much faster than pip for these big wheels.
USER_HOME="$(getent passwd "$INSTALL_USER" | cut -d: -f6)"
UV="$USER_HOME/.local/bin/uv"
if [ ! -x "$UV" ]; then
    as_user sh -c 'curl -LsSf --retry 5 --retry-delay 3 --retry-all-errors https://astral.sh/uv/install.sh | sh' >/dev/null
fi
[ -x "$UV" ] || die "Couldn't install uv (Python installer) to $UV"
if [ -d .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
    warn "Existing .venv isn't Python 3.12 - rebuilding it"
    rm -rf .venv
fi
[ -d .venv ] || as_user "$UV" venv --quiet --seed --python 3.12 .venv
as_user "$UV" pip install --quiet --python .venv/bin/python -r requirements.txt
# kokoro-onnx pulls in CPU onnxruntime; swap it for the CUDA build (same import name).
as_user "$UV" pip uninstall --quiet --python .venv/bin/python onnxruntime >/dev/null 2>&1 || true
as_user "$UV" pip install --quiet --python .venv/bin/python --reinstall --no-deps "onnxruntime-gpu>=1.20,<1.24"  # 1.24+ builds need CUDA 13; our nvidia-*-cu12 libs are CUDA 12
if as_user .venv/bin/python -c "import onnxruntime as o, sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)"; then
    ok "Python packages installed (TTS GPU runtime available)"
else
    warn "onnxruntime-gpu has no CUDA provider - TTS will fall back to CPU"
fi
chmod +x run.sh

say "Downloading speech models"
as_user .venv/bin/python download_models.py kokoro
as_user .venv/bin/python - <<'PY'
from voicebot.config import load_config
cfg = load_config("config.yaml")
if cfg.stt.backend == "faster_whisper":
    from faster_whisper import download_model
    fw = cfg.stt.faster_whisper
    print("  Whisper model ready:", download_model(fw.model, cache_dir=fw.download_root))
PY
ok "Kokoro + Whisper downloaded"

say "Installing / updating Ollama"
curl -fsSL https://ollama.com/install.sh | sh
as_root mkdir -p /etc/systemd/system/ollama.service.d
as_root tee /etc/systemd/system/ollama.service.d/voicebot.conf >/dev/null <<'EOF'
[Service]
# Keep the model in VRAM permanently - no multi-second reload before a reply
Environment="OLLAMA_KEEP_ALIVE=-1"
# One conversation slot and one model: extra slots/models multiply VRAM use
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
# 6k context: persona + people profiles + conversation history, small KV cache
Environment="OLLAMA_CONTEXT_LENGTH=6144"
# Flash attention + 8-bit KV cache roughly halves KV-cache VRAM
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
as_root systemctl daemon-reload
as_root systemctl enable ollama >/dev/null 2>&1 || true
as_root systemctl restart ollama
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:11434/api/version >/dev/null || die "Ollama didn't start (check: journalctl -u ollama)"
ok "Ollama running (local only, 127.0.0.1:11434)"

say "Downloading the LLM: $MODEL (~3.3GB)"
ollama pull "$MODEL" || die "Couldn't pull $MODEL"
ok "Model ready"

say "Installing the bot service (starts on every boot)"
as_root tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<EOF
[Unit]
Description=Discord Voice LLM Bot
After=network-online.target ollama.service
Wants=network-online.target ollama.service

[Service]
Type=simple
User=$INSTALL_USER
ExecStart="$DIR/run.sh"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
as_root systemctl daemon-reload
as_root systemctl enable "$SERVICE" >/dev/null 2>&1
START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
as_root systemctl restart "$SERVICE"

say "Starting the bot (loading models onto the GPU)"
started=0
for _ in $(seq 1 120); do  # up to ~4 min: first start loads Whisper, Kokoro and the LLM
    if as_root journalctl -u "$SERVICE" --since "$START_TS" --no-pager 2>/dev/null | grep -q "Logged in as"; then
        started=1; break
    fi
    sleep 2
done
if [ "$started" -eq 1 ]; then
    ok "Bot is online"
else
    warn "Bot hasn't logged in yet. Recent log:"
    as_root journalctl -u "$SERVICE" --since "$START_TS" --no-pager -n 30 || true
fi
ollama ps || true

printf '\n\033[1;32mAll done!\033[0m\n'
if [ -n "${APP_ID:-}" ] && [ "$APP_ID" != "0" ]; then
    echo "  Invite the bot to your server (open in a browser):"
    echo "    https://discord.com/oauth2/authorize?client_id=$APP_ID&scope=bot+applications.commands&permissions=$PERMS"
fi
cat <<EOF
  Then join a voice channel and type /join

  Live logs:        journalctl -u $SERVICE -f
  Restart the bot:  sudo systemctl restart $SERVICE     (after editing config.yaml)
  Check the GPU:    ollama ps        (should say 100% GPU)

  Also on a self-hosted Fluxer server? Fill the fluxer: section in config.yaml, then:
                    .venv/bin/python -m voicebot.platform set both --restart     (see README, "Fluxer")
EOF
