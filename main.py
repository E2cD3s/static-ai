"""Entry point: python main.py [path/to/config.yaml]"""
import asyncio
import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # model paths in config are relative to the project folder

from voicebot.config import ConfigError, load_config  # noqa: E402


def wait_for_dns(host: str = "discord.com", timeout: float = 180) -> None:
    """At boot systemd can start us before DNS works (network-online.target doesn't wait on this machine), and
    discord.py's login then dies with 'Temporary failure in name resolution'. Wait for it instead of crash-looping."""
    log = logging.getLogger("voicebot")
    deadline = time.monotonic() + timeout
    warned = False
    while True:
        try:
            socket.getaddrinfo(host, 443)
            if warned:
                log.info("Network is up")
            return
        except OSError:
            if time.monotonic() >= deadline:
                return  # let login raise the real error
            if not warned:
                log.info("Waiting for the network (can't resolve %s yet)...", host)
                warned = True
            time.sleep(2)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config.yaml"
    if not path.exists():
        shutil.copy(ROOT / "config.example.yaml", path)
        print(f"Created {path} from config.example.yaml - fill in your Discord token and LLM endpoints, then re-run.")
        return 1
    try:
        cfg = load_config(path)
    except ConfigError as e:
        print(f"Config error: {e}")
        return 1

    logging.basicConfig(
        level=getattr(logging, str(cfg.bot.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("discord.ext.voice_recv", "httpx", "faster_whisper", "discord.gateway"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from voicebot.bot import VoiceBot

    mode = cfg.platform.mode
    logging.getLogger("voicebot").info("Platform mode: %s", mode)
    if mode == "discord":
        wait_for_dns()
        bot = VoiceBot(cfg, config_path=path)
        bot.run(cfg.discord.token, log_handler=None)
    else:
        from urllib.parse import urlparse
        wait_for_dns(urlparse(cfg.fluxer.api_url).hostname or "discord.com")
        if mode == "both":
            wait_for_dns()
        bot = VoiceBot(cfg, config_path=path)
        try:
            asyncio.run(run_multi(bot, cfg, mode))
        except KeyboardInterrupt:
            pass
    return 75 if bot.restart_requested else 0  # non-zero: systemd (Restart=on-failure) starts us again


async def run_multi(bot, cfg, mode: str) -> None:
    """platform.mode fluxer/both: the Discord client object hosts the shared models + dashboard (and logs in to
    Discord only in 'both'); the Fluxer frontend runs next to it. Ends when the bot closes (dashboard restart)
    or on SIGTERM (systemctl stop/restart)."""
    import signal

    from voicebot.fluxer import FluxerBot

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.close()))
    async with bot:
        watch = [asyncio.create_task(bot.closed.wait())]
        if mode == "both":
            watch.append(asyncio.create_task(bot.start(cfg.discord.token), name="discord"))
        bot.fluxer = FluxerBot(bot)
        # Supervised inside (reconnects on its own): Fluxer trouble never ends the process or touches Discord.
        fluxer = asyncio.create_task(bot.fluxer.run(), name="fluxer")
        if mode == "fluxer":
            await bot.start_services(discord_online=False)
        await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
        for t in watch[1:]:
            if t.done() and not t.cancelled() and t.exception():
                logging.getLogger("voicebot").error("Discord stopped: %r", t.exception())
        if not bot.closed.is_set():
            await bot.close()
        fluxer.cancel()
        for t in watch:
            t.cancel()


if __name__ == "__main__":
    sys.exit(main())
