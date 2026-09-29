"""Switch which chat platforms Static runs on (platform.mode in config.yaml), from the server's shell:

    .venv/bin/python -m voicebot.platform                  # show the current mode
    .venv/bin/python -m voicebot.platform set fluxer       # discord | fluxer | both
    .venv/bin/python -m voicebot.platform set both --restart   # ...and restart the service now

The change is validated with the real config loader (e.g. fluxer needs fluxer.token) and written keeping the
file's comments, like the dashboard's settings editor. It takes effect on restart.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from .config import PLATFORM_MODES, ConfigError, load_config  # noqa: E402
from .web.confedit import ConfigEditor  # noqa: E402

SERVICE = "discord-voicebot"


def main(argv: list[str]) -> int:
    path = ROOT / "config.yaml"
    try:
        cfg = load_config(path)
    except ConfigError as e:
        print(f"Config error: {e}")
        return 1
    if not argv or argv[0] in ("show", "status"):
        print(f"platform.mode: {cfg.platform.mode}   (choices: {', '.join(PLATFORM_MODES)})")
        return 0
    if argv[0] != "set" or len(argv) < 2 or argv[1].lower() not in PLATFORM_MODES:
        print(__doc__)
        return 1
    mode, old = argv[1].lower(), cfg.platform.mode
    if mode == old:
        print(f"Already {mode}.")
    else:
        try:
            ConfigEditor(cfg, path).save({"platform.mode": mode})
        except ValueError as e:
            print(f"Not changed: {e}")
            return 1
        print(f"platform.mode: {old} -> {mode} (saved to config.yaml)")
    if "--restart" in argv:
        print(f"Restarting {SERVICE}...")
        return subprocess.call(["sudo", "systemctl", "restart", SERVICE])
    print(f"Takes effect on restart: sudo systemctl restart {SERVICE}  (or add --restart)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
