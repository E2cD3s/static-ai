"""Config editor for the dashboard: a form schema built from the live config (help text from the comments
in DEFAULTS), and a save path that writes only changed keys into config.yaml (comments kept, via ruamel),
validates with the real loader first, backs up the old file, then applies the new values in place.

Secrets (tokens, keys, passwords) are never sent to the browser; an empty secret field means "keep".
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import LiteralScalarString

from .. import config as config_mod
from ..config import ConfigError, load_config

_SECRET = re.compile(r"(^|_)(token|secret|password|api_key|key)$")
_KEY_LINE = re.compile(r'^( *)"(\w+)":(.*)$')
_COMMENT = re.compile(r'[,{\[(]\s*#\s*(.+)$')

# Read once at startup (models, DBs, the gateway login...) - changing these needs a restart.
RESTART = ("discord.token", "discord.guild_ids", "discord.members_intent", "stt.", "tts.", "llm.default",
           "llm.voice_endpoint", "sentiment.", "dashboard.", "presence.", "profiles.db_path", "profiles.enabled",
           "reminders.db_path", "reminders.enabled", "bot.log_level", "bot.creator_ids", "clips.buffer_s",
           "platform.", "fluxer.")
NEXT_JOIN = ("voice.",)  # copied into a voice session when it starts
# Under a RESTART prefix, but read each time they're used.
LIVE = ("sentiment.check_in", "dashboard.contact", "dashboard.legal_updated")


def is_secret(path: str) -> bool:
    return bool(_SECRET.search(path.rsplit(".", 1)[-1]))


def when_applied(path: str) -> str:
    if path.startswith("llm.endpoints.") and path.rsplit(".", 1)[-1] in ("base_url", "api_key", "timeout"):
        return "restart"  # the HTTP client per endpoint is built once; sampling settings are read per request
    if path.startswith(LIVE):
        return "live"
    if path.startswith(RESTART) or path in {p.rstrip(".") for p in RESTART}:
        return "restart"
    if path.startswith(NEXT_JOIN):
        return "next_join"
    return "live"


def _help() -> dict[str, str]:
    """{'voice.silence_ms': 'comment text', 'llm.endpoints.*.temperature': ...} from config.py's comments."""
    out: dict[str, str] = {}
    region, stack = None, []
    for line in Path(config_mod.__file__).read_text().splitlines():
        if line.startswith("DEFAULTS = {"):
            region, stack = "", []
            continue
        if line.startswith("ENDPOINT_DEFAULTS = {"):
            region, stack = "llm.endpoints.*.", []
            continue
        if line.startswith("}"):
            region = None
            continue
        if region is None or not (m := _KEY_LINE.match(line)):
            continue
        level = len(m.group(1)) // 4 - 1
        stack = stack[:level] + [m.group(2)]
        if (c := _COMMENT.search(m.group(3))):
            out[region + ".".join(stack)] = c.group(1).strip()
    return out


_HELP = _help()


def _help_for(path: str) -> str:
    if path in _HELP:
        return _HELP[path]
    if path.startswith("llm.endpoints."):
        parts = path.split(".")
        return _HELP.get(".".join(parts[:2] + ["*"] + parts[3:]), "")
    return ""


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "text" if "\n" in value or len(value) > 90 else "str"
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value):
        return "list"
    return "json"


def schema(cfg: dict) -> list[dict]:
    """Flat list of fields in config order; the UI groups them by path prefix."""
    fields: list[dict] = []

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                walk(value, path + ".")
                continue
            kind = _kind(value)
            secret = is_secret(path) and isinstance(value, str)
            if secret:
                shown: Any = ""
            elif kind == "int":
                shown = str(value)  # Discord IDs don't fit in a JS number
            elif kind == "list":
                shown = [str(v) for v in value]
            elif kind == "json":
                shown = json.dumps(value, indent=2)
            else:
                shown = value
            fields.append({"path": path, "kind": "secret" if secret else kind, "value": shown,
                           "set": bool(value) if secret else None, "help": _help_for(path),
                           "applies": when_applied(path)})

    walk(cfg, "")
    return fields


def _get(cfg: dict, path: str) -> Any:
    node: Any = cfg
    for k in path.split("."):
        node = node[k]
    return node


def _coerce(path: str, old: Any, new: Any) -> Any:
    kind = _kind(old)
    if kind == "bool":
        return bool(new)
    if kind == "int":
        return int(str(new).strip())
    if kind == "float":
        return float(str(new).strip())
    if kind in ("str", "text"):
        return str(new)
    if kind == "list":
        items = [str(v).strip() for v in new if str(v).strip()] if isinstance(new, list) else \
            [v.strip() for v in str(new).splitlines() if v.strip()]
        if path.endswith("_ids") or (old and all(isinstance(v, int) for v in old)):
            return [int(v) for v in items]
        if old and all(isinstance(v, float) for v in old):
            return [float(v) for v in items]
        return items
    return json.loads(new) if isinstance(new, str) else new


def _yaml_value(v: Any) -> Any:
    if isinstance(v, str) and "\n" in v:
        return LiteralScalarString(v)
    return v


def _update_in_place(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _update_in_place(dst[k], v)
        else:
            dst[k] = v


class ConfigEditor:
    def __init__(self, cfg: dict, path: str | Path, backup_dir: str | Path = "data"):
        self.cfg = cfg
        self.path = Path(path)
        self.backup = Path(backup_dir) / "config.backup.yaml"
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.width = 120

    def schema(self) -> list[dict]:
        return schema(self.cfg)

    def save(self, changes: dict[str, Any]) -> dict:
        """Blocking (file IO + validation). Returns {'saved': [...], 'restart': [...], 'next_join': [...]}.
        Raises ValueError with a readable message on bad input."""
        clean: dict[str, Any] = {}
        for path, new in changes.items():
            try:
                old = _get(self.cfg, path)
            except (KeyError, TypeError):
                raise ValueError(f"unknown setting: {path}") from None
            if is_secret(path) and isinstance(old, str) and not str(new).strip():
                continue  # blank secret = keep
            try:
                value = _coerce(path, old, new)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                raise ValueError(f"{path}: {e}") from None
            if value != old:
                clean[path] = value
        if not clean:
            return {"saved": [], "restart": [], "next_join": []}

        with open(self.path, encoding="utf-8") as f:
            doc = self.yaml.load(f) or CommentedMap()
        for path, value in clean.items():
            node = doc
            *parents, leaf = path.split(".")
            for k in parents:
                if not isinstance(node.get(k), dict):
                    node[k] = CommentedMap()
                node = node[k]
            node[leaf] = _yaml_value(value)
        buf = io.StringIO()
        self.yaml.dump(doc, buf)

        tmp = self.path.with_name(f".{self.path.name}.new")
        tmp.write_text(buf.getvalue(), encoding="utf-8")
        os.chmod(tmp, 0o600)
        try:
            fresh = load_config(tmp)
        except ConfigError as e:
            tmp.unlink(missing_ok=True)
            raise ValueError(str(e)) from None
        except Exception as e:  # noqa: BLE001 - YAML errors etc.
            tmp.unlink(missing_ok=True)
            raise ValueError(f"config didn't load: {e}") from None

        self.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, self.backup)
        os.chmod(self.backup, 0o600)
        tmp.replace(self.path)
        _update_in_place(self.cfg, fresh)

        return {"saved": sorted(clean),
                "restart": sorted(p for p in clean if when_applied(p) == "restart"),
                "next_join": sorted(p for p in clean if when_applied(p) == "next_join")}
