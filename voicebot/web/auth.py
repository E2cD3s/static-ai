"""Dashboard auth: HMAC-signed session cookies (stateless, so they survive restarts), a local scrypt
password, Discord OAuth2 for admins, and a per-IP login rate limit.

State lives in data/dashboard.json (mode 600): the cookie-signing secret and the password hash.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

SESSION_COOKIE = "static_session"
STATE_COOKIE = "static_oauth_state"
SESSION_DAYS = 14
_SCRYPT = dict(n=2**15, r=8, p=1, maxmem=64 * 2**20, dklen=32)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class AuthStore:
    def __init__(self, path: str | Path = "data/dashboard.json"):
        self.path = Path(path)
        self._data: dict = {}
        self._mtime = 0.0
        self._load()
        if not self._data.get("secret"):
            self._data["secret"] = secrets.token_hex(32)
            self.save()

    def _load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text() or "{}")
            self._mtime = self.path.stat().st_mtime

    @property
    def data(self) -> dict:
        """Re-read if the file changed (the set-password CLI writes it while the bot runs)."""
        try:
            if self.path.stat().st_mtime != self._mtime:
                self._load()
        except FileNotFoundError:
            pass
        return self._data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    # ------------------------------------------------------------ password

    @property
    def password_user(self) -> str | None:
        return (self.data.get("password") or {}).get("user")

    def set_password(self, username: str, password: str) -> None:
        """Blocking (~0.1s scrypt). Also signs out every existing password session."""
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
        self.data["password"] = {"user": username.strip(), "salt": salt.hex(), "hash": digest.hex()}
        self.data["pw_epoch"] = self.data.get("pw_epoch", 0) + 1
        self.save()

    def check_password(self, username: str, password: str) -> bool:
        """Blocking (~0.1s scrypt) - run in a thread."""
        p = self.data.get("password")
        if not p:
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(p["salt"]), **_SCRYPT)
        return hmac.compare_digest(digest.hex(), p["hash"]) & hmac.compare_digest(
            username.strip().lower().encode(), p["user"].lower().encode())

    # ------------------------------------------------------------ sessions

    def _sig(self, body: str) -> str:
        return _b64(hmac.new(bytes.fromhex(self.data["secret"]), body.encode(), hashlib.sha256).digest())

    def make_session(self, *, name: str, via: str, uid: str = "", avatar: str = "") -> str:
        payload = {"name": name, "via": via, "uid": uid, "avatar": avatar,
                   "exp": int(time.time()) + SESSION_DAYS * 86400,
                   "epoch": self.data.get("epoch", 0), "pw_epoch": self.data.get("pw_epoch", 0)}
        body = _b64(json.dumps(payload, separators=(",", ":")).encode())
        return f"{body}.{self._sig(body)}"

    def read_session(self, cookie: str | None) -> dict | None:
        if not cookie or "." not in cookie:
            return None
        body, sig = cookie.rsplit(".", 1)
        if not hmac.compare_digest(sig.encode(), self._sig(body).encode()):  # bytes: str raises on non-ASCII
            return None
        try:
            s = json.loads(_unb64(body))
        except ValueError:
            return None
        if s.get("exp", 0) < time.time() or s.get("epoch") != self.data.get("epoch", 0):
            return None
        if s.get("via") == "password" and s.get("pw_epoch") != self.data.get("pw_epoch", 0):
            return None
        return s

    def sign_out_everyone(self) -> None:
        self.data["epoch"] = self.data.get("epoch", 0) + 1
        self.save()

    # ------------------------------------------------------------ oauth state

    def make_state(self) -> str:
        nonce = secrets.token_urlsafe(16)
        return f"{nonce}.{self._sig('state:' + nonce)}"

    def check_state(self, cookie: str | None, returned: str | None) -> bool:
        if not cookie or not returned or not hmac.compare_digest(cookie.encode(), returned.encode()) or "." not in cookie:
            return False
        nonce, sig = cookie.rsplit(".", 1)
        return hmac.compare_digest(sig.encode(), self._sig("state:" + nonce).encode())


class RateLimit:
    """At most `limit` failed logins per `window_s` per client IP."""

    def __init__(self, limit: int = 5, window_s: float = 600):
        self.limit, self.window = limit, window_s
        self._fails: dict[str, deque[float]] = defaultdict(deque)

    def blocked(self, ip: str) -> float:
        """Seconds until this IP may try again (0 = allowed)."""
        q, now = self._fails.get(ip), time.monotonic()
        if not q:
            return 0
        while q and now - q[0] > self.window:
            q.popleft()
        if not q:
            del self._fails[ip]
            return 0
        return self.window - (now - q[0]) if len(q) >= self.limit else 0

    def fail(self, ip: str) -> None:
        self._fails[ip].append(time.monotonic())

    def clear(self, ip: str) -> None:
        self._fails.pop(ip, None)
