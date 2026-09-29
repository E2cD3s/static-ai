"""Web dashboard, served from inside the bot process (aiohttp on the bot's event loop, so it reads live state).

Public:  /status  - the same numbers as /status, auto-refreshing (JSON at /api/status, cached a few seconds).
Admin:   /admin   - servers (leave, voice join/leave, slash-command sync, invite link), people (everything
                    stored per user, forget / opt out), settings editor, logs, restart, account. Login: Discord OAuth2 (admin_ids + bot.creator_ids) or a local password.

Meant to sit behind a reverse proxy doing HTTPS: X-Forwarded-For/-Proto/-Host are trusted only from
`dashboard.trusted_proxies`. POSTs must be JSON and same-origin (CSRF), cookies are HttpOnly + SameSite=Lax.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

import discord
import httpx
from aiohttp import web

from .. import helpinfo
from . import snapshot
from .auth import SESSION_COOKIE, SESSION_DAYS, STATE_COOKIE, AuthStore, RateLimit
from .confedit import ConfigEditor

if TYPE_CHECKING:
    from ..bot import VoiceBot

log = logging.getLogger("voicebot.web")

STATIC = Path(__file__).parent / "static"
DISCORD_API = "https://discord.com/api/v10"
INVITE_PERMS = discord.Permissions(
    view_channel=True, send_messages=True, send_messages_in_threads=True, create_public_threads=True,
    embed_links=True, attach_files=True, read_message_history=True, add_reactions=True, use_external_emojis=True,
    use_application_commands=True, connect=True, speak=True, use_voice_activation=True,
    view_audit_log=True)  # to see who force-disconnected it (voice.boomerang)
_NEEDED = ("view_channel", "send_messages", "read_message_history", "embed_links", "attach_files", "view_audit_log",
           "add_reactions", "connect", "speak")
_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; style-src-attr 'unsafe-inline'; "
        "font-src https://fonts.gstatic.com; img-src 'self' data: https://cdn.discordapp.com; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def _err(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


class Dashboard:
    def __init__(self, bot: "VoiceBot", config_path: str | Path):
        self.bot = bot
        self.cfg = bot.cfg.dashboard
        self.auth = AuthStore()
        self.limiter = RateLimit()
        self.editor = ConfigEditor(bot.cfg, config_path)
        # Fluxer avatars come from the Fluxer server's own media host
        fx_url = str(bot.cfg.fluxer.api_url or "")
        fx_origin = "/".join(fx_url.split("/", 3)[:3]) if fx_url.startswith("https://") else ""
        self._csp = _CSP.replace("https://cdn.discordapp.com;", f"https://cdn.discordapp.com {fx_origin};") if fx_origin else _CSP
        self._proxies = [ipaddress.ip_network(n, strict=False) for n in self.cfg.trusted_proxies]
        self._status: tuple[float, dict] | None = None
        self._status_lock = asyncio.Lock()
        self._scrypt = asyncio.Semaphore(2)  # 32 MB each
        self._runner: web.AppRunner | None = None
        self._avatars: dict[int, str | None] = {}  # user id -> avatar url fetched for the People tab
        self._role_metadata_ok = False  # linked-role metadata registered with Discord this run

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if not self.cfg.enabled or self._runner is not None:
            return
        app = web.Application(middlewares=[self._middleware], client_max_size=256 * 1024)
        r = app.router
        r.add_get("/", self.root)
        r.add_get("/status", self._page("status.html"))
        r.add_get("/help", self._page("help.html"))
        r.add_get("/terms", self._legal("terms.html"))
        r.add_get("/privacy", self._legal("privacy.html"))
        r.add_get("/linked-role", self.linked_start)          # Developer Portal: Linked Roles Verification URL
        r.add_get("/auth/linked-role", self.linked_callback)  # ...and this one is its OAuth2 redirect
        r.add_get("/login", self.login_page)
        r.add_get("/admin", self._page("admin.html"))
        r.add_static("/static", STATIC)
        r.add_get("/api/status", self.public_status)
        r.add_get("/api/help", self.public_help)
        r.add_get("/api/auth/options", self.auth_options)
        r.add_post("/api/auth/login", self.password_login)
        r.add_post("/api/auth/logout", self.logout)
        r.add_get("/auth/discord", self.discord_start)
        r.add_get("/auth/callback", self.discord_callback)
        r.add_get("/api/admin/me", self.me)
        r.add_get("/api/admin/learned", self.admin_learned)
        r.add_get("/api/admin/snapshot", self.admin_snapshot)
        r.add_get("/api/admin/guilds", self.guilds)
        r.add_post("/api/admin/guilds/{gid}/leave", self.leave_guild)
        r.add_post("/api/admin/guilds/{gid}/voice", self.voice)
        r.add_post("/api/admin/guilds/{gid}/commands", self.sync_commands)
        r.add_post("/api/admin/guilds/{gid}/mute", self.mute)
        r.add_get("/api/admin/people", self.people)
        r.add_get("/api/admin/people/{uid}", self.person)
        r.add_post("/api/admin/people/{uid}/forget", self.forget_person)
        r.add_post("/api/admin/people/{uid}/profiling", self.person_profiling)
        r.add_get("/api/admin/live", self.live)
        r.add_get("/api/admin/config", self.get_config)
        r.add_post("/api/admin/config", self.save_config)
        r.add_post("/api/admin/restart", self.restart)
        r.add_post("/api/admin/platform", self.set_platform)
        r.add_get("/api/admin/platforms", self.platforms)
        r.add_get("/api/admin/logs", self.logs)
        r.add_post("/api/admin/password", self.set_password)
        r.add_post("/api/admin/signout-all", self.signout_all)
        # Browsers send every cookie set on the parent domain: other sites under it (e.g. AWS load-balancer
        # cookies) pushed the Cookie header past aiohttp's 8 KB default and every page answered 400.
        self._runner = web.AppRunner(app, access_log=None, max_line_size=65536, max_field_size=65536)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.cfg.host, int(self.cfg.port)).start()
        log.info("Dashboard on http://%s:%s (public /status, admin /admin; login: %s)", self.cfg.host, self.cfg.port,
                 " + ".join(m for m, on in (("discord", self._discord_ready()),
                                            ("password", bool(self.auth.password_user))) if on) or "NONE SET UP")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------ request plumbing

    def _trusted(self, ip: str | None) -> bool:
        try:
            addr = ipaddress.ip_address(ip or "")
        except ValueError:
            return False
        return any(addr in net for net in self._proxies)

    def _ip(self, request: web.Request) -> str:
        peer = request.remote or "?"
        if self._trusted(peer):
            if (xff := request.headers.get("X-Forwarded-For")):
                return xff.split(",")[-1].strip()
            if (real := request.headers.get("X-Real-IP")):
                return real.strip()
        return peer

    def _https(self, request: web.Request) -> bool:
        return request.secure or (self._trusted(request.remote)
                                  and request.headers.get("X-Forwarded-Proto", "").lower() == "https")

    def _same_origin(self, request: web.Request) -> bool:
        origin = request.headers.get("Origin")
        if not origin:
            return True  # non-browser client; the JSON content-type check still applies
        host = urlparse(origin).netloc.lower()
        ok = {request.host.lower(), urlparse(str(self.cfg.public_url)).netloc.lower()}
        if self._trusted(request.remote) and (fwd := request.headers.get("X-Forwarded-Host")):
            ok.add(fwd.split(",")[0].strip().lower())
        return host in ok

    def _admin_ids(self) -> set[int]:
        ids = set(self.cfg.admin_ids or []) | set(self.bot.cfg.bot.get("creator_ids") or [])
        return {int(i) for i in ids}

    def _session(self, request: web.Request) -> dict | None:
        s = self.auth.read_session(request.cookies.get(SESSION_COOKIE))
        if s and s.get("via") == "discord" and int(s.get("uid") or 0) not in self._admin_ids():
            return None  # admin rights removed since login
        return s

    def _set_session(self, request: web.Request, resp: web.StreamResponse, token: str) -> None:
        resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_DAYS * 86400, path="/", httponly=True,
                        samesite="Lax", secure=self._https(request))

    @web.middleware
    async def _middleware(self, request: web.Request, handler):
        try:
            if request.path == "/admin" or request.path.startswith("/api/admin/"):
                session = self._session(request)
                if session is None:
                    if request.path.startswith("/api/"):
                        return self._headers(request, _err(401, "not signed in"))
                    raise web.HTTPFound("/login")
                request["session"] = session
            if request.method == "POST":
                if request.content_type != "application/json":
                    return self._headers(request, _err(415, "JSON only"))
                if not self._same_origin(request):
                    log.warning("Dashboard: blocked cross-origin POST %s from %s (Origin %s)", request.path,
                                self._ip(request), request.headers.get("Origin"))
                    return self._headers(request, _err(403, "cross-origin request blocked"))
            resp = await handler(request)
        except web.HTTPException as e:
            self._headers(request, e)
            raise
        except Exception:
            log.exception("Dashboard error on %s %s", request.method, request.path)
            resp = _err(500, "internal error - see the bot's log")
        return self._headers(request, resp)

    def _headers(self, request: web.Request, resp: web.StreamResponse) -> web.StreamResponse:
        h = resp.headers
        h.setdefault("Content-Security-Policy", self._csp)
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("Referrer-Policy", "same-origin")
        h.setdefault("X-Frame-Options", "DENY")
        if request.path.startswith(("/api/", "/admin", "/auth/")):
            h["Cache-Control"] = "no-store"
        if self._https(request):
            h.setdefault("Strict-Transport-Security", "max-age=31536000")
        return resp

    async def root(self, _request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/status")

    def _page(self, name: str):
        async def handler(_request: web.Request) -> web.FileResponse:
            return web.FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})
        return handler

    def _who(self, request: web.Request) -> str:
        s = request.get("session") or {}
        return f"{s.get('name', '?')} ({s.get('via', '?')}, {self._ip(request)})"

    # ------------------------------------------------------------ public status

    def _bot_card(self) -> dict:
        b, u = self.bot, self.bot.user
        fx = getattr(b, "fluxer", None)
        mode = b.cfg.platform.mode
        fx_ready = bool(fx is not None and fx._ready.is_set())
        # "ready" = every platform in this mode is up (Fluxer-only: the models are warm and Fluxer is connected)
        ready = ((b.is_ready() or mode == "fluxer") and (fx_ready or mode == "discord")
                 and b.models_ready.is_set())
        return {"name": b.cfg.bot.name, "user": u.name if u else (fx.user.name if fx_ready and fx.user else None),
                "avatar": u.display_avatar.url if u else None, "ready": ready,
                "guilds": len(b.guilds), "uptime": time.time() - b.started_at,
                "presence": getattr(getattr(b, "presence", None), "_last", ""),
                "platform": {"mode": mode, "discord": b.is_ready(), "fluxer": fx_ready,
                             "fluxer_servers": len(fx.state.guilds) if fx_ready else 0}}

    async def _snapshot(self) -> dict:
        """The live board (public page + admin Overview), rebuilt at most every status_cache_s."""
        async with self._status_lock:
            if self._status is None or time.monotonic() - self._status[0] > float(self.cfg.status_cache_s):
                try:
                    data = await snapshot.build(self.bot, self._bot_card())
                except Exception:  # noqa: BLE001 - e.g. mid-startup, before the models exist
                    log.exception("Status snapshot failed")
                    data = {"bot": self._bot_card(), "starting": True, "ts": time.time()}
                self._status = (time.monotonic(), data)
            return self._status[1]

    async def public_status(self, _request: web.Request) -> web.Response:
        if not self.cfg.public_status:
            return _err(404, "the public status page is turned off")
        data = await self._snapshot()
        # Strangers don't need to know where people are hanging out: no server or channel names publicly.
        # The admin Overview (admin_snapshot) still shows them.
        public = {**data, "voice_now": [{k: v for k, v in s.items() if k not in ("server", "channel")}
                                        for s in data.get("voice_now") or []]}
        return web.json_response(public)

    async def public_help(self, _request: web.Request) -> web.Response:
        """The user guide's live details (name, wake words, features, commands). Public, like the guide."""
        return web.json_response(helpinfo.info(self.bot))

    # ------------------------------------------------------------ terms, privacy, linked roles

    def _contact(self) -> str:
        if self.cfg.contact:
            return str(self.cfg.contact)
        names = getattr(self.bot, "_creator_names", None) or []  # looked up at startup (bot.creator_ids)
        return f"the owner on Discord, @{names[0]}" if names else "the bot's owner on Discord"

    def _fill(self, name: str, **extra: str) -> web.Response:
        """A page from static/ with {{placeholders}} filled in (values are escaped unless passed as `raw_*`)."""
        b = self.bot.cfg
        values = {"name": b.bot.name, "contact": self._contact(), "updated": self.cfg.legal_updated,
                  "site": str(self.cfg.public_url or "this site").removeprefix("https://"),
                  "clip_seconds": f"{float(b.clips.buffer_s):.0f}", "context": str(int(b.discord.context_messages)),
                  "fresh": f"{float(b.bot.get('fresh_after_min') or 0):.0f}"}
        page = (STATIC / name).read_text()
        for k, v in values.items():
            page = page.replace("{{%s}}" % k, html.escape(v))
        for k, v in extra.items():
            page = page.replace("{{%s}}" % k.removeprefix("raw_"), v if k.startswith("raw_") else html.escape(v))
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-cache"})

    def _legal(self, name: str):
        async def handler(_request: web.Request) -> web.Response:
            return self._fill(name)
        return handler

    def _linked_page(self, state: str, title: str, *paragraphs: str) -> web.Response:
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        return self._fill("linked.html", state=state, title=title, raw_body=body)

    _ROLE_METADATA = [
        {"key": "talked", "name": "Has talked to Static", "description": "Has talked with the bot at least once",
         "type": 7},  # boolean_equal
        {"key": "messages", "name": "Times talked", "description": "Times they've talked to the bot (at least)",
         "type": 2},  # integer_greater_than_or_equal
        {"key": "known_since", "name": "Days known", "description": "Days since they first talked to the bot",
         "type": 5},  # datetime_less_than_or_equal (value = first seen, "at least N days ago")
    ]

    async def _register_role_metadata(self, http: httpx.AsyncClient) -> None:
        """Tell Discord which linked-role conditions servers can use. Idempotent; done once per run."""
        if self._role_metadata_ok:
            return
        name = self.bot.cfg.bot.name
        meta = [{**m, "name": m["name"].replace("Static", name)} for m in self._ROLE_METADATA]
        r = await http.put(f"{DISCORD_API}/applications/{self.bot.application_id}/role-connections/metadata",
                           json=meta, headers={"Authorization": f"Bot {self.bot.http.token}"})
        if r.status_code >= 300:
            log.warning("Linked roles: metadata registration failed (%s): %s", r.status_code, r.text[:200])
        else:
            self._role_metadata_ok = True
            log.info("Linked roles: metadata registered (%s)", ", ".join(m["key"] for m in meta))

    def _linked_redirect(self) -> str:
        return str(self.cfg.public_url).rstrip("/") + "/auth/linked-role"

    async def linked_start(self, request: web.Request) -> web.StreamResponse:
        """The Linked Roles Verification URL: Discord sends people here from a server's role requirements."""
        if not (self.cfg.public_url and self.cfg.discord_client_secret and self.bot.application_id):
            return self._linked_page("bad", "Linked roles aren't set up yet",
                                     "The bot's owner still needs to finish setting this up. Try again later.")
        state = self.auth.make_state()
        url = "https://discord.com/oauth2/authorize?" + urlencode({
            "client_id": self.bot.application_id, "redirect_uri": self._linked_redirect(), "response_type": "code",
            "scope": "role_connections.write identify", "state": state, "prompt": "consent"})
        resp = web.HTTPFound(url)
        resp.set_cookie(STATE_COOKIE, state, max_age=600, path="/auth", httponly=True, samesite="Lax",
                        secure=self._https(request))
        raise resp

    async def linked_callback(self, request: web.Request) -> web.Response:
        name = self.bot.cfg.bot.name
        again = '<a href="/linked-role">Try again</a>.'
        if not self.auth.check_state(request.cookies.get(STATE_COOKIE), request.query.get("state")):
            return self._linked_page("bad", "That link expired", f"The sign-in took too long or was opened twice. {again}")
        if not (code := request.query.get("code")):
            return self._linked_page("bad", "Not connected", f"You cancelled the Discord prompt, so nothing was linked. {again}")
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                tok = await http.post(f"{DISCORD_API}/oauth2/token", data={
                    "client_id": str(self.bot.application_id), "client_secret": self.cfg.discord_client_secret,
                    "grant_type": "authorization_code", "code": code, "redirect_uri": self._linked_redirect()})
                if tok.status_code != 200:
                    log.warning("Linked roles: token exchange failed (%s): %s", tok.status_code, tok.text[:200])
                    return self._linked_page("bad", "Discord said no", f"Discord didn't accept the sign-in. {again}")
                auth = {"Authorization": f"Bearer {tok.json()['access_token']}"}  # used here, never stored
                me = (await http.get(f"{DISCORD_API}/users/@me", headers=auth)).json()
                uid, username = int(me["id"]), me.get("username", "?")
                row = self.bot.profiles.store.get(uid)
                talked = bool(row and (row["messages"] or 0) > 0)
                metadata = {"talked": int(talked), "messages": int(row["messages"] or 0) if row else 0}
                if talked:
                    metadata["known_since"] = datetime.fromtimestamp(row["first_seen"], timezone.utc).isoformat()
                await self._register_role_metadata(http)
                r = await http.put(f"{DISCORD_API}/users/@me/applications/{self.bot.application_id}/role-connection",
                                   headers=auth, json={"platform_name": name, "platform_username": username,
                                                       "metadata": metadata})
                if r.status_code >= 300:
                    log.warning("Linked roles: update for %s failed (%s): %s", username, r.status_code, r.text[:200])
                    return self._linked_page("bad", "Couldn't update your roles",
                                             f"Discord refused the update. {again}")
        except httpx.HTTPError as e:
            log.warning("Linked roles: failed: %s", e)
            return self._linked_page("bad", "Couldn't reach Discord", f"Something went wrong on the way. {again}")
        log.info("🔗 Linked roles: %s connected (%s)", username, metadata)
        who = html.escape(username)
        if talked:
            since = datetime.fromtimestamp(row["first_seen"]).strftime("%B %-d, %Y")
            stats = (f"You've talked to {html.escape(name)} <b>{metadata['messages']}</b> times, starting "
                     f"<b>{since}</b>.")
        else:
            stats = (f"You haven't talked to {html.escape(name)} yet, so roles that need that won't unlock. "
                     f"Say hi in a voice channel or @mention it, then come back to this page.")
        return self._linked_page(
            "ok", "You're connected",
            f"Linked as <b>{who}</b>. {stats}",
            "Your roles update right away in servers that use these requirements. Your numbers are only refreshed "
            "when you connect, so visit this page again later to update them.",
            'You can disconnect any time in Discord under User Settings, Connections. '
            '<a href="/privacy">How your data is used</a>.')

    # ------------------------------------------------------------ login

    def _discord_ready(self) -> bool:
        return bool(self.cfg.discord_login and self.cfg.discord_client_secret and self.cfg.public_url
                    and self.bot.application_id)

    async def login_page(self, request: web.Request) -> web.StreamResponse:
        if self._session(request):
            raise web.HTTPFound("/admin")
        return await self._page("login.html")(request)

    async def auth_options(self, _request: web.Request) -> web.Response:
        u = self.bot.user
        return web.json_response({"discord": self._discord_ready(), "password": bool(self.auth.password_user),
                                  "name": self.bot.cfg.bot.name, "avatar": u.display_avatar.url if u else None})

    async def password_login(self, request: web.Request) -> web.Response:
        ip = self._ip(request)
        if (wait := self.limiter.blocked(ip)):
            return _err(429, f"Too many attempts. Try again in {max(1, round(wait / 60))} min.")
        try:
            body = await request.json()
        except ValueError:
            return _err(400, "bad request")
        user, password = str(body.get("username", ""))[:64], str(body.get("password", ""))[:256]
        ok = False
        if self.auth.password_user and user and password:
            async with self._scrypt:
                ok = await asyncio.to_thread(self.auth.check_password, user, password)
        if not ok:
            self.limiter.fail(ip)
            log.warning("Dashboard: failed password login for %r from %s", user, ip)
            return _err(401, "Wrong username or password.")
        self.limiter.clear(ip)
        log.info("Dashboard: %s signed in with password from %s", user, ip)
        resp = web.json_response({"ok": True})
        self._set_session(request, resp, self.auth.make_session(name=self.auth.password_user, via="password"))
        return resp

    async def logout(self, _request: web.Request) -> web.Response:
        resp = web.json_response({"ok": True})
        resp.del_cookie(SESSION_COOKIE, path="/")
        return resp

    def _redirect_uri(self) -> str:
        return str(self.cfg.public_url).rstrip("/") + "/auth/callback"

    async def discord_start(self, request: web.Request) -> web.StreamResponse:
        if not self._discord_ready():
            raise web.HTTPFound("/login?error=discord_off")
        state = self.auth.make_state()
        url = "https://discord.com/oauth2/authorize?" + urlencode({
            "client_id": self.bot.application_id, "redirect_uri": self._redirect_uri(), "response_type": "code",
            "scope": "identify", "state": state, "prompt": "none"})
        resp = web.HTTPFound(url)
        resp.set_cookie(STATE_COOKIE, state, max_age=600, path="/auth", httponly=True, samesite="Lax",
                        secure=self._https(request))
        raise resp

    async def discord_callback(self, request: web.Request) -> web.StreamResponse:
        def back(error: str) -> web.HTTPFound:
            r = web.HTTPFound(f"/login?error={error}")
            r.del_cookie(STATE_COOKIE, path="/auth")
            return r

        if not self._discord_ready():
            raise back("discord_off")
        if not self.auth.check_state(request.cookies.get(STATE_COOKIE), request.query.get("state")):
            raise back("state")
        if not (code := request.query.get("code")):
            raise back("denied")
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                tok = await http.post(f"{DISCORD_API}/oauth2/token", data={
                    "client_id": str(self.bot.application_id), "client_secret": self.cfg.discord_client_secret,
                    "grant_type": "authorization_code", "code": code, "redirect_uri": self._redirect_uri()})
                if tok.status_code != 200:
                    log.warning("Dashboard: Discord token exchange failed (%s): %s", tok.status_code, tok.text[:200])
                    raise back("discord")
                me = (await http.get(f"{DISCORD_API}/users/@me",
                                     headers={"Authorization": f"Bearer {tok.json()['access_token']}"})).json()
        except httpx.HTTPError as e:
            log.warning("Dashboard: Discord login failed: %s", e)
            raise back("discord") from None
        uid, name = int(me["id"]), me.get("username", "?")
        if uid not in self._admin_ids():
            log.warning("Dashboard: refused Discord login from %s (%s) at %s - not an admin", name, uid,
                        self._ip(request))
            raise back("not_admin")
        avatar = (f"https://cdn.discordapp.com/avatars/{uid}/{me['avatar']}.png?size=64" if me.get("avatar") else "")
        log.info("Dashboard: %s signed in with Discord from %s", name, self._ip(request))
        resp = web.HTTPFound("/admin")
        resp.del_cookie(STATE_COOKIE, path="/auth")
        self._set_session(request, resp, self.auth.make_session(name=name, via="discord", uid=str(uid),
                                                                avatar=avatar))
        raise resp

    # ------------------------------------------------------------ admin: overview

    async def me(self, request: web.Request) -> web.Response:
        s = request["session"]
        return web.json_response({
            "name": s["name"], "via": s["via"], "avatar": s.get("avatar", ""),
            "bot": self._bot_card(), "password_user": self.auth.password_user,
            "discord_ready": self._discord_ready(), "public_url": self.cfg.public_url,
            "redirect_uri": self._redirect_uri() if self.cfg.public_url else "",
            "application_id": str(self.bot.application_id or ""),
            "admins": [{"id": str(i), "name": await self._username(i)} for i in self._admin_ids()]})

    async def _username(self, uid: int) -> str | None:
        user = self.bot.get_user(uid)
        if user is None:
            try:
                user = await self.bot.fetch_user(uid)
            except discord.HTTPException:
                return None
        return user.name

    async def admin_snapshot(self, _request: web.Request) -> web.Response:
        return web.json_response(await self._snapshot())

    async def admin_learned(self, _request: web.Request) -> web.Response:
        """Per-server self-tuning + mood strategies, under the board in Overview."""
        fx = self._fx
        return web.json_response({"guilds": snapshot.learned(self.bot)
                                  + (snapshot.learned(fx, "fluxer") if fx is not None else [])})

    async def platforms(self, _request: web.Request) -> web.Response:
        """Both bots side by side for the Platforms tab: enabled (platform.mode), online, what each holds."""
        b, fx = self.bot, self._fx
        mode = b.cfg.platform.mode

        def people(store) -> int | None:
            try:
                return len(store.all_users())
            except Exception:  # noqa: BLE001
                return None

        u = b.user
        discord_side = {
            "enabled": mode in ("discord", "both"), "online": b.is_ready(),
            "user": u.name if u else None, "avatar": u.display_avatar.url if u else None,
            "servers": len(b.guilds), "voice": sum(1 for s in b.sessions.values() if s.vc.is_connected()),
            "people": people(b.profiles.store), "data": "data/", "where": "discord.com",
            "commands": "slash commands",
        }
        fx_cfg = b.cfg.fluxer
        fluxer_side = {
            "enabled": mode in ("fluxer", "both"), "online": fx is not None,
            "user": fx.user.name if fx is not None else None,
            "avatar": fx.user.display_avatar.url if fx is not None and fx.user.display_avatar else None,
            "servers": len(fx.state.guilds) if fx is not None else None,
            "voice": sum(1 for s in fx.sessions.values() if s.vc.is_connected()) if fx is not None else 0,
            "people": people(fx.profiles.store) if fx is not None else None,
            "data": str(fx_cfg.data_dir).rstrip("/") + "/",
            "where": "/".join(str(fx_cfg.api_url).split("/", 3)[2:3]) or None,
            "configured": bool(fx_cfg.token and fx_cfg.api_url),
            "commands": f"{fx_cfg.prefix} commands",
        }
        return web.json_response({"mode": mode, "discord": discord_side, "fluxer": fluxer_side})

    # ------------------------------------------------------------ admin: servers

    @property
    def _fx(self):
        """The Fluxer frontend when it's running (platform.mode fluxer/both), else None."""
        fx = getattr(self.bot, "fluxer", None)
        return fx if fx is not None and fx.user is not None else None

    def _pbot(self, request: web.Request):
        """Whose data a People request is about: ?platform=fluxer -> the Fluxer bot's own stores."""
        if request.query.get("platform") == "fluxer":
            if self._fx is None:
                raise web.HTTPNotFound(text='{"error": "Fluxer isn\'t running"}', content_type="application/json")
            return self._fx
        return self.bot

    def _guild(self, request: web.Request) -> discord.Guild:
        gid = request.match_info["gid"]
        guild = self.bot.get_guild(int(gid)) if gid.isdigit() else None
        if guild is None and gid.isdigit() and self._fx is not None:
            guild = self._fx.get_guild(int(gid))
        if guild is None:
            raise web.HTTPNotFound(text='{"error": "not in that server"}', content_type="application/json")
        return guild

    def _guild_info(self, g: discord.Guild) -> dict:
        me = g.me
        perms = me.guild_permissions
        session = self.bot.sessions.get(g.id)
        synced = self.bot.cfg.discord.guild_ids
        voice = []
        for ch in [*g.voice_channels, *g.stage_channels]:
            p = ch.permissions_for(me)
            if p.view_channel:
                voice.append({"id": str(ch.id), "name": ch.name, "people": len(ch.members),
                              "can_join": p.connect and p.speak})
        vc = session.vc.channel if session and session.vc.is_connected() else None
        return {
            "platform": "discord", "id": str(g.id), "name": g.name, "icon": g.icon.url if g.icon else None,
            "members": g.member_count, "humans": sum(not m.bot for m in g.members) if g.chunked else None,
            "owner": g.owner.name if g.owner else str(g.owner_id),
            "created": g.created_at.timestamp(), "joined": me.joined_at.timestamp() if me.joined_at else None,
            "text_channels": len(g.text_channels), "voice_channels": voice,
            "boost_tier": g.premium_tier, "admin": perms.administrator,
            "missing": [p for p in _NEEDED if not getattr(perms, p)],
            "voice": {"id": str(vc.id), "name": vc.name, "people": session.humans_in_channel()} if vc else None,
            "commands": "global" if not synced else ("synced" if g.id in synced else "missing"),
            "tuning": (self.bot.tuning.for_guild(g.id).level if self.bot.tuning.enabled else None),
        }

    def _fx_guild_info(self, g) -> dict:
        fx = self._fx
        me = g.me
        session = fx.sessions.get(g.id)
        vc = session.vc.channel if session and session.vc.is_connected() else None
        voice = [{"id": str(ch.id), "name": ch.name, "people": len(ch.members), "can_join": True}
                 for ch in sorted(g.channels.values(), key=lambda c: c.id) if ch.is_voice]
        return {
            "platform": "fluxer", "id": str(g.id), "name": g.name, "icon": None,
            "members": len(g.members), "humans": sum(not m.bot for m in g.members),
            "owner": (o.name if (o := g.get_member(g.owner_id)) else str(g.owner_id)),
            "created": None, "joined": me.joined_at.timestamp() if me and me.joined_at else None,
            "text_channels": sum(1 for c in g.channels.values() if c.type == 0), "voice_channels": voice,
            "boost_tier": 0, "admin": bool(me and me.guild_permissions.administrator), "missing": [],
            "voice": {"id": str(vc.id), "name": vc.name, "people": session.humans_in_channel()} if vc else None,
            "commands": "prefix", "prefix": fx.fx.prefix,
            "tuning": fx.tuning.for_guild(g.id).level if fx.tuning.enabled else None,
        }

    async def guilds(self, _request: web.Request) -> web.Response:
        app_id = self.bot.application_id
        fx = self._fx
        return web.json_response({
            "guilds": [self._guild_info(g) for g in sorted(self.bot.guilds, key=lambda g: g.name.lower())]
            + ([self._fx_guild_info(g) for g in sorted(fx.state.guilds.values(), key=lambda g: g.name.lower())]
               if fx else []),
            "invite": discord.utils.oauth_url(app_id, permissions=INVITE_PERMS,
                                              scopes=("bot", "applications.commands")) if app_id else None,
            "invite_admin": discord.utils.oauth_url(app_id, permissions=discord.Permissions(administrator=True),
                                                    scopes=("bot", "applications.commands")) if app_id else None,
        })

    async def leave_guild(self, request: web.Request) -> web.Response:
        guild = self._guild(request)
        body = await request.json()
        if str(body.get("confirm", "")).strip() != guild.name:
            return _err(400, "Type the server's name exactly to confirm.")
        log.warning("Dashboard: %s removed the bot from %s (%s)", self._who(request), guild.name, guild.id)
        if self._is_fx(guild):
            await self._fx.leave(guild.id)
            await self._fx.state.request("DELETE", f"/users/@me/guilds/{guild.id}")
            return web.json_response({"ok": True})
        await self.bot.leave(guild.id)
        await guild.leave()
        return web.json_response({"ok": True})

    async def voice(self, request: web.Request) -> web.Response:
        guild = self._guild(request)
        body = await request.json()
        cid = str(body.get("channel_id") or "")
        if self._is_fx(guild):
            return await self._fx_voice(request, guild, cid)
        if not cid:
            left = await self.bot.leave(guild.id)
            log.info("Dashboard: %s disconnected voice in %s", self._who(request), guild.name)
            return web.json_response({"ok": True, "left": left})
        channel = guild.get_channel(int(cid)) if cid.isdigit() else None
        if not isinstance(channel, discord.VoiceChannel | discord.StageChannel):
            return _err(400, "not a voice channel in that server")
        if not channel.permissions_for(guild.me).connect:
            return _err(403, "I don't have permission to join that channel")
        log.info("Dashboard: %s sent the bot to %s / %s", self._who(request), guild.name, channel.name)
        try:
            await self.bot.join_channel(channel)
        except Exception as e:  # noqa: BLE001
            log.exception("Dashboard voice join failed")
            return _err(500, f"couldn't join: {e}")
        return web.json_response({"ok": True})

    def _is_fx(self, guild) -> bool:
        return self._fx is not None and guild is self._fx.get_guild(guild.id)

    async def _fx_voice(self, request: web.Request, guild, cid: str) -> web.Response:
        fx = self._fx
        if not cid:
            left = await fx.leave(guild.id)
            log.info("Dashboard: %s disconnected Fluxer voice in %s", self._who(request), guild.name)
            return web.json_response({"ok": True, "left": left})
        channel = guild.channels.get(int(cid)) if cid.isdigit() else None
        if channel is None or not channel.is_voice:
            return _err(400, "not a voice channel in that server")
        log.info("Dashboard: %s sent the bot to Fluxer %s / %s", self._who(request), guild.name, channel.name)
        try:
            await fx.join_channel(channel)
        except Exception as e:  # noqa: BLE001
            log.exception("Dashboard Fluxer voice join failed")
            return _err(500, f"couldn't join: {e}")
        return web.json_response({"ok": True})

    async def sync_commands(self, request: web.Request) -> web.Response:
        guild = self._guild(request)
        if self._is_fx(guild):
            return web.json_response({"ok": True, "note": f"Fluxer uses {self._fx.fx.prefix} commands - nothing to sync."})
        ids = list(self.bot.cfg.discord.guild_ids)
        if not ids:
            return web.json_response({"ok": True, "note": "Commands are global already (discord.guild_ids is empty)."})
        if guild.id not in ids:
            await asyncio.to_thread(self.editor.save, {"discord.guild_ids": [str(i) for i in ids + [guild.id]]})
        obj = discord.Object(id=guild.id)
        self.bot.tree.copy_global_to(guild=obj)
        synced = await self.bot.tree.sync(guild=obj)
        log.info("Dashboard: %s synced %d slash commands to %s", self._who(request), len(synced), guild.name)
        return web.json_response({"ok": True, "note": f"{len(synced)} slash commands synced to {guild.name}."})

    # ------------------------------------------------------------ admin: people (what's stored per user)

    def _uid(self, request: web.Request) -> int:
        uid = request.match_info["uid"]
        if not uid.isdigit():
            raise web.HTTPNotFound(text='{"error": "no such user"}', content_type="application/json")
        return int(uid)

    def _avatar(self, uid: int) -> str | None:
        if uid in self._avatars:
            return self._avatars[uid]
        user = self.bot.get_user(uid) or next((m for g in self.bot.guilds if (m := g.get_member(uid))), None)
        return user.display_avatar.replace(size=64).url if user else None

    async def _fetch_avatar(self, uid: int) -> str | None:
        """No members intent, so most people aren't cached: look them up once (REST) and remember it."""
        if (url := self._avatar(uid)) is None:
            try:
                url = (await self.bot.fetch_user(uid)).display_avatar.replace(size=64).url
            except discord.HTTPException:
                url = None
            self._avatars[uid] = url
        return url

    async def people(self, request: web.Request) -> web.Response:
        b = self._pbot(request)
        rows = b.profiles.store.all_users()
        with b.mood._lock:  # the mood thread adds users
            mood_users = set(b.mood.state["users"])
        fx = b is not self.bot
        return web.json_response({"enabled": b.profiles.enabled, "platform": "fluxer" if fx else "discord",
                                  "fluxer": self._fx is not None, "people": [{
            "id": str(r["user_id"]), "username": r["username"], "display_name": r["display_name"],
            "avatar": self._fx_avatar(r["user_id"]) if fx else self._avatar(r["user_id"]), "messages": r["messages"], "first_seen": r["first_seen"],
            "last_seen": r["last_seen"], "profile_chars": len(r["profile"]), "pending": r["pending"],
            "opted_out": bool(r["opted_out"]), "mood": str(r["user_id"]) in mood_users,
        } for r in rows]})

    def _fx_avatar(self, uid: int) -> str | None:
        u = self._fx.get_user(uid) if self._fx else None
        return u.display_avatar.replace(size=64).url if u is not None and u.display_avatar else None

    async def person(self, request: web.Request) -> web.Response:
        uid = self._uid(request)
        b = self._pbot(request)
        store = b.profiles.store
        row = store.get(uid)
        mood = b.mood.user_state(uid)
        reminders = [{"id": j["id"], "text": j["text"], "due": j["due"], "created": j["created"],
                      "deliver": j["deliver"], "creator": j["creator_name"], "own": j["creator_id"] == uid}
                     for j in b.planner.store.reminders_for(uid)]
        lore = [{"id": r["id"], "created": r["created"], "text": r["text"]} for r in b.lore.store.about(uid)]
        if row is None and mood is None and not reminders and not lore:
            return _err(404, "nothing stored about that user")
        fx = b is not self.bot
        user = {"id": str(uid), "platform": "fluxer" if fx else "discord",
                "avatar": self._fx_avatar(uid) if fx else await self._fetch_avatar(uid)}
        if row is not None:
            user |= {"username": row["username"], "display_name": row["display_name"],
                     "roles": json.loads(row["roles"] or "[]"), "first_seen": row["first_seen"],
                     "last_seen": row["last_seen"], "messages": row["messages"], "profile": row["profile"],
                     "profile_updated": row["profile_updated"], "opted_out": bool(row["opted_out"]),
                     "avatar_desc": row["avatar_desc"],
                     "pending": [{"ts": p["ts"], "line": p["line"], "own": bool(p["own"])}
                                 for p in store.pending_lines(uid)]}
        else:
            u = self._fx.get_user(uid) if fx else None
            user["username"] = (u.name if u else "") if fx else (await self._username(uid) or "")
        if mood:
            base = mood.get("base") or {}
            user["tone"] = {"samples": base.get("n", 0), "loudness_db": base.get("db_m"),
                            "words_per_s": base.get("rate_m")} if base else None
            user["thresholds"] = mood.get("th") or {}
        user["reminders"] = reminders
        user["lore"] = lore
        got = self.bot.links.linked("fluxer" if fx else "discord", uid)
        user["linked"] = {"platform": "discord" if fx else "fluxer", "id": str(got[0]), "name": got[1]} if got else None
        return web.json_response(user)

    async def forget_person(self, request: web.Request) -> web.Response:
        uid = self._uid(request)
        b = self._pbot(request)
        b.profiles.store.forget(uid)
        b.mood.forget(uid)
        b.lore.forget(uid)
        self.bot.links.unlink("fluxer" if b is not self.bot else "discord", uid)
        log.warning("Dashboard: %s deleted the stored profile of user %s", self._who(request), uid)
        return web.json_response({"ok": True})

    async def person_profiling(self, request: web.Request) -> web.Response:
        uid = self._uid(request)
        enabled = bool((await request.json()).get("enabled"))
        b = self._pbot(request)
        b.profiles.store.set_opted_out(uid, not enabled)
        if not enabled:
            b.lore.forget(uid)
        log.warning("Dashboard: %s turned profiling %s for user %s", self._who(request), "on" if enabled else "off", uid)
        return web.json_response({"ok": True})

    # ------------------------------------------------------------ admin: live view

    async def live(self, request: web.Request) -> web.Response:
        """Feed lines after ?after=<seq>, plus each server's voice state and mute. `reset` = the bot restarted
        (sequence numbers started over), so the page should drop what it has."""
        q = request.query.get("after", "0")
        after = int(q) if q.isdigit() else 0
        seq = self.bot._live_seq
        reset = after > seq
        entries = [e for e in self.bot.live if e["seq"] > (0 if reset else after)]
        servers = []
        for g in sorted(self.bot.guilds, key=lambda g: g.name.lower()):
            s = self.bot.sessions.get(g.id)
            ch = s.vc.channel if s and s.vc.is_connected() else None
            servers.append({"id": str(g.id), "name": g.name, "icon": g.icon.url if g.icon else None,
                            "platform": "discord", "muted": self.bot.is_muted(g.id),
                            "voice": {"channel": ch.name, "people": s.humans_in_channel(),
                                      "speaking": s.vc.is_playing()} if ch else None})
        if (fx := self._fx) is not None:
            for g in sorted(fx.state.guilds.values(), key=lambda g: g.name.lower()):
                s = fx.sessions.get(g.id)
                ch = s.vc.channel if s and s.vc.is_connected() else None
                servers.append({"id": str(g.id), "name": g.name, "icon": None, "platform": "fluxer",
                                "muted": fx.is_muted(g.id),
                                "voice": {"channel": ch.name, "people": s.humans_in_channel(),
                                          "speaking": s.vc.is_playing()} if ch else None})
        return web.json_response({"entries": entries, "seq": seq, "reset": reset, "servers": servers})

    async def mute(self, request: web.Request) -> web.Response:
        guild = self._guild(request)
        body = await request.json()
        muted = bool(body.get("muted"))
        (self._fx if self._is_fx(guild) else self.bot).set_muted(guild.id, muted)
        log.warning("Dashboard: %s %s the bot in %s", self._who(request), "muted" if muted else "unmuted", guild.name)
        return web.json_response({"ok": True, "muted": muted})

    # ------------------------------------------------------------ admin: settings, logs, restart

    async def get_config(self, _request: web.Request) -> web.Response:
        return web.json_response({"fields": self.editor.schema(), "path": str(self.editor.path)})

    async def save_config(self, request: web.Request) -> web.Response:
        body = await request.json()
        changes = body.get("changes")
        if not isinstance(changes, dict):
            return _err(400, "bad request")
        try:
            result = await asyncio.to_thread(self.editor.save, changes)
        except ValueError as e:
            return _err(400, str(e))
        if result["saved"]:
            log.info("Dashboard: %s changed settings: %s", self._who(request), ", ".join(result["saved"]))
            self._status = None
        return web.json_response(result)

    async def set_platform(self, request: web.Request) -> web.Response:
        """Discord / Fluxer / both (platform.mode). Validated + saved like any setting; applies on restart."""
        mode = str((await request.json()).get("mode") or "").lower()
        try:
            result = await asyncio.to_thread(self.editor.save, {"platform.mode": mode})
        except ValueError as e:
            return _err(400, str(e))
        log.info("Dashboard: %s set platform mode to %s", self._who(request), mode)
        return web.json_response({**result, "mode": mode})

    async def restart(self, request: web.Request) -> web.Response:
        log.warning("Dashboard: restart requested by %s", self._who(request))
        self.bot.restart_requested = True
        asyncio.get_running_loop().call_later(0.5, lambda: asyncio.ensure_future(self.bot.close()))
        return web.json_response({"ok": True})

    async def logs(self, request: web.Request) -> web.Response:
        n = max(20, min(2000, int(request.query.get("n", "300") or 300)))
        q = request.query.get("q", "").strip().lower()
        try:
            proc = await asyncio.create_subprocess_exec(
                "journalctl", "-u", str(self.cfg.service_name), "-n", str(n if not q else 5000), "--no-pager",
                "-o", "cat", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, errs = await asyncio.wait_for(proc.communicate(), 10)
        except (OSError, asyncio.TimeoutError) as e:
            return _err(500, f"journalctl failed: {e}")
        lines = out.decode(errors="replace").splitlines()
        if q:
            lines = [line for line in lines if q in line.lower()][-n:]
        if not lines and errs:
            return _err(500, errs.decode(errors="replace")[:300])
        return web.json_response({"lines": lines})

    # ------------------------------------------------------------ admin: account

    async def set_password(self, request: web.Request) -> web.Response:
        body = await request.json()
        user, password = str(body.get("username", "")).strip(), str(body.get("password", ""))
        if not user or len(user) > 64:
            return _err(400, "Pick a username (up to 64 characters).")
        if len(password) < 10:
            return _err(400, "Use at least 10 characters for the password.")
        async with self._scrypt:
            await asyncio.to_thread(self.auth.set_password, user, password)
        log.warning("Dashboard: %s set the local password login (user %r)", self._who(request), user)
        resp = web.json_response({"ok": True})
        if request["session"].get("via") == "password":  # the change signed out old password sessions
            self._set_session(request, resp, self.auth.make_session(name=user, via="password"))
        return resp

    async def signout_all(self, request: web.Request) -> web.Response:
        log.warning("Dashboard: %s signed out every session", self._who(request))
        self.auth.sign_out_everyone()
        resp = web.json_response({"ok": True})
        resp.del_cookie(SESSION_COOKIE, path="/")
        return resp
