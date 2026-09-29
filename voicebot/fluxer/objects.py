"""Fluxer data as objects shaped like the discord.py ones the shared code reads (member.name/.roles/.voice,
guild.members/.get_member/.owner_id, channel.members/.send, ...), so profiles, reminders, member lookups and
VoiceSession work on Fluxer unchanged. Fluxer's gateway/REST payloads are Discord-shaped, which keeps this thin.

FxState is the cache: filled from READY/GUILD_CREATE and kept current from gateway events."""
from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
import discord

log = logging.getLogger("voicebot.fluxer")

ADMINISTRATOR = 1 << 3
MANAGE_GUILD = 1 << 5
POLL_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


class FxHTTPError(discord.HTTPException):
    """A failed Fluxer REST call. A discord.HTTPException, so the shared code's `except discord.HTTPException`
    handles Fluxer failures the same way."""

    def __init__(self, status: int, text: str, code: int = 0):  # noqa: D107 - no super().__init__: needs a response
        self.status = status
        self.text = text
        self.code = code
        self.response = None
        Exception.__init__(self, f"{status} {text}".strip())


def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _when(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(eq=False)
class FxRole:
    id: int
    name: str
    position: int
    permissions: int
    guild_id: int

    def is_default(self) -> bool:
        return self.id == self.guild_id or self.name == "@everyone"


@dataclass
class FxPermissions:
    administrator: bool = False
    manage_guild: bool = False


@dataclass
class FxPoll:
    """Fluxer has no bot polls: posted as a numbered list with one reaction per choice (see FxChannel.send)."""
    question: str
    options: list[str]
    multiple: bool = False
    ends: float = 0.0


@dataclass
class FxVoiceState:
    channel: "FxChannel | None"
    connection_id: str = ""
    self_mute: bool = False
    self_deaf: bool = False
    self_stream: bool = False
    self_video: bool = False


class FxAsset:
    """An avatar, shaped like discord.Asset for vision.py (key / replace / read)."""

    def __init__(self, state: "FxState", uid: int, avatar: str, size: int = 512):
        self._state, self._uid, self.key, self._size = state, uid, avatar, size

    @property
    def url(self) -> str:
        return f"{self._state.media_url}/avatars/{self._uid}/{self.key}.png?size={self._size}"

    def replace(self, size: int = 512, **_) -> "FxAsset":
        return FxAsset(self._state, self._uid, self.key, size)

    async def read(self) -> bytes:
        return await self._state.download(self.url)


class FxUser:
    """A Fluxer account. display_name is the username, same as on Discord (see bot.py's display_name patch)."""

    def __init__(self, state: "FxState", data: dict):
        self._state = state
        self.id = _int(data.get("id"))
        self.update(data)

    def update(self, data: dict) -> None:
        self.name = data.get("username") or self.__dict__.get("name") or str(self.id)
        self.global_name = data.get("global_name")
        self.bot = bool(data.get("bot", self.__dict__.get("bot", False)))
        if "avatar" in data:
            self.avatar = data.get("avatar") or None  # hash; None = default avatar (vision.py skips those)

    avatar: str | None = None
    guild_avatar = None

    @property
    def display_avatar(self) -> "FxAsset | None":
        return FxAsset(self._state, self.id, self.avatar) if self.avatar else None

    display_name = property(lambda self: self.name)
    mention = property(lambda self: f"<@{self.id}>")
    activities: tuple = ()
    voice = None
    roles: list = []

    @property
    def mutual_guilds(self) -> list["FxGuild"]:
        return [g for g in self._state.guilds.values() if self.id in g._members]

    async def send(self, content: str = "", **kwargs) -> "FxMessage":
        dm = await self._state.request("POST", "/users/@me/channels", json={"recipient_id": str(self.id)})
        channel = self._state.channel_from(dm)
        return await channel.send(content, **kwargs)

    def __eq__(self, other) -> bool:
        return getattr(other, "id", None) == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"<FxUser {self.name} {self.id}>"


class FxMember(FxUser):
    def __init__(self, state: "FxState", guild: "FxGuild", data: dict):
        self.guild = guild
        self._role_ids: list[int] = []
        self.joined_at = None
        super().__init__(state, data.get("user") or {})
        self.update_member(data)

    def update_member(self, data: dict) -> None:
        if data.get("user"):
            self.update(data["user"])
        if "roles" in data:
            self._role_ids = [_int(r) for r in data.get("roles") or []]
        self.joined_at = _when(data.get("joined_at")) or self.joined_at

    @property
    def roles(self) -> list[FxRole]:
        """Lowest first like discord.py (member_roles() reverses it), @everyone included."""
        g = self.guild
        roles = [g.roles[r] for r in self._role_ids if r in g.roles]
        if g.id in g.roles:
            roles.append(g.roles[g.id])
        return sorted(roles, key=lambda r: r.position)

    @property
    def guild_permissions(self) -> FxPermissions:
        perms = 0
        for r in self.roles:
            perms |= r.permissions
        admin = bool(perms & ADMINISTRATOR) or self.guild.owner_id == self.id
        return FxPermissions(administrator=admin, manage_guild=admin or bool(perms & MANAGE_GUILD))

    @property
    def voice(self) -> FxVoiceState | None:  # type: ignore[override]
        return self.guild.voice_states.get(self.id)

    def __repr__(self) -> str:
        return f"<FxMember {self.name} {self.id} in {self.guild.name}>"


class FxGuild:
    def __init__(self, state: "FxState", data: dict):
        self._state = state
        self.id = _int(data.get("id"))
        self.name = ""
        self.owner_id = 0
        self.roles: dict[int, FxRole] = {}
        self._members: dict[int, FxMember] = {}
        self.channels: dict[int, FxChannel] = {}
        self.voice_states: dict[int, FxVoiceState] = {}
        self.update(data)

    def update(self, data: dict) -> None:
        props = data.get("properties") or data
        self.name = props.get("name") or self.name
        self.owner_id = _int(props.get("owner_id")) or self.owner_id
        for r in data.get("roles") or []:
            self.set_role(r)
        for c in data.get("channels") or []:
            ch = self._state.channel_from(c, self)
            self.channels[ch.id] = ch
        for m in data.get("members") or []:
            self.set_member(m)
        for v in data.get("voice_states") or []:
            self.set_voice_state(v)

    def set_role(self, r: dict) -> None:
        self.roles[_int(r["id"])] = FxRole(_int(r["id"]), r.get("name") or "", _int(r.get("position")),
                                           _int(r.get("permissions")), self.id)

    def set_member(self, m: dict) -> FxMember:
        uid = _int((m.get("user") or {}).get("id"))
        member = self._members.get(uid)
        if member is None:
            member = self._members[uid] = FxMember(self._state, self, m)
            self._state.users[uid] = member
        else:
            member.update_member(m)
        return member

    def set_voice_state(self, v: dict) -> None:
        uid = _int(v.get("user_id"))
        if v.get("member"):
            self.set_member(v["member"])
        cid = _int(v.get("channel_id"))
        if not cid:
            self.voice_states.pop(uid, None)
            return
        self.voice_states[uid] = FxVoiceState(self.channels.get(cid), v.get("connection_id") or "",
                                              bool(v.get("self_mute")), bool(v.get("self_deaf")),
                                              bool(v.get("self_stream")), bool(v.get("self_video")))

    @property
    def members(self) -> list[FxMember]:
        return list(self._members.values())

    def get_member(self, uid: int) -> FxMember | None:
        return self._members.get(uid)

    @property
    def me(self) -> FxMember | None:
        return self._members.get(self._state.user_id)

    async def query_members(self, query: str = "", limit: int = 5, **_) -> list[FxMember]:
        """The whole member list is cached (GUILD_CREATE has it), so this is a local search."""
        q = query.lower()
        return [m for m in self._members.values() if m.name.lower().startswith(q)][:limit]

    def __repr__(self) -> str:
        return f"<FxGuild {self.name} {self.id}>"

    __str__ = lambda self: self.name  # noqa: E731


class FxChannel:
    TEXT, DM, VOICE, GROUP_DM, CATEGORY = 0, 1, 2, 3, 4

    def __init__(self, state: "FxState", data: dict, guild: FxGuild | None = None):
        self._state = state
        self.id = _int(data.get("id"))
        self.guild = guild
        self.update(data)

    def update(self, data: dict) -> None:
        self.name = data.get("name") or getattr(self, "name", "") or "DM"
        self.type = _int(data.get("type"))

    @property
    def is_voice(self) -> bool:
        return self.type == self.VOICE

    @property
    def members(self) -> list[FxMember]:
        """Voice channel: who's in it right now (like discord.py's VoiceChannel.members)."""
        if self.guild is None:
            return []
        return [m for uid, v in self.guild.voice_states.items()
                if v.channel is self and (m := self.guild.get_member(uid)) is not None]

    async def send(self, content: str = "", *, file: "FxFile | None" = None, reference: "FxMessage | None" = None,
                   poll=None, allowed_mentions=None, **_) -> "FxMessage":
        """Never pings anyone unless allowed_mentions says so (reminders), like the Discord bot. Voice channels
        (a voice session's clips, reminders set by voice) post in the session's text channel instead."""
        if self.is_voice and self.guild is not None and self._state.text_for_voice is not None:
            target = self._state.text_for_voice(self.guild)
            if target is not None and target is not self:
                return await target.send(content, file=file, poll=poll, allowed_mentions=allowed_mentions)
        if poll is not None:
            return await self._send_poll(content, poll)
        body: dict[str, Any] = {"content": content[:2000], "allowed_mentions": _mentions_payload(allowed_mentions)}
        if reference is not None:
            body["message_reference"] = {"message_id": str(reference.id), "channel_id": str(self.id)}
            body["allowed_mentions"]["replied_user"] = False
        if file is not None:
            form = aiohttp.FormData()
            body["attachments"] = [{"id": 0, "filename": file.filename}]
            form.add_field("payload_json", json.dumps(body), content_type="application/json")
            form.add_field("files[0]", file.fp.getvalue(), filename=file.filename)
            data = await self._state.request("POST", f"/channels/{self.id}/messages", data=form)
        else:
            data = await self._state.request("POST", f"/channels/{self.id}/messages", json=body)
        return self._state.message_from(data)

    async def _send_poll(self, content: str, poll: FxPoll) -> "FxMessage":
        lines = [f"{content}".strip(), f"📊 **{poll.question}**"]
        lines += [f"{POLL_EMOJI[i]} {opt}" for i, opt in enumerate(poll.options[:10])]
        how = "pick as many as you like" if poll.multiple else "one pick each"
        lines.append(f"-# Vote with the reactions ({how})" + (f" · closes <t:{int(poll.ends)}:R>" if poll.ends else ""))
        msg = await self.send("\n".join(x for x in lines if x))
        for emoji in POLL_EMOJI[: len(poll.options[:10])]:
            await self.add_reaction(msg.id, emoji)
            await asyncio.sleep(0.3)  # reactions are rate-limited
        return msg

    async def add_reaction(self, message_id: int, emoji: str) -> None:
        from urllib.parse import quote
        await self._state.request("PUT", f"/channels/{self.id}/messages/{message_id}/reactions/{quote(emoji)}/@me")

    async def fetch_message(self, message_id: int) -> "FxMessage":
        return self._state.message_from(await self._state.request("GET", f"/channels/{self.id}/messages/{message_id}"),
                                        self)

    def typing(self) -> "_Typing":
        return _Typing(self)

    async def history(self, limit: int = 10, after: int = 0, before: int = 0) -> list["FxMessage"]:
        params = {"limit": str(limit)}
        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)
        data = await self._state.request("GET", f"/channels/{self.id}/messages", params=params)
        return [self._state.message_from(m, self) for m in data or []]

    def __repr__(self) -> str:
        return f"<FxChannel #{self.name} {self.id}>"

    __str__ = lambda self: self.name  # noqa: E731


def _mentions_payload(am) -> dict:
    """discord.AllowedMentions (what reminders pass to ping people) -> the JSON Fluxer takes."""
    users = getattr(am, "users", None)
    out: dict[str, Any] = {"parse": []}
    if isinstance(users, list):
        out["users"] = [str(getattr(u, "id", u)) for u in users]
    elif users is True:
        out["parse"].append("users")
    return out


class _Typing:
    """`async with channel.typing():` - refreshes the typing indicator every 8 s until the block ends."""

    def __init__(self, channel: FxChannel):
        self.channel = channel
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.channel._state.request("POST", f"/channels/{self.channel.id}/typing")
            except Exception:  # noqa: BLE001 - typing is cosmetic
                pass
            await asyncio.sleep(8)

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()


@dataclass
class FxAttachment:
    id: int
    filename: str
    content_type: str | None
    size: int
    url: str
    _state: "FxState | None" = None

    async def read(self) -> bytes:
        return await self._state.download(self.url)


@dataclass
class FxFile:
    fp: io.BytesIO
    filename: str


@dataclass(eq=False)
class FxMessage:
    id: int
    channel: FxChannel
    author: FxUser
    content: str
    guild: FxGuild | None
    mentions: list[FxUser] = field(default_factory=list)
    attachments: list[FxAttachment] = field(default_factory=list)
    reference_id: int = 0
    referenced: "FxMessage | None" = None
    created_at: datetime | None = None
    type: int = 0
    reactions: dict[str, int] = field(default_factory=dict)  # emoji -> count (including ours)
    embeds: list = field(default_factory=list)  # link previews aren't used on Fluxer (vision.images_in reads this)

    async def reply(self, content: str, **_) -> FxMessage:
        return await self.channel.send(content, reference=self)


class FxState:
    """Everything the gateway has told us, plus the REST helper the objects use."""

    def __init__(self, api_url: str, token: str):
        self.api_url = api_url.rstrip("/")
        self.media_url = self.api_url.split("/api", 1)[0] + "/media"  # avatars: {media}/avatars/<id>/<hash>.png
        self._token = token
        self._session: aiohttp.ClientSession | None = None
        self.text_for_voice = None  # set by FluxerBot: guild -> text channel for a voice channel's messages
        self.user_id = 0
        self.guilds: dict[int, FxGuild] = {}
        self.channels: dict[int, FxChannel] = {}
        self.users: dict[int, FxUser] = {}

    async def request(self, method: str, path: str, **kwargs) -> Any:
        """REST call on the Fluxer API. Raises FxHTTPError (a discord.HTTPException) on failure."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        headers = {"Authorization": f"Bot {self._token}", "User-Agent": "Static (voicebot)"}
        try:
            async with self._session.request(method, self.api_url + path, headers=headers,
                                             timeout=aiohttp.ClientTimeout(total=30), **kwargs) as r:
                text = await r.text()
                status = r.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise FxHTTPError(0, f"can't reach Fluxer: {e!r}") from None
        if status >= 400:
            try:
                err = json.loads(text)
                message, code = str(err.get("message") or text), _int(err.get("code"))
            except (ValueError, AttributeError):
                message, code = text, 0
            raise FxHTTPError(status, message[:300], code)
        if status == 204 or not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return text

    async def download(self, url: str, limit_mb: int = 20) -> bytes:
        """Media from our own Fluxer server only (attachments, avatars) - never an arbitrary URL."""
        if not url.startswith(self.api_url.split("/api", 1)[0] + "/"):
            raise FxHTTPError(0, "not a Fluxer media URL")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status >= 400:
                raise FxHTTPError(r.status, "media download failed")
            data = bytearray()
            async for chunk in r.content.iter_chunked(64 * 1024):
                data += chunk
                if len(data) > limit_mb * 2**20:
                    raise FxHTTPError(413, f"over {limit_mb} MB")
            return bytes(data)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    # ------------------------------------------------------------------ building objects

    def channel_from(self, data: dict, guild: FxGuild | None = None) -> FxChannel:
        cid = _int(data.get("id"))
        guild = guild or self.guilds.get(_int(data.get("guild_id")))
        ch = self.channels.get(cid)
        if ch is None:
            ch = self.channels[cid] = FxChannel(self, data, guild)
        else:
            ch.update(data)
            ch.guild = guild or ch.guild
        if guild is not None:
            guild.channels[cid] = ch
        return ch

    def user_from(self, data: dict, guild: FxGuild | None = None, member: dict | None = None) -> FxUser:
        uid = _int(data.get("id"))
        if guild is not None:
            m = guild.get_member(uid)
            if m is None or member:
                m = guild.set_member({**(member or {}), "user": data})
            return m
        user = self.users.get(uid)
        if user is None:
            user = self.users[uid] = FxUser(self, data)
        else:
            user.update(data)
        return user

    def message_from(self, data: dict, channel: FxChannel | None = None) -> FxMessage:
        channel = channel or self.channels.get(_int(data.get("channel_id")))
        if channel is None:  # a DM we haven't seen yet
            channel = self.channel_from({"id": data.get("channel_id"), "type": FxChannel.DM})
        guild = channel.guild or self.guilds.get(_int(data.get("guild_id")))
        author = self.user_from(data.get("author") or {}, guild, data.get("member"))
        mentions = [self.user_from(u, guild, u.get("member")) for u in data.get("mentions") or []]
        atts = [FxAttachment(_int(a.get("id")), a.get("filename") or "", a.get("content_type"), _int(a.get("size")),
                             a.get("url") or "", self) for a in data.get("attachments") or []]
        ref = data.get("referenced_message")
        ref_id = _int((data.get("message_reference") or {}).get("message_id"))
        reactions = {(r.get("emoji") or {}).get("name") or "": _int(r.get("count")) for r in data.get("reactions") or []}
        return FxMessage(_int(data.get("id")), channel, author, data.get("content") or "", guild, mentions, atts,
                         ref_id, self.message_from(ref, channel) if isinstance(ref, dict) else None,
                         _when(data.get("timestamp")), _int(data.get("type")), reactions)

    # ------------------------------------------------------------------ gateway events

    def on_ready(self, data: dict) -> None:
        self.user_id = _int((data.get("user") or {}).get("id"))
        for g in data.get("guilds") or []:
            if not g.get("unavailable"):
                self.on_guild(g)

    def on_guild(self, data: dict) -> FxGuild:
        gid = _int(data.get("id"))
        guild = self.guilds.get(gid)
        if guild is None:
            guild = self.guilds[gid] = FxGuild(self, data)
        else:
            guild.update(data)
        return guild

    def on_event(self, event: str, data: Any) -> None:
        """Keeps the cache current. Unknown events are ignored."""
        if not isinstance(data, dict):
            return
        gid = _int(data.get("guild_id"))
        guild = self.guilds.get(gid)
        if event in ("GUILD_CREATE", "GUILD_UPDATE"):
            self.on_guild(data)
        elif event == "GUILD_DELETE":
            self.guilds.pop(_int(data.get("id")), None)
        elif event in ("CHANNEL_CREATE", "CHANNEL_UPDATE"):
            self.channel_from(data)
        elif event == "CHANNEL_DELETE":
            ch = self.channels.pop(_int(data.get("id")), None)
            if ch is not None and ch.guild is not None:
                ch.guild.channels.pop(ch.id, None)
        elif guild is None:
            return
        elif event in ("GUILD_MEMBER_ADD", "GUILD_MEMBER_UPDATE"):
            guild.set_member(data)
        elif event == "GUILD_MEMBER_REMOVE":
            guild._members.pop(_int((data.get("user") or {}).get("id")), None)
        elif event in ("GUILD_ROLE_CREATE", "GUILD_ROLE_UPDATE") and data.get("role"):
            guild.set_role(data["role"])
        elif event == "GUILD_ROLE_DELETE":
            guild.roles.pop(_int(data.get("role_id")), None)
        elif event == "VOICE_STATE_UPDATE":
            guild.set_voice_state(data)
