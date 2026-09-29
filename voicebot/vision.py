"""Letting the LLM see: images people post and profile pictures.

Images go to the model inline (OpenAI-style image_url parts) for the reply that's about them. Chat
history stays plain text: an image is an "[image]" placeholder there, swapped for a short caption
once the reply is out, so later turns still know what was posted without re-sending pixels.
Profile pictures are captioned once per avatar and remembered in the profiles database.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
import discord
from PIL import Image

if TYPE_CHECKING:
    from .bot import VoiceBot

log = logging.getLogger("voicebot.vision")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
PLACEHOLDER = "[image]"


@dataclass
class PostedImage:
    key: str      # cache key (attachment id / URL)
    label: str    # how the model is told about it: "image posted by X", "X's profile picture"
    data: bytes   # raw downloaded bytes
    url: str = ""  # data: URL once prepared


def _is_image(filename: str, content_type: str | None) -> bool:
    ct = content_type or ""
    return (ct.startswith("image/") and "svg" not in ct) or filename.lower().endswith(IMAGE_EXTS)


def _to_data_url(data: bytes, max_side: int) -> str:
    """First frame, RGB, longest side <= max_side, JPEG. Handles GIF/WebP/PNG-with-alpha alike."""
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)
        im.draft("RGB", (max_side, max_side))  # JPEG: decode at reduced scale - far less CPU and RAM
        im = im.convert("RGBA")  # before resizing: palette images (GIFs) only resize nearest-neighbour
        im.thumbnail((max_side, max_side))
        out = Image.new("RGB", im.size, (255, 255, 255))
        out.paste(im, mask=im.getchannel("A"))
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_parts(text: str, images: list[PostedImage]) -> list[dict]:
    """A user message with the images after the text, each with a label saying what it is."""
    parts: list[dict] = [{"type": "text", "text": text}]
    for img in images:
        parts.append({"type": "text", "text": f"({img.label}:)"})
        parts.append({"type": "image_url", "image_url": {"url": img.url}})
    return parts


class Vision:
    def __init__(self, bot: "VoiceBot"):
        self.bot = bot
        self.cfg = bot.cfg.vision
        self.enabled = bool(self.cfg.enabled)
        self._session: aiohttp.ClientSession | None = None
        self._captions: OrderedDict[str, str] = OrderedDict()
        self._avatar_queue: asyncio.Queue | None = None
        self._avatar_pending: set[int] = set()
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro, name: str) -> None:
        """Fire-and-forget, but keep a reference: the event loop only holds tasks weakly."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ fetching

    def images_in(self, message: discord.Message) -> list[tuple[str, object]]:
        """(cache key, source) for each image in a message: attachments, then link-preview images."""
        found: list[tuple[str, object]] = []
        for a in message.attachments:
            if _is_image(a.filename, a.content_type) and a.size <= int(self.cfg.max_download_mb) * 2**20:
                found.append((f"att:{a.id}", a))
        for e in message.embeds:
            # Only Discord's media proxy: never make the bot fetch an arbitrary URL someone linked.
            media = e.image if e.image and e.image.proxy_url else e.thumbnail if e.thumbnail and e.thumbnail.proxy_url else None
            if media:
                found.append((media.url, media.proxy_url))
        return found

    async def _download(self, source) -> bytes:
        if isinstance(source, (discord.Attachment, discord.Asset)):
            return await source.read()
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        limit = int(self.cfg.max_download_mb) * 2**20
        async with self._session.get(source) as r:
            r.raise_for_status()
            # content.read(n) only returns what has arrived so far, so read until EOF or the cap.
            data = bytearray()
            async for chunk in r.content.iter_chunked(64 * 1024):
                data += chunk
                if len(data) > limit:
                    raise ValueError(f"image over {self.cfg.max_download_mb} MB")
            return bytes(data)

    async def fetch(self, message: discord.Message) -> list[PostedImage]:
        if not self.enabled:
            return []
        found = self.images_in(message)
        got = await asyncio.gather(*(self._download(src) for _, src in found), return_exceptions=True)
        out = []
        for (key, _), data in zip(found, got):
            if isinstance(data, BaseException):
                log.warning("Couldn't download image %s: %s", key, data)
            else:
                out.append(PostedImage(key, f"image posted by {message.author.display_name}", data))
        return out

    async def fetch_avatar(self, member) -> PostedImage | None:
        asset = member.display_avatar.replace(size=512, static_format="png")
        try:
            return PostedImage(f"avatar:{asset.key}", f"{member.display_name}'s profile picture", await asset.read())
        except discord.HTTPException as e:
            log.warning("Couldn't download avatar of %s: %s", member.display_name, e)
            return None

    async def prepare(self, images: list[PostedImage]) -> list[PostedImage]:
        """Resize/convert off the event loop. Drops anything Pillow can't read."""
        loop = asyncio.get_running_loop()
        todo = [img for img in images if not img.url]
        urls = await asyncio.gather(
            *(loop.run_in_executor(None, _to_data_url, img.data, int(self.cfg.max_side)) for img in todo),
            return_exceptions=True)
        for img, url in zip(todo, urls):
            if isinstance(url, BaseException):
                log.warning("Unreadable image %s: %s", img.key, url)
            else:
                img.url = url
        return [img for img in images if img.url]

    # ------------------------------------------------------------------ captions

    def endpoint(self) -> str:
        return self.cfg.endpoint or self.bot.llm.text_endpoint

    async def caption(self, img: PostedImage, prompt: str | None = None) -> str:
        if img.key in self._captions:
            self._captions.move_to_end(img.key)
            return self._captions[img.key]
        await self.prepare([img])
        if not img.url:
            return ""
        t0 = time.perf_counter()
        text = await self.bot.llm.complete(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img.url}},
                                          {"type": "text", "text": prompt or self.cfg.caption_prompt}]}],
            endpoint=self.endpoint(), temperature=0.2, max_tokens=120, purpose="vision",
        )
        text = " ".join(text.split())
        log.info("👁 %s (%.1fs): %s", img.key, time.perf_counter() - t0, text)
        self._captions[img.key] = text
        while len(self._captions) > 200:
            self._captions.popitem(last=False)
        return text

    def caption_later(self, entries: list[tuple[dict, PostedImage]]) -> None:
        """In the background, replace each history entry's first "[image]" with a caption, in order."""
        if entries:
            self._spawn(self._fill_captions(entries), "image-captions")

    async def _fill_captions(self, entries: list[tuple[dict, PostedImage]]) -> None:
        for entry, img in entries:
            try:
                text = await self.caption(img)
            except Exception as e:  # noqa: BLE001
                log.warning("Captioning %s failed: %s", img.key, e)
                text = ""
            if text:
                entry["content"] = entry["content"].replace(PLACEHOLDER, f"[image: {text}]", 1)

    # ------------------------------------------------------------------ profile pictures

    def avatar_note(self, member, row) -> str:
        """Remembered description of this member's current profile picture ('' if none yet), given their
        profiles-db row (or None). Queues captioning when the avatar is new or changed."""
        if not (self.enabled and self.cfg.avatars) or member is None:
            return ""
        if getattr(member, "avatar", None) is None and getattr(member, "guild_avatar", None) is None:
            return ""  # Discord's default avatar
        if row and row["opted_out"]:
            return ""
        key = member.display_avatar.key
        if row and row["avatar_key"] == key:
            return row["avatar_desc"]
        self._queue_avatar(member)
        return row["avatar_desc"] if row else ""  # the old one until the new caption is ready

    def _queue_avatar(self, member) -> None:
        if member.id in self._avatar_pending:
            return
        if self._avatar_queue is None:
            self._avatar_queue = asyncio.Queue()
            self._spawn(self._avatar_worker(), "avatar-captions")
        self._avatar_pending.add(member.id)
        self._avatar_queue.put_nowait(member)

    async def _avatar_worker(self) -> None:
        while True:
            member = await self._avatar_queue.get()
            try:
                # Stay out of the way of a reply that's being generated right now.
                while time.monotonic() - self.bot.profiles.last_activity < 2:
                    await asyncio.sleep(1)
                img = await self.fetch_avatar(member)
                if img:
                    desc = await self.caption(img, self.cfg.avatar_prompt)
                    if desc:
                        self.bot.profiles.store.set_avatar(member.id, member.display_avatar.key, desc)
            except Exception as e:  # noqa: BLE001
                log.warning("Avatar caption for %s failed: %s", member.display_name, e)
            finally:
                self._avatar_pending.discard(member.id)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
