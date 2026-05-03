"""
LINE platform adapter for Hermes Agent.

Configuration in config.yaml:

    platforms:
      line:
        enabled: true
        extra:
          channel_access_token: "..."
          channel_secret: "..."
          webhook_host: "0.0.0.0"
          webhook_port: 8646
          webhook_path: "/line/webhook"

Or via environment variables:
    LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET,
    LINE_WEBHOOK_HOST, LINE_WEBHOOK_PORT, LINE_WEBHOOK_PATH,
    LINE_ALLOWED_USERS, LINE_ALLOW_ALL_USERS
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from aiohttp import ClientSession, ClientTimeout, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment]
    ClientTimeout = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8646
DEFAULT_PATH = "/line/webhook"
LINE_API_BASE = "https://api.line.me"
LINE_DATA_API_BASE = "https://api-data.line.me"
REPLY_TOKEN_TTL_SECONDS = 25.0
MAX_MESSAGE_LENGTH = 5000


def _normalize_path(path: str) -> str:
    value = (path or DEFAULT_PATH).strip()
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_or_extra(extra: Dict[str, Any], env_name: str, *keys: str, default: Any = "") -> Any:
    value = os.getenv(env_name)
    if value not in (None, ""):
        return value
    for key in keys:
        value = extra.get(key)
        if value not in (None, ""):
            return value
    return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def verify_signature(body: bytes, channel_secret: str, signature: str) -> bool:
    """Verify LINE's base64 HMAC-SHA256 webhook signature."""
    if not body or not channel_secret or not signature:
        return False
    digest = hmac.new(
        channel_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    computed = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(computed, signature)


def check_requirements() -> bool:
    """Return True when the adapter's Python dependencies are available."""
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    """Validate that enough LINE credentials are configured."""
    extra = getattr(config, "extra", {}) or {}
    token = (
        os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        or os.getenv("LINE_ACCESS_TOKEN")
        or getattr(config, "token", None)
        or extra.get("channel_access_token")
        or extra.get("access_token")
    )
    secret = os.getenv("LINE_CHANNEL_SECRET") or extra.get("channel_secret") or extra.get("secret")
    return bool(AIOHTTP_AVAILABLE and token and secret)


def is_connected(config) -> bool:
    return validate_config(config)


class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API adapter."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("line"))
        extra = config.extra or {}

        self.channel_access_token = (
            os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
            or os.getenv("LINE_ACCESS_TOKEN")
            or config.token
            or extra.get("channel_access_token")
            or extra.get("access_token")
            or ""
        )
        self.channel_secret = (
            os.getenv("LINE_CHANNEL_SECRET")
            or extra.get("channel_secret")
            or extra.get("secret")
            or ""
        )
        self._host = str(_env_or_extra(extra, "LINE_WEBHOOK_HOST", "webhook_host", "host", default=DEFAULT_HOST))
        self._port = _safe_int(_env_or_extra(extra, "LINE_WEBHOOK_PORT", "webhook_port", "port", default=DEFAULT_PORT), DEFAULT_PORT)
        self._path = _normalize_path(str(_env_or_extra(extra, "LINE_WEBHOOK_PATH", "webhook_path", "path", default=DEFAULT_PATH)))
        self._api_base = str(_env_or_extra(extra, "LINE_API_BASE", "api_base", default=LINE_API_BASE)).rstrip("/")
        self._data_api_base = str(_env_or_extra(extra, "LINE_DATA_API_BASE", "data_api_base", default=LINE_DATA_API_BASE)).rstrip("/")
        self._max_media_bytes = int(
            _safe_float(_env_or_extra(extra, "LINE_MEDIA_MAX_MB", "media_max_mb", default=10.0), 10.0)
            * 1024
            * 1024
        )
        self._max_body_bytes = _safe_int(
            _env_or_extra(extra, "LINE_MAX_BODY_BYTES", "max_body_bytes", default=1_048_576),
            1_048_576,
        )
        self._reply_tokens: Dict[str, tuple[str, float]] = {}
        self._chat_names: Dict[str, str] = {}
        self._session: Optional["ClientSession"] = None
        self._runner: Optional["web.AppRunner"] = None

    @property
    def name(self) -> str:
        return "LINE"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "missing_dependency",
                "aiohttp is required for the LINE platform",
                retryable=False,
            )
            return False
        if not self.channel_access_token or not self.channel_secret:
            self._set_fatal_error(
                "missing_credentials",
                "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET are required",
                retryable=False,
            )
            return False

        if not self._acquire_platform_lock(
            "line",
            f"{self._host}:{self._port}{self._path}",
            "LINE webhook listener",
        ):
            return False

        if self._is_port_in_use():
            self._release_platform_lock()
            self._set_fatal_error(
                "port_in_use",
                f"LINE webhook port {self._port} is already in use",
                retryable=False,
            )
            return False

        timeout = ClientTimeout(total=30)
        self._session = ClientSession(timeout=timeout)

        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get(self._path, self._handle_verify)
        app.router.add_post(self._path, self._handle_webhook)

        try:
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self._host, self._port)
            await site.start()
        except Exception as exc:
            await self.disconnect()
            self._set_fatal_error("connect_failed", str(exc), retryable=True)
            return False

        self._mark_connected()
        logger.info("[line] Listening on %s:%d%s", self._host, self._port, self._path)
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._session:
            await self._session.close()
            self._session = None
        self._release_platform_lock()
        self._mark_disconnected()
        logger.info("[line] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        text = self.format_message(content).strip()
        if not text:
            return SendResult(success=True, message_id="empty")
        messages = [{"type": "text", "text": chunk} for chunk in self.truncate_message(text, self.MAX_MESSAGE_LENGTH)]
        return await self._send_line_messages(chat_id, messages)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not chat_id or not self._session:
            return
        try:
            await self._post_line_json(
                "/v2/bot/chat/loading/start",
                {"chatId": chat_id, "loadingSeconds": 20},
            )
        except Exception as exc:
            logger.debug("[line] loading indicator failed for %s: %s", chat_id, exc)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not image_url.startswith(("http://", "https://")):
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata)

        messages: list[dict[str, Any]] = [
            {
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url,
            }
        ]
        if caption:
            messages.append({"type": "text", "text": self.format_message(caption)[: self.MAX_MESSAGE_LENGTH]})
        return await self._send_line_messages(chat_id, messages)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await super().send_voice(chat_id, audio_path, caption, reply_to, **kwargs)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await super().send_video(chat_id, video_path, caption, reply_to, **kwargs)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await super().send_document(chat_id, file_path, caption, file_name, reply_to, **kwargs)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": self._chat_names.get(chat_id, chat_id),
            "type": "dm" if chat_id.startswith("U") else "group",
            "chat_id": chat_id,
        }

    def format_message(self, content: str) -> str:
        """LINE text messages do not render markdown consistently."""
        text = str(content or "")
        text = re.sub(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", text)
        text = re.sub(r"(^|\s)([*_]{1,2})([^*_]+)\2(?=\s|$)", r"\1\3", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
        return text.strip()

    async def _send_line_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> SendResult:
        if not self.channel_access_token:
            return SendResult(success=False, error="LINE access token not configured")
        if not self._session:
            return SendResult(success=False, error="LINE adapter is not connected", retryable=True)
        if not messages:
            return SendResult(success=True, message_id="empty")

        reply_token = self._take_reply_token(chat_id)
        try:
            if reply_token:
                reply_batch = messages[:5]
                await self._post_line_json(
                    "/v2/bot/message/reply",
                    {"replyToken": reply_token, "messages": reply_batch},
                )
                for chunk in self._message_chunks(messages[5:]):
                    await self._push_messages(chat_id, chunk)
            else:
                for chunk in self._message_chunks(messages):
                    await self._push_messages(chat_id, chunk)
        except Exception as exc:
            if reply_token:
                logger.warning("[line] reply failed for %s; falling back to push: %s", chat_id, exc)
                try:
                    for chunk in self._message_chunks(messages):
                        await self._push_messages(chat_id, chunk)
                except Exception as push_exc:
                    return SendResult(success=False, error=str(push_exc), retryable=True)
            else:
                return SendResult(success=False, error=str(exc), retryable=True)

        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def _push_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        await self._post_line_json(
            "/v2/bot/message/push",
            {"to": chat_id, "messages": messages},
        )

    @staticmethod
    def _message_chunks(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [messages[i : i + 5] for i in range(0, len(messages), 5)]

    def _take_reply_token(self, chat_id: str) -> Optional[str]:
        token_data = self._reply_tokens.pop(chat_id, None)
        if not token_data:
            return None
        token, stored_at = token_data
        if time.monotonic() - stored_at <= REPLY_TOKEN_TTL_SECONDS:
            return token
        return None

    async def _post_line_json(self, path: str, payload: dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError("LINE adapter is not connected")
        url = f"{self._api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.channel_access_token}",
            "Content-Type": "application/json",
        }
        async with self._session.post(url, headers=headers, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"LINE API {resp.status}: {text}")
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "line", "path": self._path})

    async def _handle_verify(self, request: "web.Request") -> "web.Response":
        return web.Response(text="ok")

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        signature = request.headers.get("X-Line-Signature") or request.headers.get("x-line-signature")

        if not signature:
            if self._is_empty_verification_payload(body):
                return web.Response(text="ok")
            return web.Response(status=400, text="missing X-Line-Signature")

        if not verify_signature(body, self.channel_secret, signature):
            return web.Response(status=401, text="invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid JSON")

        events = payload.get("events") or []
        for event in events:
            task = asyncio.create_task(self._process_event(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return web.Response(text="ok")

    @staticmethod
    def _is_empty_verification_payload(body: bytes) -> bool:
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            return False
        events = parsed.get("events")
        return isinstance(events, list) and not events

    async def _process_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "postback":
            await self._handle_postback_event(event)
        elif event_type in {"follow", "unfollow", "join", "leave"}:
            logger.info("[line] %s event from %s", event_type, self._source_chat(event)[0])
        else:
            logger.debug("[line] ignoring event type %r", event_type)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        message = event.get("message") or {}
        msg_type = str(message.get("type") or "")
        message_id = str(message.get("id") or "")
        chat_id, chat_type, user_id, source_type = self._source_chat(event)
        self._store_reply_token(chat_id, event.get("replyToken"))

        await self.send_typing(chat_id)

        text = ""
        media_urls: list[str] = []
        media_types: list[str] = []
        event_message_type = MessageType.TEXT

        if msg_type == "text":
            text = str(message.get("text") or "")
        elif msg_type in {"image", "video", "audio", "file"}:
            cached_path, mime_type = await self._download_content(message_id, msg_type, message)
            if cached_path:
                media_urls.append(cached_path)
                media_types.append(mime_type or msg_type)
            text = self._media_text(msg_type, message)
            event_message_type = {
                "image": MessageType.PHOTO,
                "video": MessageType.VIDEO,
                "audio": MessageType.AUDIO,
                "file": MessageType.DOCUMENT,
            }[msg_type]
        elif msg_type == "location":
            title = message.get("title") or "Location"
            address = message.get("address") or ""
            latitude = message.get("latitude")
            longitude = message.get("longitude")
            text = f"{title}\n{address}\nlat={latitude}, lon={longitude}".strip()
            event_message_type = MessageType.LOCATION
        elif msg_type == "sticker":
            keywords = message.get("keywords") or []
            if isinstance(keywords, list) and keywords:
                text = "[sticker: " + ", ".join(str(item) for item in keywords) + "]"
            else:
                text = "[sticker]"
            event_message_type = MessageType.STICKER
        else:
            logger.debug("[line] unsupported message type %r", msg_type)
            return

        source = self.build_source(
            chat_id=chat_id,
            chat_name=self._chat_names.get(chat_id),
            chat_type=chat_type,
            user_id=user_id,
            user_name=await self._get_sender_name(source_type, chat_id, user_id),
            message_id=message_id,
        )
        await self.handle_message(
            MessageEvent(
                text=text,
                message_type=event_message_type,
                source=source,
                raw_message=event,
                message_id=message_id or None,
                media_urls=media_urls,
                media_types=media_types,
                timestamp=self._event_timestamp(event),
            )
        )

    async def _handle_postback_event(self, event: dict[str, Any]) -> None:
        data = str((event.get("postback") or {}).get("data") or "")
        if not data:
            return
        chat_id, chat_type, user_id, source_type = self._source_chat(event)
        self._store_reply_token(chat_id, event.get("replyToken"))
        source = self.build_source(
            chat_id=chat_id,
            chat_name=self._chat_names.get(chat_id),
            chat_type=chat_type,
            user_id=user_id,
            user_name=await self._get_sender_name(source_type, chat_id, user_id),
        )
        await self.handle_message(
            MessageEvent(
                text=data,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=event,
                timestamp=self._event_timestamp(event),
            )
        )

    def _source_chat(self, event: dict[str, Any]) -> tuple[str, str, Optional[str], str]:
        source = event.get("source") or {}
        source_type = str(source.get("type") or "user")
        user_id = source.get("userId")
        if source_type == "group":
            chat_id = str(source.get("groupId") or user_id or "unknown")
            return chat_id, "group", str(user_id) if user_id else None, source_type
        if source_type == "room":
            chat_id = str(source.get("roomId") or user_id or "unknown")
            return chat_id, "group", str(user_id) if user_id else None, source_type
        chat_id = str(user_id or "unknown")
        return chat_id, "dm", str(user_id) if user_id else None, source_type

    def _store_reply_token(self, chat_id: str, reply_token: Any) -> None:
        token = str(reply_token or "")
        if token:
            self._reply_tokens[chat_id] = (token, time.monotonic())

    async def _get_sender_name(self, source_type: str, chat_id: str, user_id: Optional[str]) -> Optional[str]:
        if not self._session or not user_id:
            return None
        if self._chat_names.get(f"user:{user_id}"):
            return self._chat_names[f"user:{user_id}"]
        try:
            if source_type == "group":
                path = f"/v2/bot/group/{chat_id}/member/{user_id}"
            elif source_type == "room":
                path = f"/v2/bot/room/{chat_id}/member/{user_id}"
            else:
                path = f"/v2/bot/profile/{user_id}"
            url = f"{self._api_base}{path}"
            async with self._session.get(
                url,
                headers={"Authorization": f"Bearer {self.channel_access_token}"},
            ) as resp:
                if resp.status >= 400:
                    return None
                body = await resp.json()
        except Exception:
            return None
        display_name = body.get("displayName")
        if display_name:
            self._chat_names[f"user:{user_id}"] = str(display_name)
            if source_type == "user":
                self._chat_names[chat_id] = str(display_name)
        return str(display_name) if display_name else None

    async def _download_content(
        self,
        message_id: str,
        msg_type: str,
        message: dict[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        if not message_id or not self._session:
            return None, None
        url = f"{self._data_api_base}/v2/bot/message/{message_id}/content"
        try:
            async with self._session.get(
                url,
                headers={"Authorization": f"Bearer {self.channel_access_token}"},
            ) as resp:
                if resp.status >= 400:
                    logger.warning("[line] media download failed for %s: HTTP %s", message_id, resp.status)
                    return None, None
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > self._max_media_bytes:
                    logger.warning("[line] media %s too large: %s bytes", message_id, content_length)
                    return None, None
                raw = await resp.read()
                if len(raw) > self._max_media_bytes:
                    logger.warning("[line] media %s too large after download: %d bytes", message_id, len(raw))
                    return None, None
                mime_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
        except Exception as exc:
            logger.warning("[line] media download failed for %s: %s", message_id, exc)
            return None, None

        filename = str(message.get("fileName") or "")
        ext = self._extension_for(msg_type, mime_type, filename)
        try:
            if msg_type == "image":
                return cache_image_from_bytes(raw, ext), mime_type or "image"
            if msg_type == "video":
                return cache_video_from_bytes(raw, ext), mime_type or "video"
            if msg_type == "audio":
                return cache_audio_from_bytes(raw, ext), mime_type or "audio"
            safe_filename = Path(filename or f"line-file{ext}").name
            return cache_document_from_bytes(raw, safe_filename), mime_type or "application/octet-stream"
        except Exception as exc:
            logger.warning("[line] failed caching media %s: %s", message_id, exc)
            return None, None

    @staticmethod
    def _extension_for(msg_type: str, mime_type: str, filename: str = "") -> str:
        if filename:
            suffix = Path(filename).suffix
            if suffix:
                return suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        if guessed:
            return guessed
        return {
            "image": ".jpg",
            "video": ".mp4",
            "audio": ".m4a",
            "file": ".bin",
        }.get(msg_type, ".bin")

    @staticmethod
    def _media_text(msg_type: str, message: dict[str, Any]) -> str:
        if msg_type == "file":
            file_name = message.get("fileName") or "file"
            return f"[file: {file_name}]"
        return f"[{msg_type}]"

    @staticmethod
    def _event_timestamp(event: dict[str, Any]) -> datetime:
        raw = event.get("timestamp")
        try:
            return datetime.fromtimestamp(float(raw) / 1000.0)
        except (TypeError, ValueError):
            return datetime.now()

    def _is_port_in_use(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                return sock.connect_ex(("127.0.0.1", self._port)) == 0
        except OSError:
            return False


def register(ctx) -> None:
    """Plugin entry point called by the Hermes plugin system."""
    ctx.register_platform(
        name="line",
        label="LINE",
        adapter_factory=lambda cfg: LineAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET"],
        install_hint="Install the messaging extra: pip install 'hermes-agent[messaging]'",
        allowed_users_env="LINE_ALLOWED_USERS",
        allow_all_env="LINE_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        pii_safe=True,
        emoji="LINE",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via LINE. LINE text messages render best as plain text. "
            "Avoid markdown tables and complex formatting; keep replies concise and "
            "split naturally when a response is long."
        ),
    )
