"""Tests for the LINE platform adapter plugin."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line_mod = load_plugin_adapter("line")

LineAdapter = _line_mod.LineAdapter
check_requirements = _line_mod.check_requirements
sanitize_line_reply = _line_mod.sanitize_line_reply
validate_config = _line_mod.validate_config
verify_signature = _line_mod.verify_signature
register = _line_mod.register


@pytest.fixture(autouse=True)
def clear_line_env(monkeypatch):
    for key in (
        "LINE_ACCESS_TOKEN",
        "LINE_CHANNEL_ACCESS_TOKEN",
        "LINE_CHANNEL_SECRET",
        "LINE_WEBHOOK_HOST",
        "LINE_WEBHOOK_PORT",
        "LINE_WEBHOOK_PATH",
        "LINE_API_BASE",
        "LINE_DATA_API_BASE",
        "LINE_MEDIA_MAX_MB",
        "LINE_MAX_BODY_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)


def _config(**extra):
    from gateway.config import PlatformConfig

    return PlatformConfig(enabled=True, extra=extra)


def test_verify_signature_valid():
    body = b'{"events":[]}'
    secret = "test-secret"
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode("ascii")
    assert verify_signature(body, secret, signature) is True


def test_verify_signature_invalid():
    assert verify_signature(b'{"events":[]}', "test-secret", "bad") is False


def test_init_from_env(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "env-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "env-secret")
    monkeypatch.setenv("LINE_WEBHOOK_PORT", "9901")
    monkeypatch.setenv("LINE_WEBHOOK_PATH", "line-hook")

    adapter = LineAdapter(_config())

    assert adapter.channel_access_token == "env-token"
    assert adapter.channel_secret == "env-secret"
    assert adapter._port == 9901
    assert adapter._path == "/line-hook"


def test_validate_config_from_extra():
    cfg = _config(channel_access_token="token", channel_secret="secret")
    assert validate_config(cfg) is True


def test_check_requirements_checks_dependencies():
    assert check_requirements() is True


def test_line_adapter_disables_message_editing():
    assert LineAdapter.SUPPORTS_MESSAGE_EDITING is False


def test_line_format_sanitizes_internal_self_disclosure():
    adapter = LineAdapter(_config(channel_access_token="token", channel_secret="secret"))

    text = adapter.format_message(
        "我是在 Hermes / FSA 的 S2 bridge 裡回答，透過 memory_core ontology 內部端點。"
    )

    for forbidden in ("Hermes", "FSA", "S2", "bridge", "memory_core", "ontology", "內部端點"):
        assert forbidden not in text
    assert "我這邊" in text


def test_line_sanitizer_preserves_general_engineering_endpoint_language():
    text = sanitize_line_reply("這個 API endpoint 要支援 POST，失敗時回 4xx。")

    assert "API endpoint" in text
    assert "POST" in text


def test_validate_config_from_env(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "secret")
    assert validate_config(_config()) is True


@pytest.mark.asyncio
async def test_text_message_event_dispatches_to_gateway():
    adapter = LineAdapter(_config(channel_access_token="token", channel_secret="secret"))
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_event(
        {
            "type": "message",
            "replyToken": "reply-token",
            "timestamp": 1_700_000_000_000,
            "source": {"type": "user", "userId": "U123"},
            "message": {"id": "m1", "type": "text", "text": "hello"},
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.message_id == "m1"
    assert event.source.platform.value == "line"
    assert event.source.chat_id == "U123"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "U123"
    assert adapter._reply_tokens["U123"][0] == "reply-token"


@pytest.mark.asyncio
async def test_group_message_uses_group_chat_id():
    adapter = LineAdapter(_config(channel_access_token="token", channel_secret="secret"))
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_event(
        {
            "type": "message",
            "source": {"type": "group", "groupId": "G123", "userId": "U123"},
            "message": {"id": "m1", "type": "text", "text": "hello group"},
        }
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "G123"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "U123"


@pytest.mark.asyncio
async def test_send_prefers_fresh_reply_token():
    adapter = LineAdapter(_config(channel_access_token="token", channel_secret="secret"))
    adapter._session = object()
    adapter._post_line_json = AsyncMock(return_value={})
    adapter._reply_tokens["U123"] = ("reply-token", time.monotonic())

    result = await adapter.send("U123", "hello")

    assert result.success is True
    adapter._post_line_json.assert_awaited_once()
    path, payload = adapter._post_line_json.await_args.args
    assert path == "/v2/bot/message/reply"
    assert payload["replyToken"] == "reply-token"
    assert payload["messages"] == [{"type": "text", "text": "hello"}]


@pytest.mark.asyncio
async def test_send_falls_back_to_push_when_reply_token_expired():
    adapter = LineAdapter(_config(channel_access_token="token", channel_secret="secret"))
    adapter._session = object()
    adapter._post_line_json = AsyncMock(return_value={})
    adapter._reply_tokens["U123"] = ("old-token", time.monotonic() - 100)

    result = await adapter.send("U123", "hello")

    assert result.success is True
    adapter._post_line_json.assert_awaited_once()
    path, payload = adapter._post_line_json.await_args.args
    assert path == "/v2/bot/message/push"
    assert payload["to"] == "U123"
    assert payload["messages"] == [{"type": "text", "text": "hello"}]


def test_register_adds_platform_entry():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "line"
    assert kwargs["label"] == "LINE"
    assert kwargs["allowed_users_env"] == "LINE_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "LINE_ALLOW_ALL_USERS"
