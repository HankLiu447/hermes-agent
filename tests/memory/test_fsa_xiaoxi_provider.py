"""Tests for the FSA/Xiaoxi memory provider."""

from __future__ import annotations

import httpx

from plugins.memory.fsa_xiaoxi import FsaXiaoxiConfig, FsaXiaoxiMemoryProvider


def _provider(**overrides) -> FsaXiaoxiMemoryProvider:
    config = {
        "enabled": True,
        "base_url": "http://fsa.test",
        "hank_line_user_ids": {"U123"},
        "timeout_seconds": 0.5,
        "diagnostics_enabled": False,
    }
    config.update(overrides)
    provider = FsaXiaoxiMemoryProvider(
        FsaXiaoxiConfig(**config)
    )
    provider.initialize(
        session_id="hermes-session",
        platform="line",
        user_id="U123",
        chat_id="U123",
        chat_type="dm",
        gateway_session_key="agent:main:line:dm:U123",
    )
    return provider


def test_prefetch_builds_safe_context(monkeypatch):
    provider = _provider()
    calls: list[str] = []

    def fake_post(path, payload):
        calls.append(path)
        if path == "/memory/codex-context":
            return {
                "memories": [
                    {
                        "content": "[user]: Hank 要把 FSA / Hermes 的 S1 S2 接成 LINE 任務流程。",
                    },
                    {
                        "content": "使用 FSA_CODEX_CHAT_API_KEY 呼叫本機服務，但不應暴露環境變數名稱。",
                    },
                    {
                        "content": "健康數據顯示睡眠 5.2h、位置在家，這行不能注入。",
                    },
                ]
            }
        if path == "/ontology/recall":
            return {
                "formatted": "- 客服整合專案 uses LINE OA\n- looki vision health 不該出現",
                "facts": [{"subject_name": "LINE", "predicate_name": "uses", "object_name": "Webhook"}],
            }
        raise AssertionError(path)

    def fake_get(path):
        calls.append(path)
        return {
            "/state/identity": {"tone": "professional", "mood": "focused", "energy": 0.2},
            "/state/active_context": {
                "topic": "工作請求與 LINE 接入",
                "summary": "（S1 認知迴圈運行中）",
            },
            "/state/open_tasks": [{"desc": "LINE webhook 切到任務窗口", "status": "open"}],
            "/state/immediate_narrative": ["睡眠、心率、位置這種即時狀態不能注入"],
        }[path]

    monkeypatch.setattr(provider, "_post_json", fake_post)
    monkeypatch.setattr(provider, "_get_json", fake_get)

    context = provider.prefetch("LINE webhook")

    assert "/ontology/health" not in calls
    assert "共享近況" in context
    assert "任務脈絡" in context
    assert "相關背景" in context
    assert "LINE webhook" in context
    for forbidden in ("FSA", "FSA_CODEX_CHAT_API_KEY", "Hermes", "S1", "S2", "健康", "睡眠", "位置", "looki", "vision"):
        assert forbidden not in context
    assert provider.last_recall_diagnostics["effective_depth"] == "fast"
    assert "memory_recall" not in provider.last_recall_diagnostics["calls"]


def test_prefetch_offline_returns_empty(monkeypatch):
    provider = _provider()

    def fail_post(path, payload):
        raise httpx.ConnectError("offline")

    def fail_get(path):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(provider, "_post_json", fail_post)
    monkeypatch.setattr(provider, "_get_json", fail_get)

    assert provider.prefetch("anything") == ""
    assert provider.last_recall_diagnostics["degraded"] is True


def test_auto_depth_runs_deep_recall_for_memory_question(monkeypatch):
    provider = _provider(recall_depth="auto", deep_recall_limit=2)
    calls: list[tuple[str, object]] = []

    def fake_post(path, payload):
        calls.append((path, payload))
        if path == "/memory/codex-context":
            return {"memories": [{"content": "最近正在把 LINE 任務流程接好。"}], "count": 1}
        if path == "/ontology/recall":
            return {"facts": [], "count": 0}
        raise AssertionError(path)

    def fake_get(path):
        calls.append((path, None))
        return {}

    def fake_get_params(path, params, timeout=None):
        calls.append((path, params))
        return {
            "results": [
                {
                    "memory": {
                        "content": "Hank 上次要求任務窗口必要時要做更仔細的脈絡整理。",
                        "memory_type": "episodic",
                        "source": "s2_conversation",
                    },
                    "score": 0.88,
                },
                {
                    "memory": {
                        "content": "健康與位置資料不能進模型。",
                        "memory_type": "episodic",
                        "source": "s1_dmn",
                    },
                    "score": 0.95,
                },
            ],
            "count": 2,
        }

    monkeypatch.setattr(provider, "_post_json", fake_post)
    monkeypatch.setattr(provider, "_get_json", fake_get)
    monkeypatch.setattr(provider, "_get_json_with_params", fake_get_params)

    context = provider.prefetch("妳現在記得什麼？")

    assert ("memory_recall" in provider.last_recall_diagnostics["calls"])
    assert provider.last_recall_diagnostics["effective_depth"] == "deep"
    assert any(call[0] == "/memory/recall" for call in calls)
    assert "更相關的過往脈絡" in context
    assert "更仔細的脈絡整理" in context
    assert "健康" not in context
    assert "位置" not in context


def test_auto_depth_keeps_normal_task_fast(monkeypatch):
    provider = _provider(recall_depth="auto")

    def fake_post(path, payload):
        return {"memories": [{"content": "LINE webhook 已接到任務工作區。"}]} if path == "/memory/codex-context" else {}

    def fake_get(path):
        return {}

    def fail_deep(path, params, timeout=None):
        raise AssertionError("normal task should not run deep recall")

    monkeypatch.setattr(provider, "_post_json", fake_post)
    monkeypatch.setattr(provider, "_get_json", fake_get)
    monkeypatch.setattr(provider, "_get_json_with_params", fail_deep)

    context = provider.prefetch("幫我確認 LINE webhook")

    assert provider.last_recall_diagnostics["effective_depth"] == "fast"
    assert "memory_recall" not in provider.last_recall_diagnostics["calls"]
    assert "共享近況" in context


def test_deep_recall_timeout_degrades_to_fast_context(monkeypatch):
    provider = _provider(recall_depth="deep")

    def fake_post(path, payload):
        return {"memories": [{"content": "安全上下文仍然可用。"}]} if path == "/memory/codex-context" else {}

    def fake_get(path):
        return {}

    def fail_deep(path, params, timeout=None):
        raise httpx.TimeoutException("slow recall")

    monkeypatch.setattr(provider, "_post_json", fake_post)
    monkeypatch.setattr(provider, "_get_json", fake_get)
    monkeypatch.setattr(provider, "_get_json_with_params", fail_deep)

    context = provider.prefetch("之前我們說過什麼？")

    assert "安全上下文仍然可用" in context
    assert provider.last_recall_diagnostics["effective_depth"] == "deep"
    assert provider.last_recall_diagnostics["degraded"] is True
    assert provider.last_recall_diagnostics["calls"]["memory_recall"]["ok"] is False


def test_group_does_not_inject_private_context(monkeypatch):
    provider = _provider()
    provider.initialize(
        session_id="hermes-session",
        platform="line",
        user_id="U123",
        chat_id="G123",
        chat_type="group",
    )

    def fail_post(path, payload):
        raise AssertionError("prefetch should not call FSA for group private context")

    monkeypatch.setattr(provider, "_post_json", fail_post)

    assert provider.prefetch("近期任務") == ""


def test_sync_turn_payload_shape(monkeypatch):
    provider = _provider()
    payloads: list[dict] = []

    def fake_post(path, payload):
        payloads.append({"path": path, "payload": payload})
        return {"status": "ok"}

    monkeypatch.setattr(provider, "_post_json", fake_post)

    provider.sync_turn("請幫我確認 LINE", "我這邊已確認。", session_id="hermes-session")
    assert provider._last_sync_thread is not None
    provider._last_sync_thread.join(timeout=1.0)

    assert payloads[0]["path"] == "/memory/ingest"
    payload = payloads[0]["payload"]
    assert payload["user_message"] == "請幫我確認 LINE"
    assert payload["assistant_message"] == "我這邊已確認。"
    assert payload["session_id"] == "main"
    assert payload["scope"] == "codex"
    assert payload["sender_id"] == "line:U123"
    assert payload["source_channel"] == "hermes_line"
    assert payload["metadata"]["trigger_source"] == "hermes:line"
    assert payload["metadata"]["async_memory_extraction"] is True
    assert payload["metadata"]["hermes_session_key"] == "agent:main:line:dm:U123"
    assert payload["metadata"]["platform"] == "line"
    assert payload["metadata"]["chat_type"] == "dm"
    assert payload["metadata"]["task_mode"] is True
