"""Tests for the FlySuiteFOS graph learning plugin."""

from __future__ import annotations

import json

from plugins import fos_graph_learning as plugin


FOS_TOOL = "mcp_flysuite_fos_graph_fos_query_object_set"


def setup_function():
    plugin._reset_for_tests()


def test_transform_large_fos_result_compacts(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("FOS_GRAPH_LEARNING_SYNC_ENABLED", "false")
    monkeypatch.setenv("FOS_GRAPH_LEARNING_MAX_TOOL_RESULT_CHARS", "4000")
    rows = [{"sku": f"SKU-{idx}", "qty": idx} for idx in range(200)]
    result = json.dumps({"result": json.dumps({"type": "product", "mode": "list", "rows": rows})})

    rewritten = plugin.on_transform_tool_result(tool_name=FOS_TOOL, result=result)

    assert rewritten is not None
    payload = json.loads(rewritten)
    assert payload["truncated_by"] == "fos_graph_learning"
    assert payload["result"]["rows"]["count"] == 200
    assert len(payload["result"]["rows"]["preview"]) == 12


def test_learning_roundtrip_builds_future_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("FOS_GRAPH_LEARNING_SYNC_ENABLED", "false")
    result = json.dumps(
        {
            "result": json.dumps(
                {
                    "type": "product",
                    "mode": "aggregate",
                    "rows": [{"count": 164}],
                    "lineage": {"entityType": "product", "views": ["entities", "current_claims"]},
                }
            )
        }
    )

    plugin.on_post_tool_call(
        tool_name=FOS_TOOL,
        args={"spec": {"from": {"type": "product"}, "aggregate": [{"fn": "count", "as": "count"}]}},
        result=result,
        session_id="s1",
        duration_ms=52,
    )
    plugin.on_post_llm_call(
        session_id="s1",
        user_message="FOS 裡有多少商品？",
        assistant_response="目前 product 有 164 筆。",
        platform="line",
        model="test-model",
    )

    context = plugin.on_pre_llm_call(user_message="幫我查 FOS 商品和庫存關係")

    assert context is not None
    assert "FOS 圖譜查詢工作方式" in context["context"]
    assert "FOS object-set 查詢" in context["context"]
    assert "fos_query_object_set" not in context["context"]
    assert "product" in context["context"]
    assert (tmp_path / "hermes" / "plugins" / "fos_graph_learning" / "learned_context.jsonl").exists()


def test_non_fos_query_does_not_inject_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    assert plugin.on_pre_llm_call(user_message="今天幫我整理一般備忘錄") is None


def test_sanitizer_removes_database_url_and_pii():
    compact = plugin._compact_payload(
        {
            "databaseUrl": "postgres://user:pass@localhost/db",
            "customer_email": "user@example.com",
            "summary": {"entityTypes": 47, "relationPaths": 60},
        }
    )

    assert compact["summary"]["entityTypes"] == 47
    assert "databaseUrl" not in compact
    assert "user@example.com" not in json.dumps(compact)


def test_sync_to_fsa_posts_safe_memory_payload(monkeypatch):
    posted = {}

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, timeout):
            posted["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            posted["url"] = url
            posted["json"] = json
            return Response()

    monkeypatch.setattr(plugin.httpx, "Client", Client)
    state = plugin.FosGraphLearningState()
    config = plugin.FosGraphLearningConfig(
        fsa_base_url="http://fsa.local",
        fsa_timeout_seconds=1.5,
    )
    record = {
        "session_id": "line-session",
        "platform": "line",
        "user_message": "查 FOS 商品",
        "assistant_summary": "結果來自 http://127.0.0.1:5434，user@example.com 已清洗。",
        "observations": [
            {
                "note": (
                    "FOS object-set 查詢(from=product) returned 2 rows; "
                    "contact user@example.com via http://localhost:5434."
                )
            }
        ],
    }

    state._sync_to_fsa(config, record)

    assert posted["timeout"] == 1.5
    assert posted["url"] == "http://fsa.local/memory/ingest"
    payload = posted["json"]
    assert payload["session_id"] == "fos_graph"
    assert payload["scope"] == "fos_graph"
    assert payload["source_channel"] == plugin.FOS_SOURCE_CHANNEL
    assert payload["metadata"]["trigger_source"] == "hermes:fos_graph"
    assert payload["metadata"]["hermes_session_id"] == "line-session"
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "127.0.0.1" not in dumped
    assert "localhost" not in dumped
    assert "user@example.com" not in dumped
    assert "[email]" in dumped
